
# # Imports
import os
from tqdm import tqdm 

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print('running tf: ', tf.__version__)

# from classification_models.keras import Classifiers
# SeResNeXT50, preprocess_fn = Classifiers.get('seresnext50')

# # Config

DEBUG = False

NUM_CLASSES = 2
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 30

MAX_SEQ_LENGTH = 16
NUM_FEATURES = 1280

AUTOTUNE = tf.data.experimental.AUTOTUNE

# # Dataloaders

def read_frames_from_folder(folder_path):
    folder_path = folder_path.numpy().decode('utf-8')
    # Get list of all frame files
    frame_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')])
    
    # Take the last num_frames frames if there are more frames
    if len(frame_files) > MAX_SEQ_LENGTH:
        frame_files = frame_files[-MAX_SEQ_LENGTH:]
    
    frames = []
    for frame_file in frame_files:
        img = image.load_img(frame_file, target_size=(IMG_SIZE, IMG_SIZE), interpolation='bicubic',)
        img_array = image.img_to_array(img)
        
        # preprocess/normalize - had to do it here for older tf versions
        # img_array = preprocess_fn(img_array)
        
        frames.append(img_array)
    
    frames = tf.convert_to_tensor(frames, dtype=tf.float32)

    # If fewer than num_frames, pad with zeros
    if len(frames) < MAX_SEQ_LENGTH:
        padding = tf.zeros((MAX_SEQ_LENGTH - len(frames), IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
        frames = tf.concat([padding, frames], axis=0)
        
    return frames

def load_video_from_folder(file_path, label = None):
    video = tf.py_function(func=read_frames_from_folder, inp=[file_path], Tout=tf.float32)
    video.set_shape([MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3])
    
    if label is None:
        return video
        
    label = tf.cast(label, dtype=tf.int32)
    label = tf.one_hot(label, depth=NUM_CLASSES)
    return video, label

def create_dataset(file_paths, labels, shuffle_buffer_size=None):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    # shuffle training data
    if shuffle_buffer_size:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    
    dataset = dataset.map(load_video_from_folder, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


# # Model

def FeatureExtractor():
    base_model = tf.keras.applications.EfficientNetV2M(
        weights='imagenet', 
        include_top=False,  
        input_shape=(IMG_SIZE, IMG_SIZE, 3), 
        pooling=None,
    )

#     base_model = SeResNeXT50(input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet', include_top=False)
#     preprocess_layer = tf.keras.layers.Lambda(lambda x: preprocess_fn(x))
    
    base_model.trainable = True

    model = tf.keras.models.Sequential([
        tf.keras.Input((IMG_SIZE, IMG_SIZE, 3)),
#         preprocess_layer,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
    ])
    
    return model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Layer Normalization and Multi-Head Attention
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)  
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)  
    x = tf.keras.layers.Dropout(dropout)(x)  
    res = tf.keras.layers.Add()([x, inputs])  

    # Feed Forward Part
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)  
    x = tf.keras.layers.Dense(ff_dim, activation="relu")(x)  
    x = tf.keras.layers.Dropout(dropout)(x)  
    x = tf.keras.layers.Dense(inputs.shape[-1])(x)  
    return tf.keras.layers.Add()([x, res])  

def SeqModel():
    inputs = tf.keras.Input((MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3))
    feature_extractor = FeatureExtractor()
    
    time_wrapper = tf.keras.layers.TimeDistributed(feature_extractor)(inputs)
    image_features = tf.keras.layers.Dense(256, activation="relu")(time_wrapper)


    head_size = 64
    num_heads = 4
    ff_dim = 256
    dropout = 0.2
    x = transformer_encoder(image_features, head_size, num_heads, ff_dim, dropout)  
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    outputs = tf.keras.layers.Dense(2, activation='softmax',)(x)
    
    return tf.keras.Model(inputs, outputs)

tf.keras.backend.clear_session() 
model = SeqModel()

# # Inference

WEIGHT_PATH = f'./e2e-cnn-transformer.weights.h5'
model.load_weights(WEIGHT_PATH)

def create_test_dataset(file_paths):
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(load_video_from_folder, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def run_inference(dataset, model):
    predictions = []
    for videos in dataset:
        preds = model.predict(videos, verbose = 0)
        predictions.extend(preds)
    return predictions

# Load the sample submission file
submission_df = pd.read_csv('/mnt/storage2/Sabbir/VideoClassification/Video_Classification_Pipeline/Data/sample_submission.csv')

if DEBUG:
    submission_df = submission_df[:5]

# Get test file names
test_files = submission_df['file_name'].values

# Paths to test data
freeway_test_path = '/mnt/storage2/Sabbir/VideoClassification/Video_Classification_Pipeline/Data/freeway/test'
road_test_path = '/mnt/storage2/Sabbir/VideoClassification/Video_Classification_Pipeline/Data/road/test'

# Create list of file paths
test_file_paths = []
for file_name in test_files:
    if 'road' in file_name:
        test_file_paths.append(os.path.join(road_test_path, file_name))
    elif 'freeway' in file_name:
        test_file_paths.append(os.path.join(freeway_test_path, file_name))
        
        
test_dataset = create_test_dataset(test_file_paths)

preds = run_inference(test_dataset, model)
positive_probs = [pred[1] for pred in preds]

submission_df['risk'] = positive_probs
submission_df.to_csv('submission.csv', index=False)

from datetime import datetime

print('notebook finished on: ', datetime.now())

