import os
import re
import gc
import copy
import time

import cv2
from PIL import Image
import pandas as pd
import numpy as np
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from collections import defaultdict

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose

from transformers import AutoImageProcessor, VivitImageProcessor, VivitForVideoClassification, VivitConfig, VideoMAEModel, VideoMAEConfig, VideoMAEForVideoClassification

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

from utils.configuration import CONFIG

CONFIG = CONFIG()


def sort(entry):
    # https://stackoverflow.com/a/2669120/7636462
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(entry, key = alphanum_key)

def getPaths(path):
    paths = glob(path + '/*.jpg')
    sortedPaths = sort(paths)
    maxWindowStart = len(sortedPaths) #- Config['MAX_FRAMES']
    start = 0 # np.random.randint(1, maxWindowStart)
    paths = sortedPaths#[start:Config['MAX_FRAMES']]

    return paths

def load_data():
    freeway_df = pd.read_csv("./Data/freeway_train.csv")
    road_df = pd.read_csv("./Data/road_train.csv")
    freeway_df['file_name'] = freeway_df['file_name'].apply(lambda x: os.path.join("Data/freeway/train", x))
    road_df['file_name'] = road_df['file_name'].apply(lambda x: os.path.join("Data/road/train", x))

    data_df = freeway_df.append(road_df).reset_index(drop=True)
    data_df['n_frames'] = data_df['file_name'].apply(lambda x: len(getPaths(x)))
    # data_df = data_df[data_df['n_frames']>=CONFIG.max_frames].reset_index(drop=True)
    
    return data_df

def split_data(data_df):
    train_df, valid_df = train_test_split(data_df, test_size=0.2, random_state=CONFIG.seed)
    data_df['split'] = 'train'
    data_df.loc[valid_df.index, 'split'] = 'val'
    
    return train_df, valid_df

##################################

def load_model(model_name="ViViT", cfg=CONFIG):
    
    if model_name == "ViViT" :
        cfg.model_name = "google/vivit-b-16x2-kinetics400"
        image_processor = VivitImageProcessor.from_pretrained(cfg.model_name)
        model_configuration = VivitConfig()
        model_configuration.num_frames = cfg.max_frames  # Set number of frames
        model_configuration.num_labels = cfg.num_classes  # Set number of classes
        
        model = VivitForVideoClassification.from_pretrained(
            pretrained_model_name_or_path=cfg.model_name, 
            config=model_configuration, 
            ignore_mismatched_sizes=True)
        
    elif model_name == "ViMAE":
        cfg.model_name = "MCG-NJU/videomae-base"
        image_processor = AutoImageProcessor.from_pretrained(cfg.model_name)
        model_configuration = VideoMAEConfig()
        model_configuration.num_frames = cfg.max_frames  # Set number of frames
        model_configuration.num_labels = cfg.num_classes  # Set number of classes
        model = VideoMAEForVideoClassification.from_pretrained(
            pretrained_model_name_or_path=cfg.model_name, 
            config=model_configuration, 
            ignore_mismatched_sizes=True)
        
    
    model.classifier = nn.Linear(model.classifier.in_features, cfg.num_classes)  # Adjust classifier
    
    model = model.to(cfg.device)
    
    return image_processor, model

class VideoDataset(Dataset):
    def __init__(self, df, image_processor, transform=None, is_test=False, cfg=CONFIG):
        self.df = df
        self.cfg = cfg
        # self.root_dir = "/kaggle/input/2nd-ava-challenge-ieee-mipr-2024/2nd_AVA_Dataset_2ed/freeway/train"
        self.paths = self.df['file_name'].values
        self.labels = self.df['risk'].values 
        self.image_processor = image_processor
        self.is_test = is_test
        self.transform = transform
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        frames = self.getFrames(path)
        
        if self.is_test:
            return {
                "pixel_values": frames
            }
        else:
            label = self.labels[idx]
            return {
                "pixel_values": frames,
                "labels": label
            }
        
    def getFrames(self, folder_path):
        paths = self.getPaths(folder_path)
        
        frames = []
        for img_path in paths:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(self.cfg.img_size, Image.LANCZOS)
#             if self.transform:
#                 img = self.transform(img)#*255
            
            frames.append(img)

#         if self.transform:
#             frames = [self.transform(frame) for frame in frames]
        frames = self.image_processor(frames, return_tensors="pt")['pixel_values'][0]
        return frames
            
            
    def getPaths(self, path):
        # path = os.path.join(self.root_dir, path)
        paths = glob(path + '/*.jpg')
        sortedPaths = self.sort(paths)
        paths = sortedPaths[-CONFIG.max_frames:]
        return paths
        
    def sort(self, entry):
        # https://stackoverflow.com/a/2669120/7636462
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    
        return sorted(entry, key = alphanum_key)

def get_data_loader(data_df, image_processor):
    train_df, valid_df = split_data(data_df=data_df) 
    train_dataset = VideoDataset(train_df.reset_index(drop=True), image_processor=image_processor)
    valid_dataset = VideoDataset(valid_df.reset_index(drop=True), image_processor=image_processor)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG.train_batch_size, shuffle=False)
    
    return train_loader, valid_loader

############################
def get_score(y_true, y_pred):
    f1score = f1_score(y_true, y_pred, average='micro')
    acc = accuracy_score(y_true, y_pred)
    
    return {
        'f1_score' : f1score,
        'accuracy' : acc
    }
 
def get_optimizer(parameters, cfg=CONFIG):
    return optim.AdamW(parameters, lr=cfg.learning_rate)

def get_scheduler(optimizer):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-7)
    
################################
def train_one_epoch(model, train_dataloader, valid_dataloader, optimizer, scheduler=None, epoch=1, device='cpu'):
    
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    
    steps = len(train_dataloader)
    bar = tqdm(enumerate(train_dataloader), total= len(train_dataloader), desc='Train ')
    
    for step, data in bar:
        # sending data to cpu or gpu if cuda avaiable.
        pixel_values = data["pixel_values"].to(device, dtype=torch.float)
        labels = data["labels"].to(device, dtype=torch.long)
        
        batch_size = pixel_values.size(0)
        
        #computing model output
        outputs = model(pixel_values=pixel_values, labels=labels)
#         #loss calcuation
#         loss = nn.CrossEntropyLoss()(outputs.logits, labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
        
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        train_epoch_loss= running_loss / dataset_size
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        bar.set_postfix(Epoch= epoch, 
                        Train_Loss = f'{train_epoch_loss:0.4f}',
                        LR= optimizer.param_groups[0]['lr'],
                        gpu_mem= f'{mem: 0.2f} GB'
                       )
    torch.cuda.empty_cache()
    gc.collect()

    ###################################################################
    ### Validation

    # model.eval()
    
    dataset_size= 0
    running_loss= 0.0
    
    preds= []
    true_labels= []

    bar= tqdm(enumerate(valid_dataloader), total= len(valid_dataloader), desc='Valid ')
    
    for step, data in bar:
        pixel_values = data["pixel_values"].to(device, dtype=torch.float)
        labels = data["labels"].to(device, dtype=torch.long)
        
        batch_size = pixel_values.size(0)
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, labels=labels)
        
        logits = outputs.logits
        loss = outputs.loss
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        valid_epoch_loss = running_loss / dataset_size
        
        preds.append(logits.argmax(axis=1).to('cpu').numpy())
        true_labels.append(labels.to('cpu').numpy())
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        bar.set_postfix(Epoch= epoch, 
                        Valid_Loss = valid_epoch_loss,
                        gpu_memory=f'{mem:0.2f} GB',
                       )
    
    true_labels = np.concatenate(true_labels)
    predictions = np.concatenate(preds)
    
    scores = get_score(y_true=true_labels, y_pred=predictions)

    
    torch.cuda.empty_cache()
    gc.collect()
    return train_epoch_loss, valid_epoch_loss, scores

def training_loop(model, train_dataloader, valid_dataloader, optimizer, scheduler, num_epochs, cfg=CONFIG, patience=3):
    
    start = time.time()
    best_score = - np.inf
    trigger_times = 0 # for early stoping

    history = defaultdict(list)

    for epoch in range(CONFIG.start_epoch, CONFIG.start_epoch + num_epochs + 1):
        gc.collect()
        
        train_epoch_loss, valid_epoch_loss, scores = train_one_epoch(
            model, 
            train_dataloader, 
            valid_dataloader, 
            optimizer, 
            scheduler=scheduler, 
            epoch=epoch, 
            device=CONFIG.device)
        
        history['train_loss'].append(train_epoch_loss)
        history['valid_loss'].append(valid_epoch_loss)
        history['F1_score'].append(scores['f1_score'])
        history['Accuracy'].append(scores['accuracy'])
        
        score = scores['f1_score']
        if score >= best_score:
            
            trigger_times= 0 #for early stop
            print(f"{b_}Validation Score Improved ({best_score :.6f} ---> {score :.6f}){sr_}")
            print(f"{b_}Validation Accuracy: {scores['accuracy'] :.4f} {sr_}")
            print(f"{b_}Validation F1 Socre: {score :.4f} {sr_}")
            
            best_score = score
            accuracy = scores['accuracy']
            # copy and save model
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"best_model.bin"
            torch.save(model.state_dict(), PATH)
            print(f"{y_} Model Saved to {PATH} {sr_}")

        
        else:
            trigger_times += 1
            
            if trigger_times >= patience:
                print(f"{'='*15} Early stoping {'='*15} \n")
                break

        
    end= time.time()
    time_elapsed= end - start
    
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    
    
    # load best model weights
    model.load_state_dict(best_model_wts)


    return model, history, best_score, accuracy
