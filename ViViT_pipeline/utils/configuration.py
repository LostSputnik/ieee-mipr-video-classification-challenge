import torch
class CONFIG:
    seed = 43
    img_size = (420, 420) #(224, 224)
    model_name = "ViMAE" #"ViViT" #"google/vivit-b-16x2-kinetics400" #'resnext50_32x4d'
    num_classes = 2
    
    max_frames = 8
    learning_rate = 2e-3
    
    num_epochs = 2
    start_epoch = 0
    train_batch_size = 6
    
    patience = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")