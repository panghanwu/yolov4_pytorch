from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import numpy as np
import torch
import csv
import os
 
# local modules
from models.yolov4 import YOLOv4
from utils import yolo_dataset, yolo_loss
import configs



def write_logs(fields:list, data:zip, title):
    fn = f'{title}.csv'
    with open(fn, 'w') as cf:
        writer = csv.writer(cf) 
        writer.writerow(fields)
        for d in data:
            writer.writerow(d)


# initialization for weights of model
def kaiming_init(model):
    if isinstance(model, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(model.weight.data)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)


def train_one_epoch(yolo_model, yolo_losses, train_loader, optimizer, device):
    
    yolo_model.train()  # training mode
    loss_list = list()  # loss container

    # set a batch with tqdm
    batch_iter = tqdm(
        iterable = enumerate(train_loader),
        desc = '[Train]',
        total = len(train_loader),
        postfix = dict
    )

    # for one batch
    for i, (imgs, tars) in batch_iter:
        # load data and set device
        inputs = imgs.to(device)  
        targets = [t.to(device) for t in tars]
        optimizer.zero_grad()  # set grad params zero
        
        bs = len(targets)
        
        yolo_heads = yolo_model(inputs)  # forward propagation
        
        # loss
        loss = 0
        for i in range(len(yolo_losses)):
            loss += yolo_losses[i](yolo_heads[i], targets)
        
        loss = loss / bs
        loss_list.append(loss.item())
        # backpropagation
        loss.backward()  
        optimizer.step()  # update weights

        # show info
        batch_iter.set_postfix(**{
            'loss': np.mean(loss_list),
            'lr': optimizer.param_groups[0]['lr']
        })

    batch_iter.close()
    return np.mean(loss_list)
    
    
    
def valid_one_epoch(yolo_model, yolo_losses, valid_loader, optimizer, device):
    with torch.no_grad():
        yolo_model.eval()  # evaluation mode
        loss_list = list()  # loss container

        # set a batch with tqdm
        batch_iter = tqdm(
            iterable = enumerate(valid_loader),
            desc = '[Valid]',
            total = len(valid_loader),
            postfix = dict
        )

        for i, (imgs, tars) in batch_iter:
            # load data and set device
            inputs = imgs.to(device)  
            targets = [t.to(device) for t in tars]
            bs = len(targets)
            
            yolo_heads = yolo_model(inputs)  # forward propagation

            loss = 0
            for i in range(len(yolo_losses)):
                loss += yolo_losses[i](yolo_heads[i], targets)

            loss = loss / bs
            loss_list.append(loss.item())

            # show info
            batch_iter.set_postfix(**{
                'loss': np.mean(loss_list)
            })

        batch_iter.close()
        return np.mean(loss_list)

        
    
# if __name__ == '__main__':

# configuration
config = configs.TrainingConfig()

print(f'{config.model_name} training mission start!')

# read yolo training data dictionary to list
with open(config.train_dict) as tf:
    yolo_lines = tf.readlines()

np.random.shuffle(yolo_lines)    

cut = int(config.valid_ratio*len(yolo_lines))
train_lines = yolo_lines[cut:]
valid_lines = yolo_lines[:cut]
    
    
# training/validation dataset
train_data = yolo_dataset.YOLODataset(
    train_lines,
    output_size = config.input_size,
    batch_size = config.batch_size,
    normalization = config.normalization,
    resize_mode = config.resize_mode,
    grayscale = config.grayscale,
    center_crop_ratio = config.center_crop_ratio,
    size_augmentation = config.size_aug,
    random_flip = config.random_flip,
    color_jitter = config.color_jitter,
    mosaic = config.mosaic
)

valid_data = yolo_dataset.YOLODataset(
    valid_lines,
    output_size = config.input_size,
    batch_size = config.batch_size,
    normalization = config.normalization,
    resize_mode = config.resize_mode,
    grayscale = config.grayscale,
    center_crop_ratio = config.center_crop_ratio,
    size_augmentation = None,
    random_flip = None,
    color_jitter = None,
    mosaic = False
)


# # training/validation dataloader
train_loader = DataLoader(
    train_data,
    batch_size = config.batch_size,
    shuffle = True,
    collate_fn = yolo_dataset.collator,
    num_workers = config.num_workers
)

valid_loader = DataLoader(
    valid_data,
    batch_size = config.batch_size,
    shuffle = True,
    collate_fn = yolo_dataset.collator,
    num_workers = config.num_workers
)


# model
yolo_model = YOLOv4(len(config.anchors), len(config.class_names)) 


# Kaiming initialization/pre-trained weights
if config.pretrained is None:
    yolo_model = yolo_model.apply(kaiming_init)
    print('Initialize model with Kai-Ming method.')
else:
    pretrained_weights = torch.load(config.pretrained)
    yolo_model.load_state_dict(pretrained_weights)
    print(f'Load model "{os.path.basename(config.pretrained)}" successfully.')
    
    
# set device
if config.cuda:
    device = 'cuda'
    yolo_model.to(device)
    
    print('Get CUDA device.')
else:
    device = 'cpu'
    yolo_model.to(device)
    print('Get CPU device.')
    
    
# initialize YOLO loss
yolo_losses = list()
for head_no in range(config.num_heads):
    yolo_losses.append(yolo_loss.YOLOLoss(
        anchors = config.anchors, 
        num_classes = len(config.class_names), 
        img_size = config.input_size,
        num_heads = config.num_heads,
        label_smoothing = config.label_smoothing,
        cuda = config.cuda
    ))
        
        
# TRAINING
# optimizer
optimizer = optim.Adam(yolo_model.parameters(), config.learning_rate)

if config.cos_lr:
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
else:
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)


# init params
epoch = config.initial_epoch
epoch_list = list()
train_losses = list()
valid_losses = list()
best_loss = 1e6
best_epoch = 0
title = f'{config.log_dir}{config.model_name}'
  
    
# phase of freezing yolo backbone
for param in yolo_model.backbone.parameters():
    param.requires_grad = False
    
    
for i in range(config.freeze_epochs):
    print(f'---Epoch {epoch}---')
    epoch_list.append(epoch)
    epoch += 1
    
    # training
    loss = train_one_epoch(yolo_model, yolo_losses, train_loader, optimizer, device)  
    train_losses.append(loss)

    # validation
    loss = valid_one_epoch(yolo_model, yolo_losses, valid_loader, optimizer, device)
    valid_losses.append(loss)

    lr_scheduler.step()  # learning rate scheduler step

    # save best model
    if valid_losses[-1] < best_loss:
        best_loss = valid_losses[-1]
        best_epoch = epoch_list[-1]
        torch.save(yolo_model.state_dict(), f'{title}.pth')

    # write the csv logs
    fields = ['Epoch', 'Training Loss', 'Validation Loss']
    data = zip(epoch_list, train_losses, valid_losses)
    write_logs(fields, data, title)

    
        
# phase of unfreezing
for param in yolo_model.parameters():
    param.requires_grad = True

    
for i in range(config.unfreeze_epochs):
    print(f'---Epoch {epoch}---')
    epoch_list.append(epoch)
    epoch += 1
    
    # training    
    loss = train_one_epoch(yolo_model, yolo_losses, train_loader, optimizer, device)  
    train_losses.append(loss)

    # validation
    loss = valid_one_epoch(yolo_model, yolo_losses, valid_loader, optimizer, device)
    valid_losses.append(loss)

    lr_scheduler.step()  # learning rate scheduler step

    # save best model
    if valid_losses[-1] < best_loss:
        best_loss = valid_losses[-1]
        best_epoch = epoch_list[-1]
        torch.save(yolo_model.state_dict(), f'{title}.pth')

    # write the csv logs
    fields = ['Epoch', 'Training Loss', 'Validation Loss']
    data = zip(epoch_list, train_losses, valid_losses)
    write_logs(fields, data, title)
    
# TODO txt log
# record anchors, config, and best epoch

# finish info
print(f'{config.model_name} training mission completed!')
print(f'Got the best validation loss {best_loss:.6f} at epoch {best_epoch}.')