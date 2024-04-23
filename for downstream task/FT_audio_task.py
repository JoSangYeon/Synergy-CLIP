import os
import sys
import json
import random
import librosa
import argparse
import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from transformers import AutoProcessor, ASTModel

from model import Tri_CLIP
from config import *
from utils import *

import warnings
warnings.filterwarnings(action='ignore')

INIT_JSON = {
	"esc50": {
		"result": {
			"BASE": {},
			"LARGE": {}
		}
	},
    "urbansound8k": {
		"result": {
			"BASE": {},
			"LARGE": {}
		}
	},
    "voxceleb1": {
		"result": {
			"BASE": {},
			"LARGE": {}
		}
	}
}

MAX_ACC = -1
LOSS = 0

class ESC50Data(Dataset):
  def __init__(self, df, base_path, aud_processor, fold=0, mode='train'):
    self.df = df
    self.aud_processor = aud_processor
    self.files = []
    self.labels = []
    self.c2i={}
    self.i2c={}
    self.categories = sorted(df['category'].unique())
    for i, category in enumerate(self.categories):
      self.c2i[category]=i
      self.i2c[i]=category
    
    data = df[df.fold != fold] if mode=='train' else df[df.fold == fold]
    
    for filename,_,_,category,_,_,_ in tqdm(data.values):
        self.files.append(os.path.join(base_path, filename))
        self.labels.append(self.c2i[category])
        
    self.mode = mode

  def __len__(self):
    return len(self.files)
  
  def __getitem__(self, idx):
    ## audio ##
    wav, sr = librosa.load(self.files[idx])
    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    wav = audio_augment(wav, sr=16000, noise_rate=0.001, 
                        shift_rate=0.1, stretch_rate=0.15,
                        is_shuffle=True, apply_rate=0.333) if self.mode=='train' else wav
    audio = self.aud_processor(wav, sampling_rate=16000, return_tensors="pt") # Size(1024, 64)
    audio = audio.input_values[0]

    ## label ##
    label = torch.tensor(self.labels[idx]).long()

    return audio, label

class URBANSOUND8KData(Dataset):
  def __init__(self, df, base_path, aud_processor, fold=0, mode='train'):
    self.df = df
    self.aud_processor = aud_processor
    self.files = []
    self.labels = []
    self.c2i={}
    self.i2c={}
    self.categories = sorted(df['class'].unique())
    for i, category in enumerate(self.categories):
      self.c2i[category]=i
      self.i2c[i]=category
    
    data = df[df.fold != fold] if mode=='train' else df[df.fold == fold]
    
    for file_name,_,_,_,_,fold_num,_,category in tqdm(data.values):
        self.files.append(os.path.join(base_path, f'fold{fold_num}', file_name))
        self.labels.append(self.c2i[category])

    self.mode = mode

  def __len__(self):
    return len(self.files)
  
  def __getitem__(self, idx):
    ## audio ##
    wav, sr = librosa.load(self.files[idx])
    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    wav = audio_augment(wav, sr=16000, noise_rate=0.001, 
                        shift_rate=0.1, stretch_rate=0.15,
                        is_shuffle=True, apply_rate=0.333) if self.mode=='train' else wav
    audio = self.aud_processor(wav, sampling_rate=16000, return_tensors="pt") # Size(1024, 64)
    audio = audio.input_values[0]

    ## label ##
    label = torch.tensor(self.labels[idx]).long()

    return audio, label

class DownstreamTaskModel(nn.Module):
    def __init__(self, base_path, hidden_dim=768,
                 projection_dim=768, num_classes=10):
        super(DownstreamTaskModel, self).__init__()
        self.base_model = ASTModel.from_pretrained(base_path)
        self.projection_head = nn.Linear(hidden_dim, projection_dim, bias=False) 
        self.classifier = nn.Linear(projection_dim, num_classes)
                   
        projection_head_ckpt = torch.load(os.path.join(base_path, 'projection_head.tar'))
        self.projection_head.load_state_dict(projection_head_ckpt['model_state_dict'])

    def forward(self, x):
        output = self.base_model(x)  # Extract image features
        cls = output[1]
        cls = self.projection_head(cls)
        cls = self.classifier(cls)
        return cls

def get_FT_set(DATASET, IS_BASE, IS_CAPTIONED, fold=1, batch_size = 64):
    clip_config = CLIPConfig_BASE() if IS_BASE else CLIPConfig_LARGE()
    audio_model_path  = clip_config.audio_config.model_link

    if DATASET == 'esc50':
        # CIFAR-10 Data loading
        num_classes = 50

        df = pd.read_csv(os.path.join('data','ESC-50','esc50.csv'))
        base_path = os.path.join('.','data', 'ESC-50', 'audio')
        aud_processor = AutoProcessor.from_pretrained(audio_model_path)

        train_dataset = ESC50Data(df, base_path, aud_processor, fold=fold, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        valid_dataset = ESC50Data(df, base_path, aud_processor, fold=fold, mode='test')
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    elif DATASET == 'urbansound8k':
        # CIFAR-100 Data loading
        num_classes = 10

        base_path = os.path.join('data','UrbanSound8K')
        df = pd.read_csv(os.path.join(base_path, 'UrbanSound8K.csv'))
        aud_processor = AutoProcessor.from_pretrained(audio_model_path)

        train_dataset = URBANSOUND8KData(df, base_path, aud_processor, fold=fold, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        valid_dataset = URBANSOUND8KData(df, base_path, aud_processor, fold=fold, mode='test')
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        return None, None, None

    # Load the pre-trained Tri_CLIP model
    hidden_size = clip_config.audio_config.hidden_size
    projection_dim = clip_config.projection_dim
    save_path = os.path.join(f"CLIP_audio_model_{'base' if IS_BASE else 'large'}",
                             'caption' if IS_CAPTIONED else 'prompt')

    model = DownstreamTaskModel(save_path, hidden_dim=hidden_size, projection_dim=projection_dim, num_classes=num_classes)

    return train_loader, valid_loader, model

def train(model, device, train_loader, optimizer, criterion, accumulation_steps=16):
    model.train(); optimizer.zero_grad()

    loss_lst = []
    train_pbar = tqdm(train_loader, file=sys.stdout)
    for batch_idx, (data, target) in enumerate(train_pbar):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target) / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        loss_lst.append(loss.item() * accumulation_steps)
        train_pbar.set_postfix(loss='{:.4f}'.format(np.mean(loss_lst)))
    train_pbar.close()

    if batch_idx % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

def validate(model, device, valid_loader, criterion):
    model.eval()

    valid_loss = []
    correct = []

    valid_pbar = tqdm(valid_loader, file=sys.stdout)
    with torch.no_grad():
        for data, target in valid_pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            valid_loss.append(criterion(output, target).item())  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct.append(pred.eq(target.view_as(pred)).sum().item())

            valid_pbar.set_postfix(loss='{:.4f}, Acc={}/{}'.format(np.mean(valid_loss), np.sum(correct), len(valid_loader.dataset)))
        valid_pbar.close()

    total_loss = np.mean(valid_loss)
    accuracy   = 100. * np.sum(correct) / len(valid_loader.dataset)
    print(f'Validation set: Average loss: {total_loss:.4f}, Accuracy: {np.sum(correct)}/{len(valid_loader.dataset)}({accuracy:.2f}%)')
    return total_loss, accuracy

def save_metric(DATASET, IS_BASE, IS_CAPTIONED, SEED, total_loss, accuracy):
    with open("METRIC_AUD.json", "r") as f:
	    data = json.load(f)
    metric = {'loss':total_loss, 'accuracy':accuracy}
    data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"]["BASE" if IS_BASE else "LARGE"][f'Fold_{SEED}'] = metric

    temp = data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"]["BASE" if IS_BASE else "LARGE"]

    loss_lst = []
    acc_lst = []
    for k, v in temp.items():
        loss_lst.append(temp[k]['loss'])
        acc_lst.append(temp[k]['accuracy'])

    data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"][f'{"BASE" if IS_BASE else "LARGE"}_loss_mean'] = np.mean(loss_lst)
    data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"][f'{"BASE" if IS_BASE else "LARGE"}_loss_std'] = np.std(loss_lst)
    data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"][f'{"BASE" if IS_BASE else "LARGE"}_accuracy_mean'] = np.mean(acc_lst)
    data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"][f'{"BASE" if IS_BASE else "LARGE"}_accuracy_std'] = np.std(acc_lst)

    with open("METRIC_AUD.json", "w") as f:
        json.dump(data, f, indent='\t')

def main():
    global MAX_ACC, LOSS
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--SEED", type=int, default=17)
    parser.add_argument("--FOLD", type=int, default=1)
    parser.add_argument("--IS_BASE", type=str2bool, default=True)
    parser.add_argument("--IS_CAPTIONED", type=str2bool, default=True)
    parser.add_argument("--DATASET", type=str, default='CIFAR-10')
    parser.add_argument("--EPOCHS", type=int, default=10)
    parser.add_argument("--LR", type=float, default=2e-5)
    parser.add_argument("--BATCH_SIZE", type=int, default=64)
    
    ## Type
    args = parser.parse_args()

    SEED = args.SEED; set_SEED(SEED)
    FOLD = args.FOLD
    IS_BASE = args.IS_BASE
    IS_CAPTIONED = args.IS_CAPTIONED
    DATASET = args.DATASET
    
    EPOCHS = args.EPOCHS
    LR = args.LR
    BATCH_SIZE = args.BATCH_SIZE

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, model = get_FT_set(DATASET, IS_BASE, IS_CAPTIONED, fold=FOLD, batch_size=BATCH_SIZE)

    # Optimizer and loss function
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.linspace(0, EPOCHS, (EPOCHS//5) + 1)[1:-1].astype(int).tolist(), gamma=0.99)
    criterion = nn.CrossEntropyLoss()

    # Training and validation
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train(model, DEVICE, train_loader, optimizer, criterion, accumulation_steps=8)
        total_loss, accuracy = validate(model, DEVICE, valid_loader, criterion)
        if MAX_ACC < accuracy:
            LOSS = total_loss
            MAX_ACC = accuracy
        scheduler.step()

        save_metric(DATASET=DATASET, IS_BASE=IS_BASE, IS_CAPTIONED=IS_CAPTIONED, SEED=FOLD, total_loss=LOSS, accuracy=MAX_ACC)

if __name__ == "__main__":
    main()
    """
    ### IS_BASE True | IS_CAPTIONED True
CUDA_VISIBLE_DEVICES=2 python FT_audio_task.py --SEED 17 --FOLD 1 --IS_BASE True --IS_CAPTIONED True --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 40
CUDA_VISIBLE_DEVICES=2 python FT_audio_task.py --SEED 17 --FOLD 2 --IS_BASE True --IS_CAPTIONED True --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 40
CUDA_VISIBLE_DEVICES=2 python FT_audio_task.py --SEED 17 --FOLD 3 --IS_BASE True --IS_CAPTIONED True --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 40
CUDA_VISIBLE_DEVICES=2 python FT_audio_task.py --SEED 17 --FOLD 4 --IS_BASE True --IS_CAPTIONED True --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 40
CUDA_VISIBLE_DEVICES=2 python FT_audio_task.py --SEED 17 --FOLD 5 --IS_BASE True --IS_CAPTIONED True --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 40

CUDA_VISIBLE_DEVICES=2 python FT_audio_task.py --SEED 77 --FOLD 1 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 10 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=2 python FT_audio_task.py --SEED 77 --FOLD 2 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 10 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=2 python FT_audio_task.py --SEED 77 --FOLD 3 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 10 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=2 python FT_audio_task.py --SEED 77 --FOLD 4 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 10 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=2 python FT_audio_task.py --SEED 77 --FOLD 5 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 10 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=2 python FT_audio_task.py --SEED 77 --FOLD 6 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 10 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=2 python FT_audio_task.py --SEED 77 --FOLD 7 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 10 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=2 python FT_audio_task.py --SEED 77 --FOLD 8 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 10 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=2 python FT_audio_task.py --SEED 77 --FOLD 9 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 10 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=2 python FT_audio_task.py --SEED 77 --FOLD 10 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 10 --LR 5e-6 --BATCH_SIZE 44

    
    
    ### IS_BASE False | IS_CAPTIONED True
CUDA_VISIBLE_DEVICES=3 python FT_audio_task.py --SEED 17 --FOLD 1 --IS_BASE False --IS_CAPTIONED True --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=3 python FT_audio_task.py --SEED 17 --FOLD 2 --IS_BASE False --IS_CAPTIONED True --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=3 python FT_audio_task.py --SEED 17 --FOLD 3 --IS_BASE False --IS_CAPTIONED True --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=3 python FT_audio_task.py --SEED 17 --FOLD 4 --IS_BASE False --IS_CAPTIONED True --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=3 python FT_audio_task.py --SEED 17 --FOLD 5 --IS_BASE False --IS_CAPTIONED True --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 26

CUDA_VISIBLE_DEVICES=3 python FT_audio_task.py --SEED 77 --FOLD 1 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=3 python FT_audio_task.py --SEED 77 --FOLD 2 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=3 python FT_audio_task.py --SEED 77 --FOLD 3 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=3 python FT_audio_task.py --SEED 77 --FOLD 4 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 5 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 6 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 7 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 8 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 9 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 10 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
### IS_BASE True | IS_CAPTIONED False
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 17 --FOLD 1 --IS_BASE True --IS_CAPTIONED False --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 17 --FOLD 2 --IS_BASE True --IS_CAPTIONED False --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 17 --FOLD 3 --IS_BASE True --IS_CAPTIONED False --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 17 --FOLD 4 --IS_BASE True --IS_CAPTIONED False --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 17 --FOLD 5 --IS_BASE True --IS_CAPTIONED False --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 1 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 2 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 3 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 4 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 5 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 6 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 7 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 8 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 9 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 5e-6 --BATCH_SIZE 44
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 10 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 5e-6 --BATCH_SIZE 44
### IS_BASE False | IS_CAPTIONED False
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 17 --FOLD 1 --IS_BASE False --IS_CAPTIONED False --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 17 --FOLD 2 --IS_BASE False --IS_CAPTIONED False --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 17 --FOLD 3 --IS_BASE False --IS_CAPTIONED False --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 17 --FOLD 4 --IS_BASE False --IS_CAPTIONED False --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 17 --FOLD 5 --IS_BASE False --IS_CAPTIONED False --DATASET esc50 --EPOCHS 12 --LR 2e-5 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 1 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 2 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 3 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 4 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 5 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 6 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 7 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 8 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 9 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
CUDA_VISIBLE_DEVICES=0 python FT_audio_task.py --SEED 77 --FOLD 10 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k --EPOCHS 8 --LR 3e-6 --BATCH_SIZE 26
    """