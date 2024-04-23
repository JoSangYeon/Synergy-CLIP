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
from transformers import AutoProcessor, ASTModel, AutoTokenizer

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

  def __len__(self):
    return len(self.files)
  
  def __getitem__(self, idx):
    ## audio ##
    wav, sr = librosa.load(self.files[idx])
    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
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

  def __len__(self):
    return len(self.files)
  
  def __getitem__(self, idx):
    ## audio ##
    wav, sr = librosa.load(self.files[idx])
    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
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

def get_ZS_set(DATASET, IS_BASE, IS_CAPTIONED, fold=1):
    clip_config = CLIPConfig_BASE() if IS_BASE else CLIPConfig_LARGE()
    audio_model_path  = clip_config.audio_config.model_link

    if DATASET == 'esc50':
        # CIFAR-10 Data loading
        num_classes = 50

        df = pd.read_csv(os.path.join('data','ESC-50','esc50.csv'))
        base_path = os.path.join('.','data', 'ESC-50', 'audio')
        aud_processor = AutoProcessor.from_pretrained(audio_model_path)

        valid_dataset = ESC50Data(df, base_path, aud_processor, fold=fold, mode='test')
    elif DATASET == 'urbansound8k':
        # CIFAR-100 Data loading
        num_classes = 10

        base_path = os.path.join('data','UrbanSound8K')
        df = pd.read_csv(os.path.join(base_path, 'UrbanSound8K.csv'))
        aud_processor = AutoProcessor.from_pretrained(audio_model_path)

        valid_dataset = URBANSOUND8KData(df, base_path, aud_processor, fold=fold, mode='test')
    else:
        return None, None, None

    # Load the pre-trained Tri_CLIP model
    model_sz = 'base' if IS_BASE else 'large'
    text_des = 'caption' if IS_CAPTIONED else 'prompt'
    
    clip_config = CLIPConfig_BASE() if IS_BASE else CLIPConfig_LARGE()
    vision_model_path = clip_config.vision_config.model_link
    text_model_path   = clip_config.text_config.model_link
    audio_model_path  = clip_config.audio_config.model_link

    save_path = os.path.join(f'CLIP_model_{model_sz}_{text_des}.tar')

    model = Tri_CLIP(clip_config,
                     vision_model_path=vision_model_path,
                     text_model_path=text_model_path,
                     audio_model_path=audio_model_path)
    checkpoint = torch.load(save_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    txt_processor = AutoTokenizer.from_pretrained(text_model_path)

    return valid_dataset, txt_processor, model

def ZS_validate(model, valid_dataset, txt_processor, try_num=128):
    model.eval()
    prompt = 'this is a sound of {}'
    text_list = [prompt.format(tag) for label, tag in valid_dataset.i2c.items()]
    
    top1_correct = 0
    top5_correct = 0
    with torch.no_grad():
        for _ in tqdm(range(try_num)):
            random_idx = random.randint(0, len(valid_dataset)-1)
            data, target = valid_dataset.__getitem__(random_idx)
            text_inputs = txt_processor(text_list, padding=True, return_tensors='pt')
            data = data.unsqueeze(0)
            input_ids, att_mask = text_inputs.input_ids, text_inputs.attention_mask

            logits = model.get_aud_txt_sim_score(input_ids=input_ids, att_mask=att_mask, input_values=data) # N x 1
            logits = logits.T
            probs = logits.softmax(dim=1)

            top1_pred_idx  = probs.argmax().item()
            top5_pred_idxs = probs.topk(5).indices[0].tolist()

            top1_correct += int(top1_pred_idx == target)
            top5_correct += int(target in top5_pred_idxs)
    top1_acc = top1_correct / try_num 
    top5_acc = top5_correct / try_num 
    return {'top_1':top1_acc, 'top_5':top5_acc}

def save_metric(DATASET, IS_BASE, IS_CAPTIONED, SEED, result):
    with open("ZS_AUD.json", "r") as f:
	    data = json.load(f)
    data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"]["BASE" if IS_BASE else "LARGE"][f'FOLD_{SEED}'] = result

    temp = data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"]["BASE" if IS_BASE else "LARGE"]

    top1_lst = []
    top5_lst = []
    for k, v in temp.items():
        top1_lst.append(temp[k]['top_1'])
        top5_lst.append(temp[k]['top_5'])

    data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"][f'{"BASE" if IS_BASE else "LARGE"}_top1_acc_mean'] = np.mean(top1_lst)
    data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"][f'{"BASE" if IS_BASE else "LARGE"}_top1_acc_std'] = np.std(top1_lst)
    data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"][f'{"BASE" if IS_BASE else "LARGE"}_top5_acc_mean'] = np.mean(top5_lst)
    data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"][f'{"BASE" if IS_BASE else "LARGE"}_top5_acc_std'] = np.std(top5_lst)

    with open("ZS_AUD.json", "w") as f:
        json.dump(data, f, indent='\t')

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--SEED", type=int, default=17)
    parser.add_argument("--FOLD", type=int, default=1)
    parser.add_argument("--IS_BASE", type=str2bool, default=True)
    parser.add_argument("--IS_CAPTIONED", type=str2bool, default=True)
    parser.add_argument("--DATASET", type=str, default='CIFAR-10')
    
    ## Type
    args = parser.parse_args()

    SEED = args.SEED; set_SEED(SEED)
    FOLD = args.FOLD
    IS_BASE = args.IS_BASE
    IS_CAPTIONED = args.IS_CAPTIONED
    DATASET = args.DATASET

    valid_dataset, txt_processor, model = get_ZS_set(DATASET, IS_BASE, IS_CAPTIONED, fold=FOLD)

    # Zero shot validation
    result = ZS_validate(model, valid_dataset, txt_processor, try_num=128)
    save_metric(DATASET=DATASET, IS_BASE=IS_BASE, IS_CAPTIONED=IS_CAPTIONED, SEED=FOLD, result=result)

if __name__ == "__main__":
    main()
    """
    ### IS_BASE True | IS_CAPTIONED True
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 1 --IS_BASE True --IS_CAPTIONED True --DATASET esc50
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 2 --IS_BASE True --IS_CAPTIONED True --DATASET esc50
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 3 --IS_BASE True --IS_CAPTIONED True --DATASET esc50
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 4 --IS_BASE True --IS_CAPTIONED True --DATASET esc50
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 5 --IS_BASE True --IS_CAPTIONED True --DATASET esc50

    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 1 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 2 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 3 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 4 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 5 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 6 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 7 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 8 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 9 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 10 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k  

    
    
    ### IS_BASE False | IS_CAPTIONED True
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 1 --IS_BASE False --IS_CAPTIONED True --DATASET esc50  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 2 --IS_BASE False --IS_CAPTIONED True --DATASET esc50  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 3 --IS_BASE False --IS_CAPTIONED True --DATASET esc50  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 4 --IS_BASE False --IS_CAPTIONED True --DATASET esc50  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 5 --IS_BASE False --IS_CAPTIONED True --DATASET esc50  

    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 1 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 2 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 3 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 4 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 5 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 6 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 7 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 8 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 9 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 10 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k  



    ### IS_BASE True | IS_CAPTIONED False
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 1 --IS_BASE True --IS_CAPTIONED False --DATASET esc50  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 2 --IS_BASE True --IS_CAPTIONED False --DATASET esc50  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 3 --IS_BASE True --IS_CAPTIONED False --DATASET esc50  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 4 --IS_BASE True --IS_CAPTIONED False --DATASET esc50  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 5 --IS_BASE True --IS_CAPTIONED False --DATASET esc50  

    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 1 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 2 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 3 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 4 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 5 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 6 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 7 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 8 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 9 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 10 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k  



    ### IS_BASE False | IS_CAPTIONED False
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 1 --IS_BASE False --IS_CAPTIONED False --DATASET esc50  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 2 --IS_BASE False --IS_CAPTIONED False --DATASET esc50  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 3 --IS_BASE False --IS_CAPTIONED False --DATASET esc50  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 4 --IS_BASE False --IS_CAPTIONED False --DATASET esc50  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 17 --FOLD 5 --IS_BASE False --IS_CAPTIONED False --DATASET esc50  

    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 1 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 2 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 3 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 4 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 5 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 6 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 7 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 8 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 9 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k  
    CUDA_VISIBLE_DEVICES=0 python ZS_audio_task.py --SEED 77 --FOLD 10 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k  
    """