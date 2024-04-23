import os
import sys
import json
import random
import argparse
import numpy as np 
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets
from transformers import AutoProcessor, CLIPVisionModel, AutoTokenizer
from torch.utils.data import ConcatDataset, DataLoader

from model import Tri_CLIP
from config import *
from utils import *

cifar10_label_tag = {
    0 : 'airplane',
    1 : 'automobile',
    2 : 'bird',
    3 : 'cat',
    4 : 'deer',
    5 : 'dog',
    6 : 'frog',
    7 : 'horse',
    8 : 'ship',
    9 : 'truck',
}

cifar100_label_tag = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm"
]

INIT_JSON = {
	"cifar-10": {
		"result": {
			"caption": {
				"BASE": {},
				"LARGE": {}
			},
			"prompt": {
				"BASE": {},
				"LARGE": {}
			}
		}
	},
	"cifar-100": {
		"result": {
			"caption": {
				"BASE": {},
				"LARGE": {}
			},
			"prompt": {
				"BASE": {},
				"LARGE": {}
			}
		}
	},
	"esc50": {
		"result": {
			"caption": {
				"BASE": {},
				"LARGE": {}
			},
			"prompt": {
				"BASE": {},
				"LARGE": {}
			}
		}
	},
    "urbansound8k": {
		"result": {
			"caption": {
				"BASE": {},
				"LARGE": {}
			},
			"prompt": {
				"BASE": {},
				"LARGE": {}
			}
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

def get_ZS_set(DATASET, MM, IS_BASE, IS_CAPTIONED):
    # Load the pre-trained Tri_CLIP model
    model_sz = 'base' if IS_BASE else 'large'
    text_des = 'caption' if IS_CAPTIONED else 'prompt'
    
    clip_config = CLIPConfig_BASE() if IS_BASE else CLIPConfig_LARGE()
    vision_model_path = clip_config.vision_config.model_link
    text_model_path   = clip_config.text_config.model_link
    audio_model_path  = clip_config.audio_config.model_link
    
    save_path = os.path.join(f'ZS_CLIP_model_{MM}_{model_sz}_{text_des}.tar')

    model = Tri_CLIP(clip_config,
                     vision_model_path=vision_model_path,
                     text_model_path=text_model_path,
                     audio_model_path=audio_model_path)
    checkpoint = torch.load(save_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    txt_processor = AutoTokenizer.from_pretrained(text_model_path)

    if DATASET == 'cifar-10':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),  # Adjust according to your model's input size
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        valid_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif DATASET == 'cifar-100':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),  # Adjust according to your model's input size
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        valid_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    elif DATASET == 'esc50':
        fold = random.randint(1,5)
        df = pd.read_csv(os.path.join('data','ESC-50','esc50.csv'))
        base_path = os.path.join('.','data', 'ESC-50', 'audio')
        aud_processor = AutoProcessor.from_pretrained(audio_model_path)

        valid_dataset = ESC50Data(df, base_path, aud_processor, fold=fold, mode='test')
    elif DATASET == 'urbansound8k':
        fold = random.randint(1,10)
        base_path = os.path.join('data','UrbanSound8K')
        df = pd.read_csv(os.path.join(base_path, 'UrbanSound8K.csv'))
        aud_processor = AutoProcessor.from_pretrained(audio_model_path)

        valid_dataset = URBANSOUND8KData(df, base_path, aud_processor, fold=fold, mode='test')
    else:
        return None, None

    return valid_dataset, txt_processor, model

def ZS_validate(model, DATASET, valid_dataset, txt_processor, try_num=128):
    model.eval()

    if DATASET == 'cifar-10':
        prompt = 'this is a photo of {}'
        text_list = [prompt.format(tag) for label, tag in cifar10_label_tag.items()]
    elif DATASET == 'cifar-100':
        prompt = 'this is a photo of {}'
        text_list = [prompt.format(tag) for tag in cifar100_label_tag]
    else:
        prompt = 'this is a sound of {}'
        text_list = [prompt.format(tag) for label, tag in valid_dataset.i2c.items()]

    top1_correct = 0
    top5_correct = 0
    with torch.no_grad():
        for _ in tqdm(range(try_num)):
            random_idx = random.randint(0, len(valid_dataset)-1)

            if DATASET == 'cifar-10' or DATASET == 'cifar-100':
                data, target = valid_dataset.__getitem__(random_idx)
                text_inputs = txt_processor(text_list, padding=True, return_tensors='pt')
                data = data.unsqueeze(0)
                input_ids, att_mask = text_inputs.input_ids, text_inputs.attention_mask

                logits = model.get_img_txt_sim_score(pixel_values=data, input_ids=input_ids, att_mask=att_mask)
            else:
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
    with open("ZS_OUR.json", "r") as f:
	    data = json.load(f)
    data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"]["BASE" if IS_BASE else "LARGE"][f'SEED_{SEED}'] = result

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

    with open("ZS_OUR.json", "w") as f:
        json.dump(data, f, indent='\t')

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--SEED", type=int, default=17)
    parser.add_argument("--IS_BASE", type=str2bool, default=True)
    parser.add_argument("--IS_CAPTIONED", type=str2bool, default=True)
    parser.add_argument("--DATASET", type=str, default='Image')
    parser.add_argument("--MM", type=str, default='IT')
    
    ## Type
    args = parser.parse_args()

    SEED = args.SEED; set_SEED(SEED); 
    IS_BASE = args.IS_BASE
    IS_CAPTIONED = args.IS_CAPTIONED
    DATASET = args.DATASET
    MM = args.MM

    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_dataset, txt_processor, model = get_ZS_set(DATASET, MM, IS_BASE, IS_CAPTIONED)

    # Zero shot validation
    result = ZS_validate(model, DATASET, valid_dataset, txt_processor, try_num=128)

    save_metric(DATASET=DATASET, IS_BASE=IS_BASE, IS_CAPTIONED=IS_CAPTIONED, SEED=SEED, result=result)

if __name__ == "__main__":
    main()
    """
    # Cifar-10 #
    ### IS_BASE True | IS_CAPTIONED True
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 7 --IS_BASE True --IS_CAPTIONED True --DATASET cifar-10 --MM IT
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 8 --IS_BASE True --IS_CAPTIONED True --DATASET cifar-10 --MM IT
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 9 --IS_BASE True --IS_CAPTIONED True --DATASET cifar-10 --MM IT

    ### IS_BASE False | IS_CAPTIONED True
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 11 --IS_BASE False --IS_CAPTIONED True --DATASET cifar-10 --MM IT
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 12 --IS_BASE False --IS_CAPTIONED True --DATASET cifar-10 --MM IT
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 13 --IS_BASE False --IS_CAPTIONED True --DATASET cifar-10 --MM IT

    ### IS_BASE True | IS_CAPTIONED False
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 25 --IS_BASE True --IS_CAPTIONED False --DATASET cifar-10 --MM IT
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 26 --IS_BASE True --IS_CAPTIONED False --DATASET cifar-10 --MM IT
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 27 --IS_BASE True --IS_CAPTIONED False --DATASET cifar-10 --MM IT
    
    ### IS_BASE False | IS_CAPTIONED False
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 49 --IS_BASE False --IS_CAPTIONED False --DATASET cifar-10 --MM IT
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 50 --IS_BASE False --IS_CAPTIONED False --DATASET cifar-10 --MM IT
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 51 --IS_BASE False --IS_CAPTIONED False --DATASET cifar-10 --MM IT


    # Cifar-100 #
    ### IS_BASE True | IS_CAPTIONED True
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 7 --IS_BASE True --IS_CAPTIONED True --DATASET cifar-100 --MM IT
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 8 --IS_BASE True --IS_CAPTIONED True --DATASET cifar-100 --MM IT
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 9 --IS_BASE True --IS_CAPTIONED True --DATASET cifar-100 --MM IT   

    ### IS_BASE False | IS_CAPTIONED True
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 11 --IS_BASE False --IS_CAPTIONED True --DATASET cifar-100 --MM IT
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 12 --IS_BASE False --IS_CAPTIONED True --DATASET cifar-100 --MM IT
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 13 --IS_BASE False --IS_CAPTIONED True --DATASET cifar-100 --MM IT

    ### IS_BASE True | IS_CAPTIONED False
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 25 --IS_BASE True --IS_CAPTIONED False --DATASET cifar-100 --MM IT
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 26 --IS_BASE True --IS_CAPTIONED False --DATASET cifar-100 --MM IT
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 27 --IS_BASE True --IS_CAPTIONED False --DATASET cifar-100 --MM IT
    
    ### IS_BASE False | IS_CAPTIONED False
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 49 --IS_BASE False --IS_CAPTIONED False --DATASET cifar-100 --MM IT
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 50 --IS_BASE False --IS_CAPTIONED False --DATASET cifar-100 --MM IT
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 51 --IS_BASE False --IS_CAPTIONED False --DATASET cifar-100 --MM IT



    

    # esc50 #
    ### IS_BASE True | IS_CAPTIONED True
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 7 --IS_BASE True --IS_CAPTIONED True --DATASET esc50 --MM TA
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 8 --IS_BASE True --IS_CAPTIONED True --DATASET esc50 --MM TA
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 9 --IS_BASE True --IS_CAPTIONED True --DATASET esc50 --MM TA   

    ### IS_BASE False | IS_CAPTIONED True
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 11 --IS_BASE False --IS_CAPTIONED True --DATASET esc50 --MM TA
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 12 --IS_BASE False --IS_CAPTIONED True --DATASET esc50 --MM TA
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 13 --IS_BASE False --IS_CAPTIONED True --DATASET esc50 --MM TA

    ### IS_BASE True | IS_CAPTIONED False
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 25 --IS_BASE True --IS_CAPTIONED False --DATASET esc50 --MM TA
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 26 --IS_BASE True --IS_CAPTIONED False --DATASET esc50 --MM TA
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 27 --IS_BASE True --IS_CAPTIONED False --DATASET esc50 --MM TA
    
    ### IS_BASE False | IS_CAPTIONED False
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 39 --IS_BASE False --IS_CAPTIONED False --DATASET esc50 --MM TA
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 40 --IS_BASE False --IS_CAPTIONED False --DATASET esc50 --MM TA
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 41 --IS_BASE False --IS_CAPTIONED False --DATASET esc50 --MM TA

    
    # urbansound8k #
    ### IS_BASE True | IS_CAPTIONED True
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 17 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k --MM TA
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 27 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k --MM TA
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 37 --IS_BASE True --IS_CAPTIONED True --DATASET urbansound8k --MM TA   

    ### IS_BASE False | IS_CAPTIONED True
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 17 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k --MM TA
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 27 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k --MM TA
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 37 --IS_BASE False --IS_CAPTIONED True --DATASET urbansound8k --MM TA

    ### IS_BASE True | IS_CAPTIONED False
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 17 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k --MM TA
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 27 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k --MM TA
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 37 --IS_BASE True --IS_CAPTIONED False --DATASET urbansound8k --MM TA
    
    ### IS_BASE False | IS_CAPTIONED False
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 17 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k --MM TA
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 27 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k --MM TA
    CUDA_VISIBLE_DEVICES=0 python ZS_task.py --SEED 37 --IS_BASE False --IS_CAPTIONED False --DATASET urbansound8k --MM TA
    """