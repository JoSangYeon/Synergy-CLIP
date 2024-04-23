import os
import sys
import json
import random
import argparse
import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets
from datasets import load_dataset
from transformers import AutoProcessor, CLIPVisionModel
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from model import Tri_CLIP
from config import *
from utils import *

import warnings
warnings.filterwarnings(action='ignore')

INIT_JSON = {
	"ImageNet": {
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
	"Pets": {
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
	"Flowers-102": {
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
	"CIFAR-10": {
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
	"CIFAR-100": {
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

MAX_ACC = -1
LOSS = 0

class ImageNet_Dataset(Dataset):
  def __init__(self, data, img_processor):
    self.data = data
    self.img_processor = img_processor

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    ## audio ##
    img = self.data['image'][idx]
    img = self.img_processor(img)

    ## label ##
    label = torch.tensor(self.data['label'][idx]).long()

    return img, label

class DownstreamTaskModel(nn.Module):
    def __init__(self, base_path, hidden_dim=768,
                 projection_dim=768, num_classes=10):
        super(DownstreamTaskModel, self).__init__()
        self.base_model = CLIPVisionModel.from_pretrained(base_path)
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

def get_FT_set(DATASET, IS_BASE, IS_CAPTIONED, batch_size = 64):
    if DATASET == 'CIFAR-10':
        # CIFAR-10 Data loading
        num_classes = 10

        train_trainsforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(224, 224)),
                torchvision.transforms.RandomChoice([
                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                    torchvision.transforms.RandomVerticalFlip(p=0.15)
                ]),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(brightness=0.075, contrast=0.075, saturation=0.075, hue=0.075),
                ], p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

        valid_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),  # Standardize image size
            torchvision.transforms.ToTensor(),  # Convert image to tensor
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize with ImageNet stats
        ])

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_trainsforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        valid_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=valid_transforms)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    elif DATASET == 'CIFAR-100':
        # CIFAR-100 Data loading
        num_classes = 100

        train_trainsforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(224, 224)),
                torchvision.transforms.RandomChoice([
                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                    torchvision.transforms.RandomVerticalFlip(p=0.15)
                ]),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(brightness=0.075, contrast=0.075, saturation=0.075, hue=0.075),
                ], p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

        valid_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),  # Standardize image size
            torchvision.transforms.ToTensor(),  # Convert image to tensor
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize with ImageNet stats
        ])

        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_trainsforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        valid_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=valid_transforms)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    elif DATASET == 'Flowers-102':
        # Flowers102 Data loading
        num_classes = 102

        train_trainsforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(224, 224)),
                torchvision.transforms.RandomChoice([
                    torchvision.transforms.RandomHorizontalFlip(p=0.25),
                    torchvision.transforms.RandomVerticalFlip(p=0.05)
                ]),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                ], p=0.25),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                ])

        valid_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),  # Standardize image size
            torchvision.transforms.ToTensor(),  # Convert image to tensor
            torchvision.transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),  # Normalize with ImageNet stats
        ])

        train_dataset_1 = datasets.Flowers102(root='./data', split='test', download=True, transform=train_trainsforms)
        train_dataset_2 = datasets.Flowers102(root='./data', split='train', download=True, transform=train_trainsforms)
        train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
        train_loader = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True, num_workers=4)

        valid_dataset = datasets.Flowers102(root='./data', split='val', download=True, transform=valid_transforms)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    elif DATASET == 'Pets':
        # OxfordIIITPet Data loading
        num_classes = 37

        train_trainsforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(224, 224)),
                torchvision.transforms.RandomChoice([
                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                    torchvision.transforms.RandomVerticalFlip(p=0.15)
                ]),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(brightness=0.075, contrast=0.075, saturation=0.075, hue=0.075),
                ], p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                                 (0.26862954, 0.26130258, 0.27577711))
                ])

        valid_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),  # Standardize image size
            torchvision.transforms.ToTensor(),  # Convert image to tensor
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
        ])

        train_dataset = datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=train_trainsforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        valid_dataset = datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=valid_transforms)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    elif DATASET == 'ImageNet':
        # ImageNet Data loading
        # https://huggingface.co/datasets/imagenet-1k
        num_classes = 1000

        train_trainsforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(224, 224)),
                torchvision.transforms.RandomChoice([
                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                    torchvision.transforms.RandomVerticalFlip(p=0.15)
                ]),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                ], p=0.33),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        valid_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),  # Standardize image size
            torchvision.transforms.ToTensor(),  # Convert image to tensor
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
        ])

        imagenet_train_data = load_dataset('imagenet-1k', split='train')
        train_dataset = ImageNet_Dataset(data=imagenet_train_data, img_processor=train_trainsforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        imagenet_valid_data = load_dataset('imagenet-1k', split='validation')
        valid_dataset = ImageNet_Dataset(data=imagenet_valid_data, img_processor=valid_transforms)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    else:
        return None, None, None

    # Load the pre-trained Tri_CLIP model
    clip_config = CLIPConfig_BASE() if IS_BASE else CLIPConfig_LARGE()
    hidden_size = clip_config.vision_config.hidden_size
    projection_dim = clip_config.projection_dim
    save_path = os.path.join(f"CLIP_image_model_{'base' if IS_BASE else 'large'}",
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
    with open("METRIC_IMG.json", "r") as f:
	    data = json.load(f)
    metric = {'loss':total_loss, 'accuracy':accuracy}
    data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"]["BASE" if IS_BASE else "LARGE"][f'SEED_{SEED}'] = metric

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

    with open("METRIC_IMG.json", "w") as f:
        json.dump(data, f, indent='\t')

def main():
    global MAX_ACC, LOSS

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--SEED", type=int, default=17)
    parser.add_argument("--IS_BASE", type=str2bool, default=True)
    parser.add_argument("--IS_CAPTIONED", type=str2bool, default=True)
    parser.add_argument("--DATASET", type=str, default='CIFAR-10')
    parser.add_argument("--EPOCHS", type=int, default=10)
    parser.add_argument("--LR", type=float, default=2e-5)
    parser.add_argument("--BATCH_SIZE", type=int, default=64)
    
    ## Type
    args = parser.parse_args()

    SEED = args.SEED; set_SEED(SEED)
    IS_BASE = args.IS_BASE
    IS_CAPTIONED = args.IS_CAPTIONED
    DATASET = args.DATASET
    
    EPOCHS = args.EPOCHS
    LR = args.LR
    BATCH_SIZE = args.BATCH_SIZE

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, model = get_FT_set(DATASET, IS_BASE, IS_CAPTIONED, batch_size=BATCH_SIZE)

    # Optimizer and loss function
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    # Training and validation
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train(model, DEVICE, train_loader, optimizer, criterion, accumulation_steps=8)
        total_loss, accuracy = validate(model, DEVICE, valid_loader, criterion)
        if MAX_ACC < accuracy:
            LOSS = total_loss
            MAX_ACC = accuracy

        save_metric(DATASET=DATASET, IS_BASE=IS_BASE, IS_CAPTIONED=IS_CAPTIONED, SEED=SEED, total_loss=LOSS, accuracy=MAX_ACC)
    # save_metric(DATASET=DATASET, IS_BASE=IS_BASE, IS_CAPTIONED=IS_CAPTIONED, SEED=SEED, total_loss=total_loss, accuracy=accuracy)
    

if __name__ == "__main__":
    main()
    """
    ### IS_BASE True | IS_CAPTIONED True
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 17 --IS_BASE True --IS_CAPTIONED True --DATASET CIFAR-10 --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 42 --IS_BASE True --IS_CAPTIONED True --DATASET CIFAR-10 --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 77 --IS_BASE True --IS_CAPTIONED True --DATASET CIFAR-10 --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 256

    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 18 --IS_BASE True --IS_CAPTIONED True --DATASET CIFAR-100 --EPOCHS 7 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 43 --IS_BASE True --IS_CAPTIONED True --DATASET CIFAR-100 --EPOCHS 7 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 78 --IS_BASE True --IS_CAPTIONED True --DATASET CIFAR-100 --EPOCHS 7 --LR 2e-5 --BATCH_SIZE 256

    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 19 --IS_BASE True --IS_CAPTIONED True --DATASET Flowers-102 --EPOCHS 25 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 44 --IS_BASE True --IS_CAPTIONED True --DATASET Flowers-102 --EPOCHS 25 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 79 --IS_BASE True --IS_CAPTIONED True --DATASET Flowers-102 --EPOCHS 25 --LR 2e-5 --BATCH_SIZE 256

    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 28 --IS_BASE True --IS_CAPTIONED True --DATASET "Pets" --EPOCHS 20 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 53 --IS_BASE True --IS_CAPTIONED True --DATASET "Pets" --EPOCHS 20 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 88 --IS_BASE True --IS_CAPTIONED True --DATASET "Pets" --EPOCHS 20 --LR 2e-5 --BATCH_SIZE 256

    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 29 --IS_BASE True --IS_CAPTIONED True --DATASET ImageNet --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 54 --IS_BASE True --IS_CAPTIONED True --DATASET ImageNet --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 89 --IS_BASE True --IS_CAPTIONED True --DATASET ImageNet --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 256
    


    ### IS_BASE True | IS_CAPTIONED False
    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 17 --IS_BASE True --IS_CAPTIONED False --DATASET CIFAR-10 --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 42 --IS_BASE True --IS_CAPTIONED False --DATASET CIFAR-10 --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 77 --IS_BASE True --IS_CAPTIONED False --DATASET CIFAR-10 --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 256

    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 18 --IS_BASE True --IS_CAPTIONED False --DATASET CIFAR-100 --EPOCHS 7 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 43 --IS_BASE True --IS_CAPTIONED False --DATASET CIFAR-100 --EPOCHS 7 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 78 --IS_BASE True --IS_CAPTIONED False --DATASET CIFAR-100 --EPOCHS 7 --LR 2e-5 --BATCH_SIZE 256

    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 19 --IS_BASE True --IS_CAPTIONED False --DATASET Flowers-102 --EPOCHS 25 --LR 3e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 44 --IS_BASE True --IS_CAPTIONED False --DATASET Flowers-102 --EPOCHS 25 --LR 3e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 79 --IS_BASE True --IS_CAPTIONED False --DATASET Flowers-102 --EPOCHS 25 --LR 3e-5 --BATCH_SIZE 256

    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 28 --IS_BASE True --IS_CAPTIONED False --DATASET "Pets" --EPOCHS 20 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 53 --IS_BASE True --IS_CAPTIONED False --DATASET "Pets" --EPOCHS 20 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 88 --IS_BASE True --IS_CAPTIONED False --DATASET "Pets" --EPOCHS 20 --LR 2e-5 --BATCH_SIZE 256

    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 29 --IS_BASE True --IS_CAPTIONED False --DATASET ImageNet --EPOCHS 10 --LR 1e-4 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 54 --IS_BASE True --IS_CAPTIONED False --DATASET ImageNet --EPOCHS 10 --LR 1e-4 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 89 --IS_BASE True --IS_CAPTIONED False --DATASET ImageNet --EPOCHS 10 --LR 1e-4 --BATCH_SIZE 256
    

    ### IS_BASE False | IS_CAPTIONED True
    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 17 --IS_BASE False --IS_CAPTIONED True --DATASET CIFAR-10 --EPOCHS 1 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 42 --IS_BASE False --IS_CAPTIONED True --DATASET CIFAR-10 --EPOCHS 1 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 77 --IS_BASE False --IS_CAPTIONED True --DATASET CIFAR-10 --EPOCHS 1 --LR 2e-5 --BATCH_SIZE 64

    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 18 --IS_BASE False --IS_CAPTIONED True --DATASET CIFAR-100 --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 43 --IS_BASE False --IS_CAPTIONED True --DATASET CIFAR-100 --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 78 --IS_BASE False --IS_CAPTIONED True --DATASET CIFAR-100 --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 64

    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 19 --IS_BASE False --IS_CAPTIONED True --DATASET Flowers-102 --EPOCHS 7 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 44 --IS_BASE False --IS_CAPTIONED True --DATASET Flowers-102 --EPOCHS 7 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 79 --IS_BASE False --IS_CAPTIONED True --DATASET Flowers-102 --EPOCHS 7 --LR 2e-5 --BATCH_SIZE 64

    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 28 --IS_BASE False --IS_CAPTIONED True --DATASET "Pets" --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 53 --IS_BASE False --IS_CAPTIONED True --DATASET "Pets" --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=2 python FT_image_task.py --SEED 88 --IS_BASE False --IS_CAPTIONED True --DATASET "Pets" --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 64

    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 29 --IS_BASE False --IS_CAPTIONED True --DATASET ImageNet --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 54 --IS_BASE False --IS_CAPTIONED True --DATASET ImageNet --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 89 --IS_BASE False --IS_CAPTIONED True --DATASET ImageNet --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 64
    
    

    ### IS_BASE False | IS_CAPTIONED False
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 17 --IS_BASE False --IS_CAPTIONED False --DATASET CIFAR-10 --EPOCHS 1 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 42 --IS_BASE False --IS_CAPTIONED False --DATASET CIFAR-10 --EPOCHS 1 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 77 --IS_BASE False --IS_CAPTIONED False --DATASET CIFAR-10 --EPOCHS 1 --LR 2e-5 --BATCH_SIZE 64

    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 18 --IS_BASE False --IS_CAPTIONED False --DATASET CIFAR-100 --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 43 --IS_BASE False --IS_CAPTIONED False --DATASET CIFAR-100 --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 78 --IS_BASE False --IS_CAPTIONED False --DATASET CIFAR-100 --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 64

    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 19 --IS_BASE False --IS_CAPTIONED False --DATASET Flowers-102 --EPOCHS 7 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 44 --IS_BASE False --IS_CAPTIONED False --DATASET Flowers-102 --EPOCHS 7 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 79 --IS_BASE False --IS_CAPTIONED False --DATASET Flowers-102 --EPOCHS 7 --LR 2e-5 --BATCH_SIZE 64

    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 28 --IS_BASE False --IS_CAPTIONED False --DATASET "Pets" --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 53 --IS_BASE False --IS_CAPTIONED False --DATASET "Pets" --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=3 python FT_image_task.py --SEED 88 --IS_BASE False --IS_CAPTIONED False --DATASET "Pets" --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 64

    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 29 --IS_BASE False --IS_CAPTIONED False --DATASET ImageNet --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 54 --IS_BASE False --IS_CAPTIONED False --DATASET ImageNet --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 64
    CUDA_VISIBLE_DEVICES=1 python FT_image_task.py --SEED 89 --IS_BASE False --IS_CAPTIONED False --DATASET ImageNet --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 64
    """