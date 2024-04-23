import os
import sys
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils import *

TEMPLATES = [
    # https://github.com/openai/CLIP/blob/main/data/prompts.md
    'a video of {}.',
    'a video about {}.',
    'a video of using {}.',
    'a video of doing {}.',
    'a video of during {}.',
    'a video of performing {}.',
    
    'a example of {}.',
    'a example about {}.',
    'a example of using {}.',
    'a example of doing {}.',
    'a example of during {}.',
    'a example of performing {}.',
    
    'a demonstration of {}.',
    'a demonstration about {}.',
    'a demonstration of using {}.',
    'a demonstration of doing {}.',
    'a demonstration of during {}.',
    'a demonstration of performing {}.',
    
    'a photo and sound of {}.',
    'a photo and sound about {}.',
    'a photo and sound of using {}.',
    'a photo and sound of doing {}.',
    'a photo and sound of during {}.',
    'a photo and sound of performing {}.',
    
    'a photo and audio of {}.',
    'a photo and audio about {}.',
    'a photo and audio of using {}.',
    'a photo and audio of doing {}.',
    'a photo and audio of during {}.',
    'a photo and audio of performing {}.',
    
    'a image and sound of {}.',
    'a image and sound about {}.',
    'a image and sound of using {}.',
    'a image and sound of doing {}.',
    'a image and sound of during {}.',
    'a image and sound of performing {}.',
    
    'a image and audio of {}.',
    'a image and audio about {}.',
    'a image and audio of using {}.',
    'a image and audio of doing {}.',
    'a image and audio of during {}.',
    'a image and audio of performing {}.',
    
    'this is a video of {}',
    'this is a demonstration of {}',
    'this is a photo and sound of {}',
    'this is a photo and audio of {}',
    'this is a image and sound of {}',
    'this is a image and audio of {}',
    
    'a bad video of {}.',
    'a bad example of {}.',
    'a bad demonstration of {}.',
    'a bad photo and sound of {}.',
    'a bad photo and audio of {}.',
    'a bad image and sound of {}.',
    'a bad image and audio of {}.',
    'a good video of {}.',
    'a good example of {}.',
    'a good demonstration of {}.',
    'a good photo and sound of {}.',
    'a good photo and audio of {}.',
    'a good image and sound of {}.',
    'a good image and audio of {}.',
    
    "video of {} I've taken and recorded.",
    "photo and sound of {} I've taken and recorded.",
    "photo and audio about {} I've taken and recorded.",
    "image and sound of {} I've taken and recorded.",
    "image and audio about {} I've taken and recorded.",
    
    "video of {} you've taken and recorded.",
    "photo and sound of {} you've taken and recorded.",
    "photo and audio about {} you've taken and recorded.",
    "image and sound of {} you've taken and recorded.",
    "image and audio about {} you've taken and recorded.",
]

class Dataset_Step1(Dataset):
    def __init__(self,
                 data_frame,
                 img_processor,
                 txt_tokenizer,
                 aud_processor,
                 seq_max_length=64,
                 sr=22500,
                 IS_CAPTIONED=False):
        super(Dataset_Step1, self).__init__()
        self.df = data_frame
        
        # img_path,wav_path,text     
        self.img_list = data_frame.img_path   
        self.label_list = data_frame.label
        self.caption_list = data_frame.caption
        self.audio_list = data_frame.wav_path
        
        self.max_length = seq_max_length # 77
        
        self.img_processor = img_processor
        self.txt_tokenizer = txt_tokenizer
        self.aud_processor = aud_processor
        
        self.IS_CAPTIONED = IS_CAPTIONED
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ## image ##
        img = Image.open(self.img_list[idx]).convert('RGB')
        # image = self.img_processor(images=img, return_tensors="pt")    # Size(3, img_sz, img_sz)
        # image = image.pixel_values[0]
        image = self.img_processor(img)
        
        ## audio ##
        wav, sr = librosa.load(self.audio_list[idx])
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        wav = audio_augment(wav, sr=16000, apply_rate=0.20)
        audio = self.aud_processor(wav, 
                                   sampling_rate=16000, 
                                   return_tensors="pt")                # Size(1024, 128)
        audio = audio.input_values[0]
        
        ## text ##
        if self.IS_CAPTIONED:
            text = self.caption_list[idx]
        else:
            labels = self.label_list[idx].split(',')
            word = random.choice(labels).strip()
            template = random.choice(TEMPLATES)
            text = template.format(word)
        # text = "Images and sounds about " + self.text_list[idx]
        tokenizer_output = self.txt_tokenizer(text, max_length=self.max_length, 
                                              padding='max_length', return_tensors='pt', 
                                              truncation=True, return_attention_mask=True)

        input_ids = tokenizer_output['input_ids'][0]
        att_mask = tokenizer_output['attention_mask'][0]
        # type_ids = tokenizer_output['token_type_ids'][0]
        
        return image, audio, (input_ids, att_mask)
    
class Dataset_Step23(Dataset):
    def __init__(self,
                 data_frame,
                 img_processor,
                 txt_tokenizer,
                 aud_processor,
                 seq_max_length=64,
                 sr=22500,
                 IS_CAPTIONED=False,
                 category_idx=[39,74,84]):
        super(Dataset_Step23, self).__init__()
        self.data = data_frame

        with open(os.path.join('vgg_category.json'), "r") as json_file:
            self.cate_tag = list(json.load(json_file).items())
        
        df = pd.DataFrame()
        for idx in tqdm(category_idx):
            category = self.cate_tag[idx][0]
            df = pd.concat([df, data_frame[data_frame.label==category]], axis=0)
        self.df = df.reset_index(drop=True).sample(frac=1.0)
        
        # img_path,wav_path,text     
        self.img_list = self.df.img_path   
        self.label_list =  self.df.label
        self.caption_list =  self.df.caption
        self.audio_list =  self.df.wav_path
        
        self.max_length = seq_max_length
        
        self.img_processor = img_processor
        self.txt_tokenizer = txt_tokenizer
        self.aud_processor = aud_processor
        
        self.IS_CAPTIONED = IS_CAPTIONED

        global TEMPLATES
        random.shuffle(TEMPLATES)
        self.template = TEMPLATES[:16]
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ## image ##
        img = Image.open(self.img_list[idx]).convert('RGB')
        image = self.img_processor(images=img, return_tensors="pt")    # Size(3, img_sz, img_sz)
        image = image.pixel_values[0]
        
        ## audio ##
        wav, sr = librosa.load(self.audio_list[idx])
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        # wav = audio_augment(wav, sr=16000)
        audio = self.aud_processor(wav, 
                                   sampling_rate=16000, 
                                   return_tensors="pt")                # Size(1024, 128)
        audio = audio.input_values[0]
        
        ## text ##
        if self.IS_CAPTIONED:
            text = self.caption_list[idx]
        else:
            labels = self.label_list[idx].split(',')
            template = random.choice(self.template)
            text = template.format(labels)

        # text = "Images and sounds about " + self.text_list[idx]
        tokenizer_output = self.txt_tokenizer(text, max_length=self.max_length, 
                                              padding='max_length', return_tensors='pt', 
                                              truncation=True, return_attention_mask=True)

        input_ids = tokenizer_output['input_ids'][0]
        att_mask = tokenizer_output['attention_mask'][0]
        # type_ids = tokenizer_output['token_type_ids'][0]
        
        return image, audio, (input_ids, att_mask)