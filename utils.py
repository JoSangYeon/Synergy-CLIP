import os
import random
import librosa
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler

from tqdm import tqdm

"""
https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
https://gist.github.com/spnova12/3c4388f66514e3f7506aa8629453b955
"""
class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 1.0]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def set_SEED(SEED):
    random.seed(SEED) #  Python의 random 라이브러리가 제공하는 랜덤 연산이 항상 동일한 결과를 출력하게끔
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# https://www.kaggle.com/code/CVxTz/audio-data-augmentation
def add_white_noise(data, sr=16000, rate=0.005):
    wn = np.random.randn(len(data))
    data_wn = data + rate * wn
    return data_wn
def shift_data(data, sr=16000, rate=0.1):
    data_roll = np.roll(data, int(len(data) * rate))
    return data_roll
def stretch_data(data, sr=16000, rate=0.75):
    stretch_data = librosa.effects.time_stretch(data, rate=rate)
    return stretch_data
def minus_sound(data, sr=16000, rate=0):
    minus_data = (-1) * data
    return minus_data
def audio_augment(data, sr=16000, noise_rate=0.005, 
                  shift_rate=0.1, stretch_rate=0.85,
                  is_shuffle=True, apply_rate=0.5):
    aug_method_list = [(add_white_noise, noise_rate), 
                       (shift_data, shift_rate),
                       (stretch_data, stretch_rate), 
                       (minus_sound, 0.0)]
    if is_shuffle: random.shuffle(aug_method_list)
    
    for method, rate in aug_method_list:
        is_apply = random.random()
        if is_apply < apply_rate:
            data = method(data, sr, rate)
    return data
