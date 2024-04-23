import os
import sys
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from model import *
from dataset import *
from config import *
from utils import *
from inference_MMR import deploy

import warnings
warnings.filterwarnings(action='ignore')

MIN_LOSS = 999999

def save_model(model, IS_BASE, IS_CAPTIONED, MM):
    model_sz = "base" if IS_BASE else "large"
    text_des = "caption" if IS_CAPTIONED else "prompt"
    ckpt_savepath = os.path.join('inference', f'{model_sz}_{text_des}', 
                                 MM, f'MRL_model_{model_sz}_{text_des}.tar')
    torch.save({
        'model_state_dict': model.state_dict(),
    }, ckpt_savepath)

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    global MIN_LOSS
    SEED = args.SEED; set_SEED(SEED)
    IS_BASE = args.IS_BASE; model_sz = "base" if IS_BASE else "large"
    IS_CAPTIONED = args.IS_CAPTIONED; text_des = "caption" if IS_CAPTIONED else "prompt"
    MM = args.MM
    IDX = args.IDX
    CATE_IDX = [[39,74], [75, 83, 99], [42, 62, 225], [21,24,20]][IDX]

    epochs = args.epochs
    batch_size = 32
    
    seq_max_length = args.seq_max_length
    lr = args.learning_rate
    alpha = args.alpha
    beta  = args.beta
    gamma = args.gamma
    hyper_param = {'img':alpha, 'txt':beta, 'aud':gamma}
    
    clip_config = CLIPConfig_BASE() if IS_BASE else CLIPConfig_LARGE()
    recon_config = ReconstructionConfig_BASE() if IS_BASE else ReconstructionConfig_LARGE()
    vision_model_path = recon_config.vision_config.model_link
    text_model_path   = recon_config.text_config.model_link
    audio_model_path  = recon_config.audio_config.model_link

    train_path = 'vgg_sound_train_captioned.csv'
    test_path = 'vgg_sound_test_captioned.csv'
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_data = train_data.sample(frac=0.10, random_state=SEED+7);   train_data = train_data.reset_index(drop=True)
    valid_data = train_data.sample(frac=0.10, random_state=SEED+77);  valid_data = valid_data.reset_index(drop=True)
    test_data  = train_data.sample(frac=0.10 random_state=SEED+777); test_data  = test_data.reset_index(drop=True)

    img_processor = AutoProcessor.from_pretrained(vision_model_path, do_normalize=True, image_mean= [0., 0., 0.], image_std = [1., 1., 1.])
    txt_processor = AutoTokenizer.from_pretrained(text_model_path)
    aud_processor = AutoProcessor.from_pretrained(audio_model_path, do_normalize=True, image_mean= [0., 0., 0.], image_std = [1., 1., 1.])

    train_dataset = Dataset_Step23(train_data, img_processor, txt_processor, aud_processor, seq_max_length=seq_max_length, IS_CAPTIONED=IS_CAPTIONED, category_idx=CATE_IDX)
    valid_dataset = Dataset_Step23(valid_data, img_processor, txt_processor, aud_processor, seq_max_length=seq_max_length, IS_CAPTIONED=IS_CAPTIONED, category_idx=CATE_IDX)
    test_dataset  = Dataset_Step23(test_data,  img_processor, txt_processor, aud_processor, seq_max_length=seq_max_length, IS_CAPTIONED=IS_CAPTIONED, category_idx=CATE_IDX)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    clip_model = Tri_CLIP(clip_config,
                          vision_model_path=vision_model_path,
                          text_model_path=text_model_path,
                          audio_model_path=audio_model_path)
    checkpoint = torch.load(os.path.join(f'CLIP_model_{model_sz}_{text_des}.tar'), map_location=device)
    clip_model.load_state_dict(checkpoint['model_state_dict'])

    vision_model = clip_model.vision_model
    text_model   = clip_model.text_model
    audio_model  = clip_model.audio_model

    if MM == 'img':
        text_model   = clip_model.text_model
        audio_model  = clip_model.audio_model
        model = TXT_AUD_2_IMG(recon_config, text_model, audio_model).to(device)
    elif MM == 'txt':
        vision_model = clip_model.vision_model
        audio_model  = clip_model.audio_model
        model = IMG_AUD_2_TXT(recon_config, vision_model, audio_model).to(device)
    else:
        vision_model = clip_model.vision_model
        text_model   = clip_model.text_model
        model = IMG_TXT_2_AUD(recon_config, vision_model, text_model).to(device)


    opt = optim.AdamW(model.parameters(), lr=lr)
    # opt = optim.AdamW(ddp_model.parameters(), lr=lr, betas=[0.9, 0.999], eps=1e-08, weight_decay=0.05)
    # opt = optim.SGD(ddp_model.parameters(), lr=lr)
    # opt = optim.AdamW([{'params': ddp_model.module.MM_encoder.parameters(),'lr': 1e-6},
    #                          {'params': ddp_model.module.img_decoder.parameters(),'lr': 1e-6},
    #                          {'params': ddp_model.module.txt_decoder.parameters(),'lr': 1e-6},
    #                          {'params': ddp_model.module.aud_decoder.parameters(),'lr': 1e-6}], lr=1e-6)
    # milestones = np.linspace(0, epochs, (epochs//25)+1)[1:-1].astype(int).tolist()
    # scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.9)
    
    images, audios, (input_ids, att_mask) = next(iter(train_loader))
    images, audios = images.to(device), audios.to(device)
    input_ids, att_mask = input_ids.to(device), att_mask.to(device)

    pbar = tqdm(range(epochs), file=sys.stdout)
    for i, e in enumerate(pbar):
        model.train()
        
        loss_lst = []
        metric_1_lst = []
        metric_2_lst = []

        opt.zero_grad()
        if MM == 'img':
            output = model(input_ids, att_mask, audios, label=images, alpha=0.75)
        elif MM == 'txt':
            output = model(images, audios, label=input_ids)
        else:
            output = model(images, input_ids, att_mask, label=audios, alpha=0.75)

        loss, metric_1, metric_2 = output
        loss *= hyper_param[MM]
        loss.backward()
        opt.step()
        
        loss_lst.append(loss.item())
        metric_1_lst.append(metric_1.item()); metric_2_lst.append(metric_2.item())
    
        if MM == 'img' or MM == 'aud':
            post_fix = f'epoch={e+1}/{epochs}, loss={np.mean(loss_lst):.4f}, MSE={np.mean(metric_1_lst):.4f}, SSIM={np.mean(metric_2_lst):.4f}'
        else:
            post_fix = f'epoch={e+1}/{epochs}, loss={np.mean(loss_lst):.4f}, Acc={np.mean(metric_1_lst):.4f}'
        pbar.set_postfix_str(post_fix)
    pbar.close()
    
    if MIN_LOSS > np.mean(loss_lst):
        MIN_LOSS = np.mean(loss_lst)
        save_model(model, IS_BASE, IS_CAPTIONED, MM)

    image_size = recon_config.img_size # 128x128
    img_channels = recon_config.img_channels
    label_images = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(image_size, image_size), 
                                        interpolation=Image.BICUBIC),
    ])
        
    audio_height, audio_width = recon_config.aud_size # 256x64
    aud_channels = recon_config.aud_channels
    label_audios = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(audio_height, audio_width),
                                        interpolation=Image.BICUBIC),
    ])


    if MM == 'img': 
        label = label_images(images)
        mm_recon = model(input_ids, att_mask, audios, label=None)
        for _ in tqdm(range(16)):
            deploy(MM, model_sz, text_des, 
                   mm_recon, label, txt_processor, 
                   idx=random.randint(0, batch_size-1),
                   cate=IDX)  
        psnr = PSNR()
        psnr_score = psnr(mm_recon, label)
        ssim_score = ssim(mm_recon, label, data_range=1.0, size_average=True)    
        print(f"psnr_score : {psnr_score:.3f} | ssim_score : {ssim_score:.4f}")
    elif MM == 'txt': 
        label = input_ids
        mm_recon = model(images, audios, label=None)
        for _ in tqdm(range(16)):
            deploy(MM, model_sz, text_des, 
                   mm_recon, label, txt_processor, 
                   idx=random.randint(0, batch_size-1),
                   cate=IDX)  
        pred_txt = mm_recon.argmax(dim=-1)
        acc = (pred_txt.view(-1) == label.view(-1)).sum() / len(pred_txt.view(-1))
        print(f"accuracy : {acc:.4f} | bleu : {0:.4f}")
    else: 
        label = label_audios(audios.unsqueeze(1)).squeeze(1)
        mm_recon = model(images, input_ids, att_mask, label=None)
        for _ in tqdm(range(16)):
            deploy(MM, model_sz, text_des,  
                   mm_recon, label, txt_processor, 
                   idx=random.randint(0, batch_size-1), 
                   cate=IDX) 
        psnr = PSNR()
        psnr_score = psnr(mm_recon.unsqueeze(1), label.unsqueeze(1))
        ssim_score = ssim(mm_recon.unsqueeze(1), label.unsqueeze(1), data_range=1.0, size_average=True)
        print(f"psnr_score : {psnr_score:.3f} | ssim_score : {ssim_score:.4f}")

 


def parse_args():
    parser = argparse.ArgumentParser(description='STEP 2 : Modality Representation Learning')
    parser.add_argument('--SEED', type=int, default=17, help='Random Seed')
    parser.add_argument('--IS_BASE', type=str2bool, default=True, help='Model Size : True is "BASE" | False is "Large"')
    parser.add_argument('--IS_CAPTIONED', type=str2bool, default=True, help='Text Description : True is "Caption" | False is "Prompt"')
    parser.add_argument('--MM', type=str, default='img', help='Which modalities will be missing? [ img : 0 | txt : 1 | aud : 2 ]')
    parser.add_argument('--IDX', type=int, default=0, help='Cate Tag')

    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--seq_max_length', type=int, default=32, help='Max Sequence Length for TextModel')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--alpha', type=float, default=1.0, help='Hyperparameter for Vision Loss')
    parser.add_argument('--beta', type=float, default=1.0, help='Hyperparameter for Text Loss')
    parser.add_argument('--gamma', type=float, default=1.0, help='Hyperparameter for Audio Loss')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
    """
CUDA_VISIBLE_DEVICES=3 python main_MMR.py --IS_BASE False --IS_CAPTIONED True --MM img --IDX 3 --epochs 768 --learning_rate 8e-4
CUDA_VISIBLE_DEVICES=2 python main_MMR.py --IS_BASE False --IS_CAPTIONED True --MM txt --IDX 3  --epochs 48 --learning_rate 2e-4
CUDA_VISIBLE_DEVICES=1 python main_MMR.py --IS_BASE False --IS_CAPTIONED True --MM aud --IDX 3 --epochs 384 --learning_rate 8e-4
    """
    """
CUDA_VISIBLE_DEVICES=0 python main_MMR.py --IS_BASE True --IS_CAPTIONED True --MM img --IDX 0 --epochs 384 --learning_rate 7.5e-4
CUDA_VISIBLE_DEVICES=0 python main_MMR.py --IS_BASE True --IS_CAPTIONED True --MM img --IDX 1 --epochs 384 --learning_rate 7.5e-4
CUDA_VISIBLE_DEVICES=0 python main_MMR.py --IS_BASE True --IS_CAPTIONED True --MM img --IDX 2 --epochs 384 --learning_rate 7.5e-4
CUDA_VISIBLE_DEVICES=0 python main_MMR.py --IS_BASE True --IS_CAPTIONED True --MM txt --IDX 0  --epochs 48 --learning_rate 2e-4
CUDA_VISIBLE_DEVICES=0 python main_MMR.py --IS_BASE True --IS_CAPTIONED True --MM txt --IDX 1  --epochs 48 --learning_rate 2e-4
CUDA_VISIBLE_DEVICES=0 python main_MMR.py --IS_BASE True --IS_CAPTIONED True --MM txt --IDX 2  --epochs 48 --learning_rate 2e-4
CUDA_VISIBLE_DEVICES=0 python main_MMR.py --IS_BASE True --IS_CAPTIONED True --MM aud --IDX 0 --epochs 768 --learning_rate 7.5e-4
CUDA_VISIBLE_DEVICES=0 python main_MMR.py --IS_BASE True --IS_CAPTIONED True --MM aud --IDX 1 --epochs 768 --learning_rate 7.5e-4
CUDA_VISIBLE_DEVICES=0 python main_MMR.py --IS_BASE True --IS_CAPTIONED True --MM aud --IDX 2 --epochs 768 --learning_rate 7.5e-4

CUDA_VISIBLE_DEVICES=1 python main_MMR.py --IS_BASE False --IS_CAPTIONED True --MM img --IDX 0 --epochs 384 --learning_rate 7.5e-4
CUDA_VISIBLE_DEVICES=1 python main_MMR.py --IS_BASE False --IS_CAPTIONED True --MM img --IDX 1 --epochs 384 --learning_rate 7.5e-4
CUDA_VISIBLE_DEVICES=1 python main_MMR.py --IS_BASE False --IS_CAPTIONED True --MM img --IDX 2 --epochs 384 --learning_rate 7.5e-4
CUDA_VISIBLE_DEVICES=1 python main_MMR.py --IS_BASE False --IS_CAPTIONED True --MM txt --IDX 0 --epochs 48 --learning_rate 2e-4
CUDA_VISIBLE_DEVICES=1 python main_MMR.py --IS_BASE False --IS_CAPTIONED True --MM txt --IDX 1 --epochs 48 --learning_rate 2e-4
CUDA_VISIBLE_DEVICES=1 python main_MMR.py --IS_BASE False --IS_CAPTIONED True --MM txt --IDX 2 --epochs 48 --learning_rate 2e-4
CUDA_VISIBLE_DEVICES=1 python main_MMR.py --IS_BASE False --IS_CAPTIONED True --MM aud --IDX 0 --epochs 768 --learning_rate 1e-3
CUDA_VISIBLE_DEVICES=1 python main_MMR.py --IS_BASE False --IS_CAPTIONED True --MM aud --IDX 1 --epochs 768 --learning_rate 1e-3
CUDA_VISIBLE_DEVICES=1 python main_MMR.py --IS_BASE False --IS_CAPTIONED True --MM aud --IDX 2 --epochs 768 --learning_rate 1e-3

CUDA_VISIBLE_DEVICES=2 python main_MMR.py --IS_BASE True --IS_CAPTIONED False --MM img --IDX 0 --epochs 384 --learning_rate 7.5e-4
CUDA_VISIBLE_DEVICES=2 python main_MMR.py --IS_BASE True --IS_CAPTIONED False --MM img --IDX 1 --epochs 384 --learning_rate 7.5e-4
CUDA_VISIBLE_DEVICES=2 python main_MMR.py --IS_BASE True --IS_CAPTIONED False --MM img --IDX 2 --epochs 384 --learning_rate 7.5e-4
CUDA_VISIBLE_DEVICES=2 python main_MMR.py --IS_BASE True --IS_CAPTIONED False --MM txt --IDX 0  --epochs 48 --learning_rate 2e-4
CUDA_VISIBLE_DEVICES=2 python main_MMR.py --IS_BASE True --IS_CAPTIONED False --MM txt --IDX 1  --epochs 48 --learning_rate 2e-4
CUDA_VISIBLE_DEVICES=2 python main_MMR.py --IS_BASE True --IS_CAPTIONED False --MM txt --IDX 2  --epochs 48 --learning_rate 2e-4
CUDA_VISIBLE_DEVICES=2 python main_MMR.py --IS_BASE True --IS_CAPTIONED False --MM aud --IDX 0 --epochs 768 --learning_rate 7.5e-4
CUDA_VISIBLE_DEVICES=2 python main_MMR.py --IS_BASE True --IS_CAPTIONED False --MM aud --IDX 1 --epochs 768 --learning_rate 7.5e-4
CUDA_VISIBLE_DEVICES=2 python main_MMR.py --IS_BASE True --IS_CAPTIONED False --MM aud --IDX 2 --epochs 768 --learning_rate 7.5e-4

CUDA_VISIBLE_DEVICES=3 python main_MMR.py --IS_BASE False --IS_CAPTIONED False --MM img --IDX 0 --epochs 384 --learning_rate 7.5e-4
CUDA_VISIBLE_DEVICES=3 python main_MMR.py --IS_BASE False --IS_CAPTIONED False --MM img --IDX 1 --epochs 384 --learning_rate 7.5e-4
CUDA_VISIBLE_DEVICES=3 python main_MMR.py --IS_BASE False --IS_CAPTIONED False --MM img --IDX 2 --epochs 384 --learning_rate 7.5e-4
CUDA_VISIBLE_DEVICES=3 python main_MMR.py --IS_BASE False --IS_CAPTIONED False --MM txt --IDX 0 --epochs 48 --learning_rate 2e-4
CUDA_VISIBLE_DEVICES=3 python main_MMR.py --IS_BASE False --IS_CAPTIONED False --MM txt --IDX 1 --epochs 48 --learning_rate 2e-4
CUDA_VISIBLE_DEVICES=3 python main_MMR.py --IS_BASE False --IS_CAPTIONED False --MM txt --IDX 2 --epochs 48 --learning_rate 2e-4
CUDA_VISIBLE_DEVICES=3 python main_MMR.py --IS_BASE False --IS_CAPTIONED False --MM aud --IDX 0 --epochs 768 --learning_rate 1e-3
CUDA_VISIBLE_DEVICES=3 python main_MMR.py --IS_BASE False --IS_CAPTIONED False --MM aud --IDX 1 --epochs 768 --learning_rate 1e-3
CUDA_VISIBLE_DEVICES=3 python main_MMR.py --IS_BASE False --IS_CAPTIONED False --MM aud --IDX 2 --epochs 768 --learning_rate 1e-3
    """