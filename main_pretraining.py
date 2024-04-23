import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


from model import *
from dataset import *
from config import *
from utils import *

import warnings
warnings.filterwarnings(action='ignore')

MIN_LOSS = 9999

def save_model(model, IS_BASE, IS_CAPTIONED):
    model_sz = "base" if IS_BASE else "large"
    text_des = "caption" if IS_CAPTIONED else "prompt"
    ckpt_savepath = os.path.join(f'CLIP_model_{model_sz}_{text_des}.tar')
    torch.save({
        'model_state_dict': model.state_dict(),
    }, ckpt_savepath)

    modal_list = ['image', 'text', 'audio']
    for modal in modal_list:
        save_path = os.path.join(f"CLIP_{modal}_model_{model_sz}", text_des)
        if modal == 'image':
            model.vision_model.save_pretrained(save_path)
            projection_path = os.path.join(save_path, 'projection_head.tar')
            torch.save({
                    'model_state_dict': model.vision_projection.state_dict(),
                }, projection_path)
        elif modal == 'text':
            model.text_model.save_pretrained(save_path)
            projection_path = os.path.join(save_path, 'projection_head.tar')
            torch.save({
                    'model_state_dict': model.text_projection.state_dict(),
                }, projection_path)
        else:
            model.audio_model.save_pretrained(save_path)
            projection_path = os.path.join(save_path, 'projection_head.tar')
            torch.save({
                    'model_state_dict': model.audio_projection.state_dict(),
                }, projection_path)

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, port, args):
    setup(rank, world_size, port)
    
    global MIN_LOSS
    SEED = args.SEED; set_SEED(SEED)
    IS_BASE = args.IS_BASE
    IS_CAPTIONED = args.IS_CAPTIONED

    epochs = args.epochs

    batch_size = 35 if IS_BASE else 14
    accumulation_steps = 4 if IS_BASE else 8
    
    seq_max_length = args.seq_max_length
    lr = args.learning_rate
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    
    clip_config = CLIPConfig_BASE() if IS_BASE else CLIPConfig_LARGE()
    vision_model_path = clip_config.vision_config.model_link
    text_model_path   = clip_config.text_config.model_link
    audio_model_path  = clip_config.audio_config.model_link
    print(vision_model_path, text_model_path, audio_model_path)

    train_path = 'vgg_sound_train_captioned.csv'
    valid_path = 'vgg_sound_test_captioned.csv'
    test_path = 'vgg_sound_test_captioned.csv'
    train_data = pd.read_csv(train_path); train_data = train_data.reset_index()
    valid_data = pd.read_csv(valid_path); valid_data = valid_data.reset_index()
    test_data = pd.read_csv(test_path); test_data = test_data.reset_index()

    # img_processor = AutoProcessor.from_pretrained(vision_model_path)
    img_processor = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomResizedCrop(size=(224,224)),
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.15)
            ]),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.075, contrast=0.075, saturation=0.075, hue=0.075),
            ], p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize((0., 0., 0.), (1., 1., 1.))
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711))
        ])
    txt_processor = AutoTokenizer.from_pretrained(text_model_path)
    aud_processor = AutoProcessor.from_pretrained(audio_model_path)

    train_dataset = Dataset_Step1(train_data, img_processor, txt_processor, aud_processor, seq_max_length=seq_max_length, IS_CAPTIONED=IS_CAPTIONED)
    valid_dataset = Dataset_Step1(valid_data, img_processor, txt_processor, aud_processor, seq_max_length=seq_max_length, IS_CAPTIONED=IS_CAPTIONED)
    test_dataset = Dataset_Step1(test_data, img_processor, txt_processor, aud_processor, seq_max_length=seq_max_length, IS_CAPTIONED=IS_CAPTIONED)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
    # test_sampler  = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4*world_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=4*world_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, num_workers=4*world_size)


    model = Tri_CLIP(clip_config, 
                     vision_model_path=vision_model_path,
                     text_model_path=text_model_path, 
                     audio_model_path=audio_model_path)
    # model = model.apply(initialize_weights)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])# find_unused_parameters=True)
    opt = optim.AdamW(ddp_model.parameters(), lr=lr)#, weight_decay=0.01)
    # scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=np.linspace(0, epochs, 5)[1:-1].astype(int).tolist(), gamma=0.95)
    
    # CLIP 사전학습 #
    for e in range(epochs):
        ddp_model.train()
        train_sampler.set_epoch(e)
        valid_sampler.set_epoch(e)
        
        loss_lst = []
        img_loss_lst = []
        txt_loss_lst = []
        aud_loss_lst = []

        if rank == 0:
            train_pbar = tqdm(train_loader, file=sys.stdout)
        else:
            train_pbar = train_loader

        opt.zero_grad()
        for batch_idx, (images, audios, (input_ids, att_mask)) in enumerate(train_pbar):
            images, audios = images.to(rank), audios.to(rank)
            input_ids, att_mask = input_ids.to(rank), att_mask.to(rank)

            output = ddp_model(pixel_values=images,
                                input_ids=input_ids, att_mask=att_mask,
                                input_values=audios)
            IT, TA, AI = (output[0] * alpha), (output[1] * beta), (output[2] * gamma)
            loss = IT + TA + AI
            
            loss_lst.append(loss.item())
            img_loss_lst.append(IT.item()); txt_loss_lst.append(TA.item()); aud_loss_lst.append(AI.item()) 
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                opt.step()
                opt.zero_grad()

            if rank == 0:
                train_pbar.set_postfix(epoch=f'{e+1}/{epochs}', loss='{:.4f}, IT={:.4f}, TA={:.4f}, AI={:.4f}'.format(np.mean(loss_lst),
                                                                                                                      np.mean(img_loss_lst),
                                                                                                                      np.mean(txt_loss_lst), 
                                                                                                                      np.mean(aud_loss_lst)))
        if rank == 0:
            train_pbar.close()
            
        # 필요시 마지막에 남은 그래디언트를 처리
        if batch_idx % accumulation_steps != 0:
            opt.step()
            opt.zero_grad()
        
        ddp_model.eval()
        if rank == 0:
            valid_pbar = tqdm(valid_loader, file=sys.stdout)
        else:
            valid_pbar = valid_loader

        with torch.no_grad():
            loss_lst = []
            img_loss_lst = []
            txt_loss_lst = []
            aud_loss_lst = []
            
            for batch_idx, (images, audios, (input_ids, att_mask)) in enumerate(valid_pbar):
                images, audios = images.to(rank), audios.to(rank)
                input_ids, att_mask = input_ids.to(rank), att_mask.to(rank)

                output = ddp_model(pixel_values=images,
                                    input_ids=input_ids, att_mask=att_mask,
                                    input_values=audios)
                IT, TA, AI = (output[0] * alpha), (output[1] * beta), (output[2] * gamma)
                loss = IT + TA + AI
                
                loss_lst.append(loss.item())
                img_loss_lst.append(IT.item()); txt_loss_lst.append(TA.item()); aud_loss_lst.append(AI.item()) 

                if rank == 0:
                    valid_pbar.set_postfix(loss='{:.4f}, IT={:.4f}, TA={:.4f}, AI={:.4f}'.format(np.mean(loss_lst),
                                                                                                 np.mean(img_loss_lst),
                                                                                                 np.mean(txt_loss_lst), 
                                                                                                 np.mean(aud_loss_lst)))
            if rank == 0:
                valid_pbar.close()
                print()
                if MIN_LOSS > np.mean(loss_lst):
                    MIN_LOSS = np.mean(loss_lst)
                    save_model(model, IS_BASE, IS_CAPTIONED)

        # scheduler.step()

    ddp_model.eval()
    if rank == 0:        
        test_pbar = tqdm(test_loader, file=sys.stdout)
        with torch.no_grad():
            loss_lst = []
            img_loss_lst = []
            txt_loss_lst = []
            aud_loss_lst = []
            
            for batch_idx, (images, audios, (input_ids, att_mask)) in enumerate(test_pbar):
                images, audios = images.to(rank), audios.to(rank)
                input_ids, att_mask = input_ids.to(rank), att_mask.to(rank)

                output = ddp_model(pixel_values=images,
                                    input_ids=input_ids, att_mask=att_mask,
                                    input_values=audios)
                IT, TA, AI = (output[0] * alpha), (output[1] * beta), (output[2] * gamma)
                loss = IT + TA + AI
                
                loss_lst.append(loss.item())
                img_loss_lst.append(IT.item()); txt_loss_lst.append(TA.item()); aud_loss_lst.append(AI.item()) 

                test_pbar.set_postfix(loss='{:.4f}, IT={:.4f}, TA={:.4f}, AI={:.4f}'.format(np.mean(loss_lst),
                                                                                            np.mean(img_loss_lst),
                                                                                            np.mean(txt_loss_lst), 
                                                                                            np.mean(aud_loss_lst)))
            if MIN_LOSS > np.mean(loss_lst):
                MIN_LOSS = np.mean(loss_lst)
                save_model(model, IS_BASE, IS_CAPTIONED)
        test_pbar.close()
                
    cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description='STEP 1 : CLIP-PreTraining')
    parser.add_argument('--SEED', type=int, default=17, help='Random Seed')
    parser.add_argument('--WORLD_SIZE', type=int, default=2, help='number of distributed processes')
    parser.add_argument('--PORT', type=str, default='12356', help='number of Master PORT Number')
    parser.add_argument('--IS_BASE', type=str2bool, default=True, help='Model Size : True is "BASE" | False is "Large"')
    parser.add_argument('--IS_CAPTIONED', type=str2bool, default=True, help='Text Description : True is "Caption" | False is "Prompt"')

    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--seq_max_length', type=int, default=32, help='Max Sequence Length for TextModel')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Max Sequence Length for TextModel')
    parser.add_argument('--alpha', type=float, default=1.0, help='Hyperparameter for Vision Loss')
    parser.add_argument('--beta', type=float, default=1.0, help='Hyperparameter for Text Loss')
    parser.add_argument('--gamma', type=float, default=1.0, help='Hyperparameter for Audio Loss')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(main, args=(args.WORLD_SIZE, args.PORT, args), nprocs=args.WORLD_SIZE, join=True)
    """
CUDA_VISIBLE_DEVICES=0,1 python main_pretraining.py --SEED 77 --WORLD_SIZE 2 --PORT 12345 --IS_BASE True --IS_CAPTIONED True --learning_rate 5e-6
CUDA_VISIBLE_DEVICES=0,1 python main_pretraining.py --SEED 42 --WORLD_SIZE 2 --PORT 12346 --IS_BASE False --IS_CAPTIONED True --learning_rate 5e-6

CUDA_VISIBLE_DEVICES=2,3 python main_pretraining.py --SEED 77 --WORLD_SIZE 2 --PORT 12347 --IS_BASE True --IS_CAPTIONED False --learning_rate 5e-6
CUDA_VISIBLE_DEVICES=2,3 python main_pretraining.py --SEED 42 --WORLD_SIZE 2 --PORT 12348 --IS_BASE False --IS_CAPTIONED False --learning_rate 5e-6

    """
    