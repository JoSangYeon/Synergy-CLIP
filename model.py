import random
from PIL import Image
from itertools import tee
from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler

import torchvision
import torchaudio
from transformers import AutoProcessor, CLIPVisionModel
from transformers import AutoTokenizer, AutoModel, CLIPTextModel
from transformers import ASTModel

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM # https://github.com/VainF/pytorch-msssim?tab=readme-ov-file
import piqa
# class SSIM_Loss(SSIM):
#     def forward(self, img1, img2):
#         return 1. - super(SSIM_Loss, self).forward(img1, img2)

# class MS_SSIM_Loss(MS_SSIM):
#     def forward(self, img1, img2):
#         return 1. - super(MS_SSIM_Loss, self).forward(img1, img2)

#https://www.geeksforgeeks.org/python-pair-iteration-in-list/
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(model.weight.data, 0, 0.02)
    elif classname.find('Parameter') != -1:
        nn.init.normal_(model.data, 0, 0.02)

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

class Tri_CLIP(nn.Module):
    def __init__(self, config,
                 vision_model_path='openai/clip-vit-base-patch16',
                 text_model_path='openai/clip-vit-base-patch16',
                 audio_model_path='MIT/ast-finetuned-audioset-10-10-0.4593'):
        super(Tri_CLIP, self).__init__()
        self.config = config
        self.vision_config = config.vision_config
        self.text_config = config.text_config
        self.audio_config = config.audio_config
        
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_path)
        self.text_model = AutoModel.from_pretrained(text_model_path)
        # self.text_model = CLIPTextModel.from_pretrained(text_model_path)
        self.audio_model = ASTModel.from_pretrained(audio_model_path)
        
        self.vision_projection = nn.Linear(self.vision_config.hidden_size, self.config.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_config.hidden_size, self.config.projection_dim, bias=False)
        self.audio_projection = nn.Linear(self.audio_config.hidden_size, self.config.projection_dim, bias=False)
        
        self.logit_scale_for_IT = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.logit_scale_for_TA = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.logit_scale_for_AI = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        
    def get_image_features(self, pixel_values):
        vision_outputs = self.vision_model(
            pixel_values = pixel_values,
            output_attentions = self.vision_config.output_attentions,
            output_hidden_states = self.vision_config.output_hidden_states,
            return_dict = self.config.return_dict
        )
        
        pooler_output = vision_outputs[1] # pooler output
        img_features = self.vision_projection(pooler_output)
        return img_features
    def get_text_features(self, input_ids, att_mask, pos_ids):
        text_outputs = self.text_model(
            input_ids = input_ids,
            attention_mask = att_mask,
            position_ids = pos_ids,
            output_attentions = self.text_config.output_attentions,
            output_hidden_states = self.text_config.output_hidden_states,
            return_dict = self.config.return_dict
        )
        txt_embeds = text_outputs[1] # pooler output
        txt_features = self.text_projection(txt_embeds)
        
        # txt_embeds = text_outputs[0] # last_hidden_state : (batch, len, dim)
        # batch_sz, seq_len, dim = txt_embeds.shape
        # txt_embeds = txt_embeds[att_mask.unsqueeze(-1).repeat(1, 1, dim) == 1].view(batch_sz, -1, dim)
        # txt_embeds = F.adaptive_avg_pool2d(txt_embeds, (1,None)).squeeze(1) # (batch, dim)
        # txt_features = self.text_projection(txt_embeds)
        return txt_features
    def get_audio_features(self, input_values, head_mask):
        audio_outputs = self.audio_model(
            input_values = input_values,
            head_mask = head_mask,
            output_attentions = self.audio_config.output_attentions,
            output_hidden_states = self.audio_config.output_hidden_states,
            return_dict = self.config.return_dict
        )
        
        pooler_output = audio_outputs[1] # pooler output
        aud_features = self.audio_projection(pooler_output)
        return aud_features
    
    def get_img_txt_sim_score(self, pixel_values=None, input_ids=None, 
                              att_mask=None, pos_ids=None,):
        # for image-text Zero shot classification
        
        vision_outputs = self.vision_model(
            pixel_values = pixel_values,
            output_attentions = self.vision_config.output_attentions,
            output_hidden_states = self.vision_config.output_hidden_states,
            return_dict = self.config.return_dict
        )
        
        text_outputs = self.text_model(
            input_ids = input_ids,
            attention_mask = att_mask,
            position_ids = pos_ids,
            output_attentions = self.text_config.output_attentions,
            output_hidden_states = self.text_config.output_hidden_states,
            return_dict = self.config.return_dict
        )
        
        img_embeds = vision_outputs[1] # pooler output
        img_embeds = self.vision_projection(img_embeds)
        
        txt_embeds = text_outputs[1] # pooler output
        txt_embeds = self.text_projection(txt_embeds)
        # txt_embeds = text_outputs[0] # last_hidden_state : (batch, len, dim)
        # batch_sz, seq_len, dim = txt_embeds.shape
        # txt_embeds = txt_embeds[att_mask.unsqueeze(-1).repeat(1, 1, dim) == 1].view(batch_sz, -1, dim)
        # txt_embeds = F.adaptive_avg_pool2d(txt_embeds, (1,None)).squeeze(1) # (batch, dim)
        # txt_embeds = self.text_projection(txt_embeds)
        
        # normalized features
        img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
        txt_embeds = txt_embeds / txt_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # cosine similarity as logits : image - text
        logit_scale_IT = self.logit_scale_for_IT.exp()
        logits_img_txt = torch.matmul(img_embeds, txt_embeds.t()) * logit_scale_IT
        return logits_img_txt

    def get_aud_txt_sim_score(self, input_ids=None, att_mask=None, pos_ids=None,
                              input_values=None, head_mask=None,):
        text_outputs = self.text_model(
            input_ids = input_ids,
            attention_mask = att_mask,
            position_ids = pos_ids,
            output_attentions = self.text_config.output_attentions,
            output_hidden_states = self.text_config.output_hidden_states,
            return_dict = self.config.return_dict
        )
        
        audio_outputs = self.audio_model(
            input_values = input_values,
            head_mask = head_mask,
            output_attentions = self.audio_config.output_attentions,
            output_hidden_states = self.audio_config.output_hidden_states,
            return_dict = self.config.return_dict
        )
        
        txt_embeds = text_outputs[1] # pooler output
        txt_embeds = self.text_projection(txt_embeds)
        # txt_embeds = text_outputs[0] # last_hidden_state : (batch, len, dim)
        # batch_sz, seq_len, dim = txt_embeds.shape
        # txt_embeds = txt_embeds[att_mask.unsqueeze(-1).repeat(1, 1, dim) == 1].view(batch_sz, -1, dim)
        # txt_embeds = F.adaptive_avg_pool2d(txt_embeds, (1,None)).squeeze(1) # (batch, dim)
        # txt_embeds = self.text_projection(txt_embeds)
        
        aud_embeds = audio_outputs[1] # pooler output
        aud_embeds = self.audio_projection(aud_embeds)
        
        # normalized features
        txt_embeds = txt_embeds / txt_embeds.norm(p=2, dim=-1, keepdim=True)
        aud_embeds = aud_embeds / aud_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # cosine similarity as logits : audio-text
        logit_scale_TA = self.logit_scale_for_TA.exp()
        logits_txt_aud = torch.matmul(txt_embeds, aud_embeds.t()) * logit_scale_TA
        return logits_txt_aud
        
    def forward(self, pixel_values=None,
                input_ids=None, att_mask=None, pos_ids=None,
                input_values=None, head_mask=None,):
        
        vision_outputs = self.vision_model(
            pixel_values = pixel_values,
            output_attentions = self.vision_config.output_attentions,
            output_hidden_states = self.vision_config.output_hidden_states,
            return_dict = self.config.return_dict
        )
        
        text_outputs = self.text_model(
            input_ids = input_ids,
            attention_mask = att_mask,
            position_ids = pos_ids,
            output_attentions = self.text_config.output_attentions,
            output_hidden_states = self.text_config.output_hidden_states,
            return_dict = self.config.return_dict
        )
        
        audio_outputs = self.audio_model(
            input_values = input_values,
            head_mask = head_mask,
            output_attentions = self.audio_config.output_attentions,
            output_hidden_states = self.audio_config.output_hidden_states,
            return_dict = self.config.return_dict
        )
        
        img_embeds = vision_outputs[1] # pooler output
        img_embeds = self.vision_projection(img_embeds)
        
        txt_embeds = text_outputs[1] # pooler output
        txt_embeds = self.text_projection(txt_embeds)
        # txt_embeds = text_outputs[0] # last_hidden_state : (batch, len, dim)
        # batch_sz, seq_len, dim = txt_embeds.shape
        # txt_embeds = txt_embeds[att_mask.unsqueeze(-1).repeat(1, 1, dim) == 1].view(batch_sz, -1, dim)
        # txt_embeds = F.adaptive_avg_pool2d(txt_embeds, (1,None)).squeeze(1) # (batch, dim)
        # txt_embeds = self.text_projection(txt_embeds)
        
        aud_embeds = audio_outputs[1] # pooler output
        aud_embeds = self.audio_projection(aud_embeds)
        
        # normalized features
        img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
        txt_embeds = txt_embeds / txt_embeds.norm(p=2, dim=-1, keepdim=True)
        aud_embeds = aud_embeds / aud_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # cosine similarity as logits
        ## 1. image - text
        logit_scale_IT = self.logit_scale_for_IT.exp()
        logits_IT_per_image = torch.matmul(img_embeds, txt_embeds.t()) * logit_scale_IT
        # logits_IT_per_text = logits_IT_per_image.t()
        
        ## 2. text - audio
        logit_scale_TA = self.logit_scale_for_TA.exp()
        logits_TA_per_text = torch.matmul(txt_embeds, aud_embeds.t()) * logit_scale_TA
        # logits_TA_per_audio = logits_TA_per_text.t()
        
        ## 3. audio - image
        logit_scale_AI = self.logit_scale_for_AI.exp()
        logits_AI_per_aud = torch.matmul(aud_embeds, img_embeds.t()) * logit_scale_AI
        # logits_AI_per_text = logits_AI_per_aud.t()
        
        if self.config.is_PT:
            IT_loss = clip_loss(logits_IT_per_image)
            TA_loss = clip_loss(logits_TA_per_text)
            AI_loss = clip_loss(logits_AI_per_aud)
            return IT_loss, TA_loss, AI_loss
            # return (IT_loss * 0.15) + TA_loss + AI_loss
        else:
            if self.config.return_logits:
                return ((logits_IT_per_image, logits_TA_per_text, logits_AI_per_aud),
                        img_embeds, txt_embeds, aud_embeds)
            elif self.config.return_lhs:
                return (vision_outputs[0], text_outputs[0], audio_outputs[0])
            else:
                return img_embeds, txt_embeds, aud_embeds

# https://visionhong.tistory.com/m/38
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# Attention, MLP 이전에 수행되는 Layer Normalization
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 각 쿼리(패치)가 다른 패치와 어느정도 연관성을 가지는지 구하는것이 바로 attention의 목적.
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads  # 512
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads  # multi head attention (시퀀스를 병렬로 분할함으로써 다르게 주의를 기울이고 다양한 특징을 얻을 수 있다고 함)
        self.scale = dim_head ** -0.5  # 큰값을 softmax에 올리면 gradient vanishing이 일어나기 때문에 downscale에 사용될 값 (softmax 함수 그래프를 보면 이해가능)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # query, key, value로 분할하기 위해 3을 곱해줌

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # embed dim 기준으로 3분할 (튜플로 감싸져 있음)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # q = k = v (b, heads, num_patches, dim)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # query와 key간의 dot product를 위한 차원변경 + scaling
        # dots = (b, heads, num_patches, dim) * (b, heads, dim, num_patches) = (b, heads, num_patches, num_patches)

        attn = self.attend(dots)  # self attention (각 패치간의 연관성을 softmax 확률값으로 나타냄)

        out = torch.matmul(attn, v)  # 구한 확률값을 실제 값(value)에 dot product 시킴 (원래 차원으로 복구) (b, heads, num_patches, dim)
        # out = (b, heads, num_patches, num_patches) * (b, heads, num_patches, dim) = (b, heads, num_patches, dim)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)  # 원래 dim으로 복귀

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # skip connection
            x = ff(x) + x
        return x        

class MultiModal_Encoder(nn.Module):
    def __init__(self, config):
        super(MultiModal_Encoder, self).__init__()

        self.config = config

        encoder_dim = config.encoder_dim
        bottleneck_dim = config.encoder_bottleneck_dim
        dropout = config.encoder_dropout
        decoder_dim = config.decoder_dim

        self.encoder = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, encoder_dim),
        )

        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)

    def forward(self, x):
        # x : (batch, 1, encder_dim)
        x = x + self.encoder(x)
        x = self.enc_to_dec(x)
        return x

class PixelShuffle(nn.Module):
    def __init__(self, input_dim, output_dim, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.relu(x)
        return x

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        max_pool = F.adaptive_max_pool2d(x, 1)
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        ca = self.channel_attention(max_pool) + self.channel_attention(avg_pool)
        x = x * ca

        # Spatial attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_attention(spatial)
        x = x * sa
        return x

class MLP(nn.Module):
    # Multi-layer perceptron module
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
    def forward(self, x):
        b, ch, h, w = x.size()

        x = x.view(b, -1, h*w)
        x = self.layer(x)
        x = x.view(b, ch, h, w)
        return x

class Recon_Block(nn.Module):
    def __init__(self, 
                 in_ch, out_ch, out_h, out_w, 
                 hidden_dim=3072, upscale_p=2):
        super(Recon_Block, self).__init__()
        
        self.pixel_shuffle = PixelShuffle(in_ch, out_ch, upscale_p) if upscale_p > 1 else nn.Identity()

        self.attention = nn.Sequential(
            nn.LayerNorm([out_ch, out_h, out_w]),
            CBAM(out_ch),
        )
        self.feedforward = nn.Sequential(
            nn.LayerNorm([out_ch, out_h, out_w]),
            MLP(out_h*out_w, hidden_dim),
        )
    
    def forward(self, x):
        x = self.pixel_shuffle(x)
        
        att = self.attention(x)
        x = x + att
        
        ff = self.feedforward(x)
        x = x + ff
        
        return x

class IMG_Decoder(nn.Module):
    def __init__(self, config):
        super(IMG_Decoder, self).__init__()
        self.config = config

        decoder_dim = config.decoder_dim
        channels = config.img_channels
        layer_info = config.img_decoder_layer_info
        paired_layer_info = list(map(lambda x: (x[0], x[1]), pairwise(layer_info)))

        _, init_ch, init_h, init_w = layer_info[0]
        self.initial_layer = nn.Sequential(
            nn.Linear(decoder_dim, init_ch * init_h * init_w),  # Expand feature dimensions
            nn.Unflatten(1, (init_ch, init_h, init_w)),        # Unflatten to (B, C, H, W)
        )

        recon_layer = OrderedDict()
        for idx, ((up_p, in_ch, in_h, in_w), (_, out_ch, out_h, out_w)) in enumerate(paired_layer_info, start=1):
            recon_layer[f'recon_layer_{idx}'] = Recon_Block(in_ch, out_ch, out_h, out_w, upscale_p=up_p)
            # recon_layer[f'batchnorm_{idx}'] = nn.BatchNorm2d(out_ch)
        self.recon_layer = nn.Sequential(recon_layer)

        self.final_layer = nn.Sequential(
            PixelShuffle(out_ch, channels, 2),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=1, stride=1, padding=0),
        )
    
    def forward(self, img_embed):
        """
        inputs:
            img_embed : batch, dim
        outputs:
            image     : batch, ch, w, h
        """
        x = self.initial_layer(img_embed)
        x = self.recon_layer(x)
        x = self.final_layer(x)
        return x
    
class TXT_Decoder(nn.Module):
    def __init__(self, config,):
        super(TXT_Decoder, self).__init__()
        self.config = config
        
        txt_tokens = config.txt_tokens  # 64
        txt_length = config.text_length # 32
        vocab_size = config.text_config.vocab_size
        
        decoder_dim = config.decoder_dim
        decoder_depth = config.decoder_depth
        decoder_heads = config.decoder_heads
        decoder_head_dim = config.decoder_head_dim
        decoder_mlp_dim = config.decoder_mlp_dim
        decoder_dropout = config.decoder_dropout
        
        self.dim_to_tokens = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=txt_tokens,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(txt_tokens),
            nn.ReLU(),
            nn.Conv1d(in_channels=txt_tokens, out_channels=txt_length,
                      kernel_size=1, stride=1, padding=0)
        )
        
        self.decoder_pos_emb = nn.Parameter(torch.randn(1, txt_length, decoder_dim)) # nn.Embedding(num_patches, decoder_dim)
        decoder_layer = nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=decoder_heads, 
                                                   dropout=decoder_dropout,
                                                   activation='gelu', batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        # self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, 
        #                            dim_head=decoder_head_dim, mlp_dim=decoder_mlp_dim, dropout=decoder_dropout)
        
        self.to_words = nn.Linear(decoder_dim, vocab_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.to_words.bias = self.bias
        
    def forward(self, txt_embed):
        """
        inputs:
            txt_embed : batch, 1, dim
        outputs:
            vision : batch, decode_words, vocab_size
        """
        batch, tokens, *_ = txt_embed.shape
        
        decoder_tokens = self.dim_to_tokens(txt_embed) + self.decoder_pos_emb # batch, txt_tokens, decoder_dim -> batch, decode_words, decoder_dim
        decoded_tokens = self.decoder(decoder_tokens) # batch, decode_words, decoder_dim
        
        pred_words_values = self.to_words(decoded_tokens) # batch, decode_words, vocab_size

        return pred_words_values

class AUD_Decoder(nn.Module):
    def __init__(self, config,):
        super(AUD_Decoder, self).__init__()
        self.config = config

        decoder_dim = config.decoder_dim
        channels = config.aud_channels
        layer_info = config.aud_decoder_layer_info
        paired_layer_info = list(map(lambda x: (x[0], x[1]), pairwise(layer_info)))

        _, init_ch, init_h, init_w = layer_info[0]
        self.initial_layer = nn.Sequential(
            nn.Linear(decoder_dim, init_h * init_w * init_ch),  # Expand feature dimensions
            nn.Unflatten(1, (init_ch, init_h, init_w)),        # Unflatten to (B, C, H, W)
        )

        recon_layer = OrderedDict()
        for idx, ((up_p, in_ch, in_h, in_w), (_, out_ch, out_h, out_w)) in enumerate(paired_layer_info, start=1):
            recon_layer[f'recon_layer_{idx}'] = Recon_Block(in_ch, out_ch, out_h, out_w, upscale_p=up_p)
        self.recon_layer = nn.Sequential(recon_layer)

        self.final_layer = nn.Sequential(
            PixelShuffle(out_ch, channels, 2),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=1, stride=1, padding=0),
        )
    
    def forward(self, aud_embed, is_squeeze=True):
        """
        inputs:
            aud_embed : batch, dim
        outputs:
            audio : batch, w, h
        """
        x = self.initial_layer(aud_embed)

        x = self.recon_layer(x)

        x = self.final_layer(x)
        
        return x.squeeze(1) if is_squeeze else x

class IMG_TXT_2_AUD(nn.Module):
    def __init__(self, config, img_encoder, txt_encoder):
        super(IMG_TXT_2_AUD, self).__init__()
        self.config = config

        self.img_encoder = img_encoder
        self.txt_encoder = txt_encoder

        encoder_dim = config.encoder_dim
        self.img_dim_mapper = nn.Linear(config.vision_config.hidden_size, encoder_dim, bias=False)
        self.txt_dim_mapper = nn.Linear(config.text_config.hidden_size, encoder_dim, bias=False)

        self.mm_encoder = MultiModal_Encoder(config)

        self.decoder = AUD_Decoder(config)

        audio_height, audio_width = config.aud_size # 256x64
        aud_channels = config.aud_channels
        self.label_audios = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(audio_height, audio_width),
                                          interpolation=Image.BICUBIC),
        ])

    def get_loss(self, recon_aud, label_aud, alpha=0.75):
        label_aud = self.label_audios(label_aud.unsqueeze(1)).squeeze(1)
        recon_loss = F.mse_loss(recon_aud, label_aud) #+ F.l1_loss(recon_aud, label_aud)
        ssim_loss = 1 - ssim(recon_aud.unsqueeze(1), label_aud.unsqueeze(1), 
                             data_range=1.0, size_average=True)
        return (alpha * ssim_loss) + ((1-alpha) * recon_loss), recon_loss, ssim_loss

    def forward(self, images, input_ids, att_mask, label=None, alpha=0.75):
        """
        inputs:
            images: batch, ch, w, h
            text  : input_ids, att_maks > batch, seq_len
        outputs:
            audio : batch, aud_w, aud_h
        """
        # modality fusion -> simple sum?
        with torch.no_grad():
            img_outputs = self.img_encoder(
                pixel_values = images,
                output_attentions = self.config.return_att,
                output_hidden_states = self.config.return_lhs,
                return_dict = self.config.return_dict)
            txt_outputs = self.txt_encoder(
                input_ids = input_ids, attention_mask = att_mask,
                output_attentions = self.config.return_att,
                output_hidden_states = self.config.return_lhs,
                return_dict = self.config.return_dict)
            
        # all tokens embed -> avgpool #
        # img_lhs = img_outputs[0]
        # txt_lhs = txt_outputs[0]
        # img_embed = self.img_dim_mapper(F.adaptive_avg_pool2d(img_lhs, (1, None)))
        # txt_embed = self.txt_dim_mapper(F.adaptive_avg_pool2d(txt_lhs, (1, None)))
        
        # # # cls token embed #
        img_embed = self.img_dim_mapper(img_outputs[1]).unsqueeze(1)
        txt_embed = self.txt_dim_mapper(txt_outputs[1]).unsqueeze(1)

        mm_embed = img_embed + txt_embed

        # encoding
        mm_embed = self.mm_encoder(mm_embed).squeeze(1)

        # decoding
        recon_aud = self.decoder(mm_embed)

        if label is None:
            return recon_aud
        else:
            return self.get_loss(recon_aud, label, alpha)

class TXT_AUD_2_IMG(nn.Module):
    def __init__(self, config, txt_encoder, aud_encoder):
        super(TXT_AUD_2_IMG, self).__init__()
        self.config = config

        self.txt_encoder = txt_encoder
        self.aud_encoder = aud_encoder

        encoder_dim = config.encoder_dim
        self.txt_dim_mapper = nn.Linear(config.text_config.hidden_size, encoder_dim, bias=False)
        self.aud_dim_mapper = nn.Linear(config.audio_config.hidden_size, encoder_dim, bias=False)

        self.mm_encoder = MultiModal_Encoder(config)

        self.decoder = IMG_Decoder(config)

        image_size = config.img_size # 128x128
        img_channels = config.img_channels
        self.label_images = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(image_size, image_size), 
                                          interpolation=Image.BICUBIC),
        ])

    def get_loss(self, recon_img, label_img, alpha=0.75):
        label_img = self.label_images(label_img)
        recon_loss = F.mse_loss(recon_img, label_img) + F.l1_loss(recon_img, label_img)
        ssim_loss = 1 - ssim(recon_img, label_img, data_range=1.0, size_average=True)
        return (alpha * ssim_loss) + ((1-alpha) * recon_loss), recon_loss, ssim_loss

    def forward(self, input_ids, att_mask, audios, label=None, alpha=0.75):
        """
        inputs:
            text   : input_ids, att_maks > batch, seq_len
            audios : batch, h, w
        outputs:
            images : batch, ch, img_h, img_w
        """

        # modality fusion -> simple sum?
        with torch.no_grad():
            txt_outputs = self.txt_encoder(
                input_ids = input_ids, attention_mask = att_mask,
                output_attentions = self.config.return_att,
                output_hidden_states = self.config.return_lhs,
                return_dict = self.config.return_dict)
            aud_outputs = self.aud_encoder(
                input_values = audios,
                output_attentions = self.config.return_att,
                output_hidden_states = self.config.return_lhs,
                return_dict = self.config.return_dict)
            
        # all tokens embed -> avgpool #
        txt_lhs = txt_outputs[0]
        aud_lhs = aud_outputs[0]
        txt_embed = self.txt_dim_mapper(F.adaptive_avg_pool2d(txt_lhs, (1, None)))
        aud_embed = self.aud_dim_mapper(F.adaptive_avg_pool2d(aud_lhs, (1, None)))
        
        # # cls token embed #
        # txt_embed = self.txt_dim_mapper(txt_outputs[1]).unsqueeze(1)
        # aud_embed = self.aud_dim_mapper(aud_outputs[1]).unsqueeze(1)

        mm_embed = txt_embed + aud_embed

        # encoding
        mm_embed = self.mm_encoder(mm_embed).squeeze(1)

        # decoding
        recon_aud = self.decoder(mm_embed)

        if label is None:
            return recon_aud
        else:
            return self.get_loss(recon_aud, label, alpha)

class IMG_AUD_2_TXT(nn.Module):
    def __init__(self, config, img_encoder, aud_encoder):
        super(IMG_AUD_2_TXT, self).__init__()
        self.config = config

        self.img_encoder = img_encoder
        self.aud_encoder = aud_encoder

        encoder_dim = config.encoder_dim
        self.img_dim_mapper = nn.Linear(config.vision_config.hidden_size, encoder_dim, bias=False)
        self.aud_dim_mapper = nn.Linear(config.audio_config.hidden_size, encoder_dim, bias=False)

        self.mm_encoder = MultiModal_Encoder(config)

        self.decoder = TXT_Decoder(config)

    def get_loss(self, recon_txt, input_ids):
        recon_loss = F.cross_entropy(recon_txt.view(-1, self.config.text_config.vocab_size),
                                     input_ids.view(-1),) #ignore_index=self.config.text_config.pad_token_id)
        correts = (recon_txt.argmax(dim=-1).view(-1) == input_ids.view(-1)).sum() / len(input_ids.view(-1))
        return recon_loss, correts, torch.tensor(0)

    def forward(self, images, audios, label=None):
        """
        inputs:
            images : batch, ch, img_h, img_w
            audios : batch, aud_h, aud_w
        outputs:
            text   : input_ids
        """

        with torch.no_grad():
            img_outputs = self.img_encoder(
                pixel_values = images,
                output_attentions = self.config.return_att,
                output_hidden_states = self.config.return_lhs,
                return_dict = self.config.return_dict)
            aud_outputs = self.aud_encoder(
                input_values = audios,
                output_attentions = self.config.return_att,
                output_hidden_states = self.config.return_lhs,
                return_dict = self.config.return_dict)
            
        # all tokens embed -> avgpool #
        img_lhs = img_outputs[0]
        aud_lhs = aud_outputs[0]
        img_embed = self.img_dim_mapper(F.adaptive_avg_pool2d(img_lhs, (1, None)))
        aud_embed = self.aud_dim_mapper(F.adaptive_avg_pool2d(aud_lhs, (1, None)))
        
        # # cls token embed #
        # img_embed = self.img_dim_mapper(img_outputs[1]).unsqueeze(1)
        # aud_embed = self.aud_dim_mapper(aud_outputs[1]).unsqueeze(1)

        mm_embed = img_embed + aud_embed

        # encoding
        mm_embed = self.mm_encoder(mm_embed)

        # decoding
        recon_aud = self.decoder(mm_embed)

        if label is None:
            return recon_aud
        else:
            return self.get_loss(recon_aud, label)
