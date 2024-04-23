from dataclasses import dataclass



##### Base Model #####
@dataclass
class VisionConfig_BASE:
    model_link = 'openai/clip-vit-base-patch16'
    
    attention_dropout = 0.0
    dropout = 0.0
    hidden_act = "quick_gelu"
    hidden_size = 768
    image_size = 224
    initializer_factor = 1.0
    initializer_range = 0.02
    intermediate_size = 3072
    layer_norm_eps = 1e-05
    num_attention_heads = 12
    num_hidden_layers = 12
    num_channels = 3
    patch_size = 16
    
    output_attentions = False
    output_hidden_states = False
    
@dataclass
class TextConfig_BASE:
    model_link = 'FacebookAI/roberta-base'
    # model_link = 'FacebookAI/xlm-roberta-base'
    
    attention_dropout = 0.1
    hidden_dropout = 0.1
    dropout = 0.1
    hidden_act = "gelu"
    bos_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    hidden_size = 768
    initializer_factor = 1.0
    initializer_range = 0.02
    intermediate_size = 3072
    layer_norm_eps = 1e-05
    max_position_embeddings = 514
    position_embedding_type= "absolute",
    num_attention_heads = 12
    num_hidden_layers = 12
    vocab_size = 50265
    # vocab_size = 250002
    
    output_attentions = False
    output_hidden_states = False
    
# @dataclass
# class TextConfig_BASE:
#     # model_link = 'FacebookAI/roberta-base'
#     model_link = 'openai/clip-vit-base-patch16'
    
#     attention_dropout = 0.0
#     hidden_dropout = 0.0
#     dropout = 0.0
#     hidden_act = "guick_gelu"
#     bos_token_id = 0
#     pad_token_id = 1
#     eos_token_id = 2
#     hidden_size = 512
#     initializer_factor = 1.0
#     initializer_range = 0.02
#     intermediate_size = 2048
#     layer_norm_eps = 1e-05
#     max_position_embeddings = 77
#     position_embedding_type= "absolute",
#     num_attention_heads = 8
#     num_hidden_layers = 12
#     # vocab_size = 50265
#     vocab_size = 49408
    
#     output_attentions = False
#     output_hidden_states = False
        
@dataclass
class AudioConfig_BASE:
    model_link = 'MIT/ast-finetuned-audioset-12-12-0.447'
    
    attention_probs_dropout_prob = 0.0
    hidden_dropout_prob = 0.0
    frequency_stride = 12
    time_stride: 12
    hidden_size = 768
    hidden_act = 'glue'
    initializer_range = 0.02
    intermediate_size = 3072
    qkv_bias = True
    layer_norm_eps: 1e-12
    max_length = 1024
    num_attention_heads = 12
    num_hidden_layers = 12
    num_mel_bins = 128
    patch_size = 16
    
    output_attentions = False
    output_hidden_states = False

@dataclass
class CLIPConfig_BASE:
    num_of_modality = 3
    
    is_PT = True
    return_logits = False
    return_lhs = False
    
    logit_scale_init_value = 2.6592
    projection_dim = 768
    return_dict = True
    
    vision_config = VisionConfig_BASE
    text_config = TextConfig_BASE
    audio_config = AudioConfig_BASE
    
@dataclass
class ReconstructionConfig_BASE:
    IS_BASE = True
    num_of_modality = 3
    
    is_PT = False
    return_logits = False
    return_att = False
    return_lhs = True
    return_dict = True
    
    # in Multi_Modal_Encoder
    # vision_length -> img_tokens
    # text_length -> txt_tokens
    # audio_length -> aud_tokens
    projection_dim = 768

    vision_length = 197
    text_length   = 32 # seq_max_length in main_*.py line 51, 83
    audio_length  = 852
    img_tokens = 96
    txt_tokens = 96
    aud_tokens = 96
    
    # MM Encoder args
    encoder_dim = 768
    encoder_bottleneck_dim = 128
    encoder_dropout = 0.1

    # Img Decoder : Reconstruction args
    img_size = 128
    img_channels = 3
    img_decoder_layer_info = [
        [2, 1024, 4, 4],
        [2, 512, 8, 8],
        [2, 256, 16, 16],
        [2, 128, 32, 32],
        [2, 64, 64, 64],
    ]
    # img_decoder_layer_info = [       
    #     [2, 1024, 4, 4],

    #     [0, 512, 8, 8],
    #     [2, 512, 8, 8],
        
    #     [0, 256, 16, 16],
    #     [2, 256, 16, 16],

    #     [0, 128, 32, 32],
    #     [2, 128, 32, 32],
        
    #     [0, 64, 64, 64],
    #     [2, 64, 64, 64],
    # ]

    # Txt Decoder : Reconstruction args
    decoder_dim = 512
    decoder_depth = 4
    decoder_heads = 8
    decoder_head_dim = 128
    decoder_mlp_dim = 128
    decoder_dropout = 0.1
    
    # Aud Decoder : Reconstruction args
    aud_size = [256, 64]
    aud_channels = 1
    aud_decoder_layer_info = [
        [2, 1024, 8, 2],
        [2, 512, 16, 4],
        [2, 256, 32, 8],
        [2, 128, 64, 16],
        [2, 64, 128, 32],
    ]
    
    vision_config = VisionConfig_BASE
    text_config = TextConfig_BASE
    audio_config = AudioConfig_BASE
    
    
##### LARGE Model #####
@dataclass
class VisionConfig_LARGE:
    model_link = 'openai/clip-vit-large-patch14'
    
    attention_dropout = 0.0
    dropout = 0.0
    hidden_act = "quick_gelu"
    hidden_size = 1024
    image_size = 224
    initializer_factor = 1.0
    initializer_range = 0.02
    intermediate_size = 4096
    layer_norm_eps = 1e-05
    num_attention_heads = 16
    num_hidden_layers = 24
    num_channels = 3
    patch_size = 14
    
    output_attentions = False
    output_hidden_states = False
    
@dataclass
class TextConfig_LARGE:
    model_link = 'FacebookAI/roberta-large' 
    # model_link = 'FacebookAI/xlm-roberta-large'
    
    attention_dropout = 0.1
    hidden_dropout = 0.1
    dropout = 0.0
    hidden_act = "gelu"
    bos_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    hidden_size = 1024
    initializer_range = 0.02
    intermediate_size = 4096
    layer_norm_eps = 1e-05
    max_position_embeddings = 514
    num_attention_heads = 16
    num_hidden_layers = 24
    vocab_size = 50265
    # vocab_size = 250002
    
    output_attentions = False
    output_hidden_states = False
    
# @dataclass
# class TextConfig_LARGE:
#     # model_link = 'FacebookAI/roberta-large' 
#     model_link = 'openai/clip-vit-large-patch14'
    
#     attention_dropout = 0.0
#     hidden_dropout = 0.0
#     dropout = 0.0
#     hidden_act = "gelu"
#     bos_token_id = 0
#     pad_token_id = 1
#     eos_token_id = 2
#     hidden_size = 768
#     initializer_range = 0.02
#     intermediate_size = 3072
#     layer_norm_eps = 1e-05
#     max_position_embeddings = 77
#     num_attention_heads = 12
#     num_hidden_layers = 12
#     # vocab_size = 50265
#     vocab_size = 49408
    
#     output_attentions = False
#     output_hidden_states = False
        
@dataclass
class AudioConfig_LARGE:
    model_link = 'MIT/ast-finetuned-audioset-10-10-0.4593'
    
    attention_probs_dropout_prob = 0.0
    hidden_dropout_prob = 0.0
    frequency_stride = 10
    time_stride: 10
    hidden_size = 768
    hidden_act = 'glue'
    initializer_range = 0.02
    intermediate_size = 3072
    qkv_bias = True
    layer_norm_eps: 1e-12
    max_length = 1024
    num_attention_heads = 12
    num_hidden_layers = 12
    num_mel_bins = 128
    patch_size = 16
    
    output_attentions = False
    output_hidden_states = False

@dataclass
class CLIPConfig_LARGE:
    num_of_modality = 3
    
    is_PT = True
    return_logits = False
    return_lhs = False
    
    logit_scale_init_value = 2.6592
    projection_dim = 1024
    return_dict = True
    
    vision_config = VisionConfig_LARGE
    text_config = TextConfig_LARGE
    audio_config = AudioConfig_LARGE
    
@dataclass
class ReconstructionConfig_LARGE:
    IS_BASE = False
    num_of_modality = 3
    
    is_PT = False
    return_logits = False
    return_att = False
    return_lhs = True
    return_dict = True
    
    # in Multi_Modal_Encoder
    # vision_length -> img_tokens
    # text_length -> txt_tokens
    # audio_length -> aud_tokens
    projection_dim = 1024

    vision_length = 257
    text_length   = 32 # seq_max_length in main_*.py line 51, 83
    audio_length  = 1214

    img_tokens = 192
    txt_tokens = 192
    aud_tokens = 192
    
    # MM Encoder args
    encoder_dim = 1024
    encoder_bottleneck_dim = 128
    encoder_dropout = 0.1

    # Img Decoder : Reconstruction args
    img_size = 128
    img_channels = 3
    img_decoder_layer_info = [
        [2, 2048, 2, 2],
        [2, 1024, 4, 4],
        [2, 512, 8, 8],
        [2, 256, 16, 16],
        [2, 128, 32, 32],
        [2, 64, 64, 64],
    ]

    # Txt Decoder : Reconstruction args
    decoder_dim = 768
    decoder_depth = 6
    decoder_heads = 12
    decoder_head_dim = 192
    decoder_mlp_dim = 192
    decoder_dropout = 0.1
    
    # Aud Decoder : Reconstruction args
    aud_size = [256, 64]
    aud_channels = 1
    aud_decoder_layer_info = [
        [2, 2048, 4, 1],
        [2, 1024, 8, 2],
        [2, 512, 16, 4],
        [2, 256, 32, 8],
        [2, 128, 64, 16],
        [2, 64, 128, 32],
    ]
    
    vision_config = VisionConfig_LARGE
    text_config = TextConfig_LARGE
    audio_config = AudioConfig_LARGE