# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan flf2v A14B ------------------------#

flf2v_A14B = EasyDict(__name__='Config: Wan flf2v A14B')
flf2v_A14B.update(wan_shared_cfg)

flf2v_A14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
flf2v_A14B.t5_tokenizer = 'google/umt5-xxl'

# vae
flf2v_A14B.vae_checkpoint = 'Wan2.1_VAE.pth'
flf2v_A14B.vae_stride = (4, 8, 8)

# transformer
flf2v_A14B.patch_size = (1, 2, 2)
flf2v_A14B.dim = 5120
flf2v_A14B.ffn_dim = 13824
flf2v_A14B.freq_dim = 256
flf2v_A14B.num_heads = 40
flf2v_A14B.num_layers = 40
flf2v_A14B.window_size = (-1, -1)
flf2v_A14B.qk_norm = True
flf2v_A14B.cross_attn_norm = True
flf2v_A14B.eps = 1e-6
flf2v_A14B.low_noise_checkpoint = 'low_noise_model'
flf2v_A14B.high_noise_checkpoint = 'high_noise_model'

# inference
flf2v_A14B.sample_shift = 5.0
flf2v_A14B.sample_steps = 50
flf2v_A14B.boundary = 0.900
flf2v_A14B.sample_guide_scale = (3.5, 3.5)  # low noise, high noise
