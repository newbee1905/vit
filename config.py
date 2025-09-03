import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.norm import DyT, RMSNorm

# from dataclasses import dataclass

class ViTConfig:
	d_model=512
	d_ff=2048

	n_head=16
	n_block=4
	# n_patch=16
	n_patch=7
	out_d = 200 # TinyImageNet

	norm=RMSNorm
	activation="swiglu"

	dropout=0.1

	use_layer_scale=True
	use_qk_norm=False

	# chw=(3, 224, 224)
	chw=(1, 28, 28)

	kernel_size=16
	stride=16

class CoAtNetConfig:
	dims=[64, 96, 192, 384, 768]
	depths=[2, 2, 3, 5, 2]
	num_heads=32

	# chw=(3, 224, 224)
	chw=(1, 28, 28)

	activation="silu"
	norm=RMSNorm

	dropout=0.1

	out_d = 200 # TinyImageNet

class CvTConfig:
	# Based on CvT-13, adjusted for TinyImageNet
	chw = (3, 64, 64)
	out_d = 200 # TinyImageNet
	
	# Stage 1
	s1_emb_kernel = 7
	s1_emb_stride = 4
	s1_emb_pad = 2
	s1_emb_dim = 64
	s1_depth = 1
	s1_heads = 1
	s1_mlp_ratio = 4.0
	s1_qkv_kernel = 3
	s1_qkv_stride = 1
	s1_qkv_pad = 1

	# Stage 2
	s2_emb_kernel = 3
	s2_emb_stride = 2
	s2_emb_pad = 1
	s2_emb_dim = 192
	s2_depth = 2
	s2_heads = 3
	s2_mlp_ratio = 4.0
	s2_qkv_kernel = 3
	s2_qkv_stride = 2
	s2_qkv_pad = 1

	# Stage 3
	s3_emb_kernel = 3
	s3_emb_stride = 2
	s3_emb_pad = 1
	s3_emb_dim = 384
	s3_depth = 10
	s3_heads = 6
	s3_mlp_ratio = 4.0
	s3_qkv_kernel = 3
	s3_qkv_stride = 2
	s3_qkv_pad = 1

	dropout=0.1
	norm=RMSNorm
	activation="gelu"
	use_layer_scale=False
