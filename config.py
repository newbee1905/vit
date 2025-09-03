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
	out_d=10

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
