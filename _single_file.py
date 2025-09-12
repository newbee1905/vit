# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import argparse
import random
import zipfile
import requests
import math
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler

from torchvision.datasets.utils import download_url
from torchvision.transforms import v2
from torchvision.models import resnet18, ResNet18_Weights

from tqdm import tqdm
from einops import rearrange

# ==============================================================================
# LAYERS
# ==============================================================================

# layers/norm.py
class RMSNorm(nn.Module):
	def __init__(
		self,
		dim: int,
		eps: float = 1e-6,
		add_unit_offset: bool = True,
	):
		super().__init__()
		self.eps = eps
		self.add_unit_offset = add_unit_offset
		self.weight = nn.Parameter(torch.zeros(dim))

	def _norm(self, x):
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

	def forward(self, x):
		output = self._norm(x.float())
		if self.add_unit_offset:
			output = output * (1 + self.weight.float())
		else:
			output = output * self.weight.float()
		return output.type_as(x)

class DyT(nn.Module):
	def __init__(self, C, init_alpha=0.5):
		super().__init__()
		self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
		self.gamma = nn.Parameter(torch.ones(C))
		self.beta = nn.Parameter(torch.zeros(C))

	def forward(self, x):
		x = torch.tanh(self.alpha * x)
		return self.gamma * x + self.beta

class QKNorm(nn.Module):
	def __init__(self, d_head: int, eps: float = 1e-6):
		super().__init__()
		self.eps = eps
		self.gain = nn.Parameter(torch.ones(1, 1, 1, d_head))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		norm_x = F.normalize(x, p=2, dim=-1, eps=self.eps)
		return norm_x * self.gain

# layers/conv.py
class MBConv(nn.Module):
	def __init__(self, in_chans, out_chans, stride=1, expand_ratio=4, activation="gelu"):
		super().__init__()
		self.stride = stride
		hidden_dim = in_chans * expand_ratio
		self.activation = activation.lower()
		if self.activation == "gelu":
			self.act_fn = nn.GELU
		elif self.activation == "silu":
			self.act_fn = nn.SiLU
		else:
			self.act_fn = nn.ReLU
		self.use_res_connect = self.stride == 1 and in_chans == out_chans
		layers = []
		layers.append(nn.Conv2d(in_chans, hidden_dim, 1, 1, 0, bias=False))
		layers.append(nn.BatchNorm2d(hidden_dim))
		layers.append(self.act_fn())
		layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
		layers.append(nn.BatchNorm2d(hidden_dim))
		layers.append(self.act_fn())
		layers.append(nn.Conv2d(hidden_dim, out_chans, 1, 1, 0, bias=False))
		layers.append(nn.BatchNorm2d(out_chans))
		self.conv = nn.Sequential(*layers)

	def forward(self, x):
		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)

# layers/embeddings.py
def patchify(images, n_patch):
	_, _, h, w = images.shape
	assert h == w, "Patchify method is implemented for square images only"
	return rearrange(
		images,
		'b c (p1 h_patch) (p2 w_patch) -> b (p1 p2) (c h_patch w_patch)',
		p1=n_patch, p2=n_patch, h_patch=h // n_patch, w_patch=w // n_patch,
	)

class ConvEmbedding(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
		self.norm = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		x = self.conv(x)
		x = self.norm(x)
		x = rearrange(x, 'b c h w -> b (h w) c')
		return x

# layers/attention.py
class MHA(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.d_model = config.d_model
		self.n_head = config.n_head
		self.d_head = self.d_model // self.n_head
		self.qkv_proj = nn.Linear(self.d_model, self.d_model * 3, bias=False)
		self.out_proj = nn.Linear(self.d_model, self.d_model)
		self.dropout = nn.Dropout(config.dropout)
		self.use_qk_norm = config.use_qk_norm
		if self.use_qk_norm:
			self.q_norm = QKNorm(self.d_head)
			self.k_norm = QKNorm(self.d_head)

	def forward(self, query):
		qkv = self.qkv_proj(query)
		q, k, v = qkv.chunk(3, dim=-1)
		q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_head)
		k = rearrange(k, 'b s (h d) -> b h s d', h=self.n_head)
		v = rearrange(v, 'b s (h d) -> b h s d', h=self.n_head)
		if self.use_qk_norm:
			q = self.q_norm(q)
			k = self.k_norm(k)
		attn_output = F.scaled_dot_product_attention(
			q, k, v,
			dropout_p=self.dropout.p if self.training else 0.0,
			is_causal=False,
		)
		attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
		output = self.out_proj(attn_output)
		return output

class ConvAttention(nn.Module):
	def __init__(self, dim, n_head, kernel_size=3, qkv_stride=1, padding=1, dropout=0.1):
		super().__init__()
		self.n_head = n_head
		self.dim = dim
		self.d_head = dim // n_head
		self.q_proj = nn.Linear(dim, dim)
		self.k_proj = nn.Linear(dim, dim)
		self.v_proj = nn.Linear(dim, dim)
		self.conv_proj_k = nn.Conv2d(dim, dim, kernel_size, stride=qkv_stride, padding=padding, groups=dim, bias=False)
		self.conv_proj_v = nn.Conv2d(dim, dim, kernel_size, stride=qkv_stride, padding=padding, groups=dim, bias=False)
		self.out_proj = nn.Linear(dim, dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, h, w):
		q = self.q_proj(x)
		k = self.k_proj(x)
		v = self.v_proj(x)
		k = rearrange(k, 'b (h w) c -> b c h w', h=h, w=w)
		v = rearrange(v, 'b (h w) c -> b c h w', h=h, w=w)
		k = self.conv_proj_k(k)
		v = self.conv_proj_v(v)
		k = rearrange(k, 'b c h w -> b (h w) c')
		v = rearrange(v, 'b c h w -> b (h w) c')
		q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_head)
		k = rearrange(k, 'b n (h d) -> b h n d', h=self.n_head)
		v = rearrange(v, 'b n (h d) -> b h n d', h=self.n_head)
		attn_output = F.scaled_dot_product_attention(
			q, k, v,
			dropout_p=self.dropout.p if self.training else 0.0,
			is_causal=False,
		)
		attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
		output = self.out_proj(attn_output)
		return output

# layers/transformers.py
class FeedForward(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.activation = config.activation.lower()
		supported_activations = (
			'swiglu', 'silu', 'geglu', 'gelu', 'reglu', 'relu', 'reglu6', 'relu6'
		)
		if self.activation not in supported_activations:
			raise ValueError(f"Unknown activation type: {self.activation}")
		self.uses_gate = self.activation in ('swiglu', 'geglu', 'reglu', 'reglu6')
		if 'silu' in self.activation or 'swiglu' in self.activation:
			self.act_fn = F.silu
		elif 'gelu' in self.activation or 'geglu' in self.activation:
			self.act_fn = F.gelu
		elif 'relu6' in self.activation or 'reglu6' in self.activation:
			self.act_fn = F.relu6
		elif 'relu' in self.activation or 'reglu' in self.activation:
			self.act_fn = F.relu
		if self.activation in ('swiglu', 'geglu'):
			self.fc_in = nn.Linear(config.d_model, config.d_ff * 2)
		else:
			self.fc_in = nn.Linear(config.d_model, config.d_ff)
		self.fc_out = nn.Linear(config.d_ff, config.d_model)
		self.dropout1 = nn.Dropout(config.dropout)
		self.dropout2 = nn.Dropout(config.dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x_proj = self.fc_in(x)
		if self.activation in ('swiglu', 'geglu'):
			gate, x_proj = x_proj.chunk(2, dim=-1)
			x_proj = gate * self.act_fn(x_proj)
		else:
			x_proj = self.act_fn(x_proj)
		x = self.fc_out(self.dropout1(x_proj))
		x = self.dropout2(x)
		return x

class I2T(nn.Module):
	""" Image to Tokens module from CeiT """
	def __init__(self, config):
		super().__init__()
		self.conv = nn.Conv2d(config.chw[0], config.d_model, kernel_size=config.kernel_size, stride=config.stride)
		self.bn = nn.BatchNorm2d(config.d_model)

	def forward(self, x):
		x = self.bn(self.conv(x))
		return x.flatten(2).transpose(1, 2)

class LeFF(nn.Module):
	""" Locally-enhanced Feed-Forward from CeiT """
	def __init__(self, config, kernel_size=3):
		super().__init__()
		self.d_model = config.d_model
		self.d_ff = config.d_ff
		self.activation = config.activation
		self.kernel_size = kernel_size
		if self.activation in ('swiglu', 'geglu'):
			self.linear1 = nn.Linear(config.d_model, config.d_ff * 2)
		else:
			self.linear1 = nn.Linear(config.d_model, config.d_ff)
		self.act_fn = F.silu if self.activation in ('swiglu', 'silu') else F.gelu
		self.conv = nn.Conv2d(self.d_ff, self.d_ff, self.kernel_size, stride=1, padding=self.kernel_size // 2, groups=self.d_ff)
		self.bn = nn.BatchNorm2d(self.d_ff)
		self.linear2 = nn.Linear(self.d_ff, self.d_model)
		self.drop = nn.Dropout(config.dropout)

	def forward(self, x, H, W):
		B, N, C = x.shape
		x = self.linear1(x)
		if self.activation in ('swiglu', 'geglu'):
			gate, x = x.chunk(2, dim=-1)
			x = gate * self.act_fn(x)
		else:
			x = self.act_fn(x)
		x_conv = x.transpose(1, 2).reshape(B, self.d_ff, H, W)
		x_conv = self.bn(self.conv(x_conv))
		x_conv = x_conv.reshape(B, self.d_ff, N).transpose(1, 2)
		x = x + x_conv
		x = self.linear2(x)
		x = self.drop(x)
		return x

# layers/vit.py
class ViTBlock(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.d_model = config.d_model
		self.n_head = config.n_head
		self.attn_norm = config.norm(self.d_model)
		self.attn = MHA(config)
		self.attn_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.attn_dropout = nn.Dropout(config.dropout)
		self.ff_norm = config.norm(self.d_model)
		self.ff = FeedForward(config)
		self.ff_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.ff_dropout = nn.Dropout(config.dropout)

	def forward(self, x):
		norm_x = self.attn_norm(x)
		attn_out = self.attn(norm_x)
		attn_out = self.attn_dropout(attn_out)
		if self.attn_layer_scale is not None:
			x = x + self.attn_layer_scale * attn_out
		else:
			x = x + attn_out
		norm_x = self.ff_norm(x)
		ff_out = self.ff(norm_x)
		ff_out = self.ff_dropout(ff_out)
		if self.ff_layer_scale is not None:
			x = x + self.ff_layer_scale * ff_out
		else:
			x = x + ff_out
		return x

# layers/cvt.py
class CvTBlock(nn.Module):
	def __init__(self, config, kernel_size, qkv_stride, padding):
		super().__init__()
		self.d_model = config.d_model
		self.attn_norm = config.norm(self.d_model)
		self.attn = ConvAttention(
			dim=self.d_model,
			n_head=config.n_head,
			kernel_size=kernel_size,
			qkv_stride=qkv_stride,
			padding=padding,
			dropout=config.dropout
		)
		self.attn_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.attn_dropout = nn.Dropout(config.dropout)
		self.ff_norm = config.norm(self.d_model)
		self.ff = FeedForward(config)
		self.ff_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.ff_dropout = nn.Dropout(config.dropout)

	def forward(self, x, h, w):
		norm_x = self.attn_norm(x)
		attn_out = self.attn(norm_x, h, w)
		attn_out = self.attn_dropout(attn_out)
		if self.attn_layer_scale is not None:
			x = x + self.attn_layer_scale * attn_out
		else:
			x = x + attn_out
		norm_x = self.ff_norm(x)
		ff_out = self.ff(norm_x)
		ff_out = self.ff_dropout(ff_out)
		if self.ff_layer_scale is not None:
			x = x + self.ff_layer_scale * ff_out
		else:
			x = x + ff_out
		return x, h, w


# ==============================================================================
# MODELS
# ==============================================================================

# models/utils.py
def get_positional_embeddings(seq_len, d_model, theta=10000.0):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(-1, 1)
	dim = torch.arange(d_model, dtype=torch.float, device=device).reshape(1, -1)
	div_term = theta ** (torch.div(dim, 2, rounding_mode='floor') * 2 / d_model)
	embeddings = torch.zeros(seq_len, d_model, device=device)
	embeddings[:, 0::2] = torch.sin(pos / div_term[0, 0::2])
	embeddings[:, 1::2] = torch.cos(pos / div_term[0, 1::2])
	return embeddings

# models/vit.py
class ViT(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.chw = config.chw
		self.d_model = config.d_model
		self.n_patch = config.n_patch
		self.n_block = config.n_block
		assert self.chw[1] % self.n_patch == 0, "Input shape not entirely divisible by number of patch"
		assert self.chw[2] % self.n_patch == 0, "Input shape not entirely divisible by number of patch"
		self.patch_size = (self.chw[1] // self.n_patch, self.chw[2] // self.n_patch)
		self.input_d = self.chw[0] * self.patch_size[0] * self.patch_size[1]
		self.linear_mapper = nn.Linear(self.input_d, self.d_model)
		self.class_token = nn.Parameter(torch.rand(1, self.d_model))
		self.pos_embed = nn.Parameter(torch.randn(1, self.n_patch ** 2 + 1, self.d_model))
		self.blocks = nn.ModuleList([ViTBlock(config) for _ in range(self.n_block)])
		self.mlp = nn.Linear(self.d_model, config.out_d)

	def forward(self, images):
		bsz = images.size(0)
		patch = patchify(images, self.n_patch)
		tokens = self.linear_mapper(patch)
		tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
		pos_embed = self.pos_embed.repeat(bsz, 1, 1)
		out = tokens + pos_embed
		for block in self.blocks:
			out = block(out)
		out = out[:, 0]
		return self.mlp(out)

# models/coatnet.py
class CoAtNet(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.activation = config.activation.lower()
		if self.activation == "gelu":
			self.act_fn = nn.GELU
		else:
			self.act_fn = nn.SiLU
		self.s0 = nn.Sequential(
			nn.Conv2d(config.chw[0], config.dims[0], kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(config.dims[0]),
			self.act_fn(),
		)
		self.s1 = self._make_conv_stage(config.dims[0], config.dims[1], config.depths[0], stride=2, activation=config.activation)
		self.s2 = self._make_conv_stage(config.dims[1], config.dims[2], config.depths[1], stride=2, activation=config.activation)
		self.s3_pe = nn.Conv2d(config.dims[2], config.dims[3], 1)
		self.s3 = self._make_transformer_stage(config.dims[3], config.depths[2], config.num_heads, config.norm, config.activation)
		self.s4_pe = nn.Conv2d(config.dims[3], config.dims[4], 2, 2)
		self.s4 = self._make_transformer_stage(config.dims[4], config.depths[3], config.num_heads, config.norm, config.activation)
		self.head = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(config.dims[-1], config.out_d)
		)

	def _make_conv_stage(self, in_chans, out_chans, depth, stride, activation):
		layers = [MBConv(in_chans, out_chans, stride=stride, activation=activation)]
		for _ in range(depth - 1):
			layers.append(MBConv(out_chans, out_chans, stride=1, activation=activation))
		return nn.Sequential(*layers)

	def _make_transformer_stage(self, dim, depth, num_heads, norm=RMSNorm, activation="silu"):
		class MHAConfig: pass
		config = MHAConfig()
		config.d_model, config.n_head, config.dropout, config.use_qk_norm, config.use_layer_scale = dim, num_heads, 0.1, False, False
		config.norm, config.activation, config.d_ff = norm, activation, dim * 4
		blocks = []
		for _ in range(depth):
			blocks.append(ViTBlock(config))
		return nn.ModuleList(blocks)

	def forward(self, x):
		B = x.shape[0]
		x = self.s0(x)
		x = self.s1(x)
		x = self.s2(x)
		x = self.s3_pe(x)
		_, C, H, W = x.shape
		x = rearrange(x, 'b c h w -> b (h w) c')
		for blk in self.s3:
			x = blk(x)
		x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
		x = self.s4_pe(x)
		_, C, H, W = x.shape
		x = rearrange(x, 'b c h w -> b (h w) c')
		for blk in self.s4:
			x = blk(x)
		x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
		x = self.head(x)
		return x

# models/cvt.py
class CvT(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.stage1_embedding = ConvEmbedding(
			in_channels=config.chw[0], out_channels=config.s1_emb_dim, kernel_size=config.s1_emb_kernel,
			stride=config.s1_emb_stride, padding=config.s1_emb_pad,
		)
		self.stage1_blocks = nn.ModuleList([
			CvTBlock(
				config=self.get_stage_config(config.s1_emb_dim, config.s1_heads, config.s1_mlp_ratio),
				kernel_size=config.s1_qkv_kernel, qkv_stride=config.s1_qkv_stride, padding=config.s1_qkv_pad,
			) for _ in range(config.s1_depth)
		])
		self.stage2_embedding = ConvEmbedding(
			in_channels=config.s1_emb_dim, out_channels=config.s2_emb_dim, kernel_size=config.s2_emb_kernel,
			stride=config.s2_emb_stride, padding=config.s2_emb_pad,
		)
		self.stage2_blocks = nn.ModuleList([
			CvTBlock(
				config=self.get_stage_config(config.s2_emb_dim, config.s2_heads, config.s2_mlp_ratio),
				kernel_size=config.s2_qkv_kernel, qkv_stride=config.s2_qkv_stride, padding=config.s2_qkv_pad,
			) for _ in range(config.s2_depth)
		])
		self.stage3_embedding = ConvEmbedding(
			in_channels=config.s2_emb_dim, out_channels=config.s3_emb_dim, kernel_size=config.s3_emb_kernel,
			stride=config.s3_emb_stride, padding=config.s3_emb_pad,
		)
		self.stage3_blocks = nn.ModuleList([
			CvTBlock(
				config=self.get_stage_config(config.s3_emb_dim, config.s3_heads, config.s3_mlp_ratio),
				kernel_size=config.s3_qkv_kernel, qkv_stride=config.s3_qkv_stride, padding=config.s3_qkv_pad,
			) for _ in range(config.s3_depth)
		])
		self.norm = config.norm(config.s3_emb_dim)
		self.head = nn.Linear(config.s3_emb_dim, config.out_d)

	def get_stage_config(self, d_model, n_head, mlp_ratio):
		class StageConfig: pass
		s_config = StageConfig()
		s_config.d_model = d_model
		s_config.d_ff = int(d_model * mlp_ratio)
		s_config.n_head = n_head
		s_config.norm = self.config.norm
		s_config.dropout = self.config.dropout
		s_config.use_layer_scale = self.config.use_layer_scale
		s_config.activation = self.config.activation
		return s_config

	def forward(self, x):
		h, w = self.config.chw[1], self.config.chw[2]
		x = self.stage1_embedding(x)
		h, w = h // self.config.s1_emb_stride, w // self.config.s1_emb_stride
		for blk in self.stage1_blocks:
			x, h, w = blk(x, h, w)
		x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
		x = self.stage2_embedding(x)
		h, w = h // self.config.s2_emb_stride, w // self.config.s2_emb_stride
		for blk in self.stage2_blocks:
			x, h, w = blk(x, h, w)
		x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
		x = self.stage3_embedding(x)
		h, w = h // self.config.s3_emb_stride, w // self.config.s3_emb_stride
		for blk in self.stage3_blocks:
			x, h, w = blk(x, h, w)
		x = x.mean(dim=1)
		x = self.norm(x)
		return self.head(x)


# ==============================================================================
# CONFIGS
# ==============================================================================

# config.py
class ViTConfig:
	d_model=192
	d_ff=768

	n_head=16
	n_block=4
	# n_patch=16
	n_patch=4
	out_d = 100

	attention=QuadrangleAttention
	window_size=[2, 2]
	norm=RMSNorm
	activation="swiglu"

	dropout=0.3

	use_layer_scale=True
	use_qk_norm=False

	# chw=(3, 224, 224)
	chw=(3, 32, 32)

	kernel_size=16
	stride=16

class CoAtNetConfig:
	dims=[64, 96, 192, 384, 768]
	depths=[2, 2, 3, 5, 2]
	num_heads=32
	chw=(3, 64, 64)
	activation="silu"
	norm=RMSNorm
	dropout=0.1
	out_d = 200

class CvTConfig:
	chw = (3, 64, 64)
	out_d = 200
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

# ==============================================================================
# UTILS
# ==============================================================================

# utils.py
def get_config(model_name: str, args=None):
	if model_name.lower() == "cvt":
		return CvTConfig()
	elif model_name.lower() == "vit":
		return ViTConfig()
	elif model_name.lower() == "coatnet":
		return CoAtNetConfig()
	elif model_name.lower() == "resnet18":
		return None
	else:
		raise ValueError(f"Unknown model {model_name}")

def get_model(model_name: str, config=None, num_classes=200):
	model_name = model_name.lower()
	if model_name == "cvt":
		return CvT(config)
	elif model_name == "vit":
		return ViT(config)
	elif model_name == "coatnet":
		return CoAtNet(config)
	elif model_name == "resnet18":
		model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
		model.fc = nn.Linear(model.fc.in_features, num_classes)
		return model
	else:
		raise ValueError(f"Unknown model {model_name}")

def get_param_groups(model, weight_decay):
	decay, no_decay = [], []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if (name.endswith("bias") or "norm" in name.lower() or "layerscale" in name.lower()):
			no_decay.append(param)
		else:
			decay.append(param)
	return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

def set_seed(seed=0):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def load_teacher_model(path, model_name, num_classes, device):
	checkpoint = torch.load(path, map_location=device)
	config = checkpoint.get('config')
	model = get_model(model_name, config, num_classes=num_classes)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.to(device)
	model.eval()
	return model

# ==============================================================================
# SCHEDULERS
# ==============================================================================

# schedulers.py
class WarmupScheduler(_LRScheduler):
	def __init__(self, optimizer, warmup_steps, after_scheduler=None, last_epoch=-1):
		self.warmup_steps = warmup_steps
		self.after_scheduler = after_scheduler
		self.finished_warmup = False
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		if self.last_epoch < self.warmup_steps:
			return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
		if self.after_scheduler:
			if not self.finished_warmup:
				self.after_scheduler.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
				self.finished_warmup = True
			return self.after_scheduler.get_last_lr()
		return [group["lr"] for group in self.optimizer.param_groups]

	def step(self, epoch=None, metrics=None):
		if self.last_epoch < self.warmup_steps:
			return super(WarmupScheduler, self).step(epoch)
		if self.after_scheduler:
			if isinstance(self.after_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
				self.after_scheduler.step(metrics)
			else:
				if epoch is None:
					self.after_scheduler.step(None)
				else:
					self.after_scheduler.step(epoch - self.warmup_steps)


# ==============================================================================
# DATASET
# ==============================================================================

# dataset.py
class TinyImageNet(Dataset):
	url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
	filename = 'tiny-imagenet-200.zip'
	folder_name = 'tiny-imagenet-200'

	def __init__(self, root, split='train', transform=None, download=False):
		self.root = os.path.expanduser(root)
		self.split = split
		self.transform = transform
		if download:
			self.download()
		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')
		self.data = []
		self.targets = []
		self.class_to_idx = {}
		with open(os.path.join(self.root, self.folder_name, 'wnids.txt'), 'r') as f:
			for i, line in enumerate(f):
				self.class_to_idx[line.strip()] = i
		self._load_data()

	def _load_data(self):
		if self.split in ['train', 'val']:
			self._load_from_folder(self.split)
		else:
			raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'val'")

	def _load_from_folder(self, folder):
		if folder == 'train':
			split_dir = os.path.join(self.root, self.folder_name, folder)
			for class_folder in os.listdir(split_dir):
				class_idx = self.class_to_idx[class_folder]
				class_dir = os.path.join(split_dir, class_folder, 'images')
				for img_name in os.listdir(class_dir):
					img_path = os.path.join(class_dir, img_name)
					self.data.append(img_path)
					self.targets.append(class_idx)
		elif folder == 'val':
			split_dir = os.path.join(self.root, self.folder_name, folder)
			with open(os.path.join(split_dir, 'val_annotations.txt'), 'r') as f:
				for line in f:
					parts = line.strip().split('\t')
					img_name, class_id = parts[0], parts[1]
					img_path = os.path.join(split_dir, 'images', img_name)
					self.data.append(img_path)
					self.targets.append(self.class_to_idx[class_id])

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path = self.data[idx]
		target = self.targets[idx]
		with open(img_path, 'rb') as f:
			img = Image.open(f).convert('RGB')
		if self.transform:
			img = self.transform(img)
		return img, target

	def _check_integrity(self):
		return os.path.isdir(os.path.join(self.root, self.folder_name))

	def download(self):
		if self._check_integrity():
			print('Files already downloaded and verified')
			return
		download_url(self.url, self.root, self.filename)
		with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zip_ref:
			zip_ref.extractall(self.root)


# ==============================================================================
# TRAINERS
# ==============================================================================

# train.py
class Trainer:
	def __init__(self, model, optimizer, criterion, device, scheduler=None, scheduler_type=None, patience=10, min_delta=1e-4, writer=None):
		self.model = model
		self.optimizer = optimizer
		self.criterion = criterion
		self.device = device
		self.writer = writer
		self.scheduler = scheduler
		self.scheduler_type = scheduler_type
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.early_stop = False
		self.start_epoch = 0
		self.best_val_loss = float('inf')

	def run_one_epoch(self, dataloader, state='train'):
		is_training = (state == 'train')
		self.model.train(is_training)
		total_loss = 0.0
		correct = 0
		total = 0
		with torch.set_grad_enabled(is_training):
			for batch in dataloader:
				x, y = batch
				x, y = x.to(self.device), y.to(self.device)
				if is_training:
					self.optimizer.zero_grad()
				y_hat = self.model(x)
				loss = self.compute_loss(y_hat, y, x)
				if is_training:
					loss.backward()
					self.optimizer.step()
				total_loss += loss.detach().cpu().item()
				pred_for_acc = y_hat[0] if isinstance(y_hat, tuple) else y_hat
				correct += torch.sum(torch.argmax(pred_for_acc, dim=1) == y).detach().cpu().item()
				total += len(x)
		return total_loss / len(dataloader), correct / total

	def compute_loss(self, y_hat, y, x=None):
		return self.criterion(y_hat, y)

	def train(self, n_epochs, train_dl, val_dl, save_path=None, config=None, args=None):
		with tqdm(range(self.start_epoch, n_epochs), desc="Training Progress") as pbar:
			for epoch in pbar:
				train_loss, train_acc = self.run_one_epoch(train_dl, state='train')
				val_loss, val_acc = self.run_one_epoch(val_dl, state='eval')

				is_best = val_loss < self.best_val_loss

				if self.writer:
					self.writer.add_scalar('Loss/train', train_loss, epoch)
					self.writer.add_scalar('Accuracy/train', train_acc, epoch)
					self.writer.add_scalar('Loss/val', val_loss, epoch)
					self.writer.add_scalar('Accuracy/val', val_acc, epoch)
					self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

				if self.scheduler:
					if self.scheduler_type == "reduceonplateau": self.scheduler.step(val_loss)
					else: self.scheduler.step()

				if save_path and is_best:
					self.save_checkpoint(save_path, epoch=epoch, config=config, args=args, best=True)

				if val_loss < self.best_val_loss - self.min_delta:
					self.best_val_loss = val_loss
					self.counter = 0
				else:
					self.counter += 1

				if self.counter >= self.patience:
					print(f"\nEarly stopping triggered at epoch {epoch+1}")
					self.early_stop = True
					break

				pbar.set_postfix({
					'epoch': epoch + 1, 'train_loss': f'{train_loss:.4f}', 'train_acc': f'{train_acc:.3f}',
					'val_loss': f'{val_loss:.4f}', 'val_acc': f'{val_acc:.3f}', 'best_val_loss': f'{self.best_val_loss:.4f}'
				})


	def save_checkpoint(self, path, epoch, config=None, args=None, best=False):
		checkpoint = {
			'epoch': epoch + 1, 'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(), 'best_val_loss': self.best_val_loss,
		}
		if self.scheduler: checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
		if config: checkpoint['config'] = config
		if args: checkpoint['args'] = args
		
		if best:
			best_path = path.replace(".pt", "_best.pt")
			torch.save(checkpoint, best_path)
			print(f"Best model updated and saved to {best_path}")
		else:
			torch.save(checkpoint, path)

	def load_checkpoint(self, checkpoint):
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		if self.scheduler and 'scheduler_state_dict' in checkpoint:
			self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		self.start_epoch = checkpoint.get('epoch', 0)
		self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
		print(f"Resumed from epoch {self.start_epoch}, best val loss {self.best_val_loss:.4f}")

# deit_trainer.py
class DeiTTrainer(Trainer):
	def __init__(self, model, teacher_model, optimizer, criterion, device, scheduler=None, scheduler_type=None, writer=None, alpha=0.5, tau=1.0):
		super().__init__(
			model, optimizer, criterion, device,
			scheduler=scheduler, scheduler_type=scheduler_type, writer=writer
		)
		self.teacher_model = teacher_model
		self.teacher_model.eval()
		for param in self.teacher_model.parameters():
			param.requires_grad = False
		self.alpha = alpha
		self.tau = tau

	def compute_loss(self, y_hat, y, x=None):
		student_pred = y_hat[0] if isinstance(y_hat, tuple) else y_hat
		student_distill = y_hat[1] if isinstance(y_hat, tuple) else y_hat
		base_loss = self.criterion(student_pred, y)
		if x is None:
			return base_loss
		with torch.no_grad():
			teacher_pred = self.teacher_model(x)
			if isinstance(teacher_pred, tuple):
				teacher_pred = teacher_pred[0]
		distill_loss = F.kl_div(
			F.log_softmax(student_distill / self.tau, dim=1),
			F.softmax(teacher_pred / self.tau, dim=1),
			reduction='batchmean'
		) * (self.tau * self.tau)
		return (1 - self.alpha) * base_loss + self.alpha * distill_loss

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
def main():
	class SimpleArgs: pass
	args = SimpleArgs()
	args.data_root = "./"
	args.download = True
	args.save_path = "checkpoints"
	args.device = "auto"
	args.num_classes = 200
	args.optimizer = "adamw"
	args.lr = 1e-3
	args.weight_decay = 1e-4
	args.scheduler = "cosineannealing"
	args.batch_size = 128
	args.num_workers = 2
	args.seed = 42
	args.label_smoothing = 0.1
	args.warmup_steps = 0

	set_seed(args.seed)
	os.makedirs(args.save_path, exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"\nUsing device: {device}")

	train_transform = v2.Compose([
		v2.ToImage(), v2.TrivialAugmentWide(), v2.RandomResizedCrop(64, scale=(0.7, 1.0)),
		v2.RandomHorizontalFlip(), v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
		v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		v2.RandomErasing(p=0.25, scale=(0.02, 0.2)),
	])
	test_transform = v2.Compose([
		v2.ToImage(), v2.Resize(72), v2.CenterCrop(64), v2.ToDtype(torch.float32, scale=True),
		v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	train_ds = TinyImageNet(root=args.data_root, split='train', download=True, transform=train_transform)
	val_ds = TinyImageNet(root=args.data_root, split='val', transform=test_transform)
	train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
	val_dl = DataLoader(val_ds, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

	# --- 1. Train Teacher Model (ResNet18) ---
	print("\n" + "="*60 + "\nPHASE 1: TRAINING TEACHER MODEL (ResNet18)\n" + "="*60)
	teacher_args = args
	teacher_args.transfer_epochs = 10
	teacher_args.finetune_epochs = 40
	teacher_args.model = "resnet18"
	teacher_model = get_model(teacher_args.model, num_classes=teacher_args.num_classes).to(device)

	print("\n--- Teacher Training: Transfer Learning ---")
	for param in teacher_model.parameters(): param.requires_grad = False
	for param in teacher_model.fc.parameters(): param.requires_grad = True
	nn.init.xavier_uniform_(teacher_model.fc.weight); nn.init.zeros_(teacher_model.fc.bias)
	optimizer = AdamW(teacher_model.fc.parameters(), lr=teacher_args.lr)
	criterion = nn.CrossEntropyLoss(label_smoothing=teacher_args.label_smoothing)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=teacher_args.transfer_epochs)
	trainer = Trainer(teacher_model, optimizer, criterion, device, scheduler=scheduler, scheduler_type="cosineannealing", writer=SummaryWriter(f"runs/{teacher_args.model}"))
	if teacher_args.transfer_epochs > 0:
		trainer.train(teacher_args.transfer_epochs, train_dl, val_dl, save_path=f"{teacher_args.save_path}/resnet18_teacher.pt", args=vars(teacher_args))

	if teacher_args.finetune_epochs > 0:
		print("\n--- Teacher Training: Fine-tuning ---")
		for param in teacher_model.parameters(): param.requires_grad = True
		param_groups = get_param_groups(teacher_model, teacher_args.weight_decay)
		optimizer = AdamW(param_groups, lr=teacher_args.lr / 10)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=teacher_args.finetune_epochs)
		trainer.optimizer, trainer.scheduler = optimizer, scheduler
		trainer.start_epoch = teacher_args.transfer_epochs
		total_epochs = teacher_args.transfer_epochs + teacher_args.finetune_epochs
		trainer.train(total_epochs, train_dl, val_dl, save_path=f"{teacher_args.save_path}/resnet18_teacher.pt", args=vars(teacher_args))
	
	teacher_path = f"{teacher_args.save_path}/resnet18_teacher_best.pt"
	print(f"Teacher model training complete. Best model saved to {teacher_path}")

	# --- 2. Train ViT (Baseline) ---
	print("\n" + "="*60 + "\nPHASE 2: TRAINING ViT (BASELINE)\n" + "="*60)
	vit_args = args
	vit_args.model = "vit"
	vit_args.epochs = 50
	config = get_config(vit_args.model, vit_args)
	vit_model = get_model(vit_args.model, config).to(device)
	param_groups = get_param_groups(vit_model, vit_args.weight_decay)
	optimizer = AdamW(param_groups, lr=vit_args.lr)
	criterion = nn.CrossEntropyLoss(label_smoothing=vit_args.label_smoothing)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=vit_args.epochs)
	vit_trainer = Trainer(vit_model, optimizer, criterion, device, scheduler=scheduler, scheduler_type="cosineannealing", writer=SummaryWriter(f"runs/{vit_args.model}_baseline"))
	vit_trainer.train(vit_args.epochs, train_dl, val_dl, save_path=f"{vit_args.save_path}/{vit_args.model}_baseline.pt", config=config, args=vars(vit_args))
	print(f"ViT baseline training complete.")

	# --- 3. Train DeiT (with Distillation) ---
	print("\n" + "="*60 + "\nPHASE 3: TRAINING DeiT (DISTILLATION)\n" + "="*60)
	deit_args = args
	deit_args.model = "vit"
	deit_args.epochs = 50
	deit_args.teacher_model = "resnet18"
	deit_args.teacher_path = teacher_path
	deit_args.alpha, deit_args.tau = 0.5, 1.0
	
	teacher_model = load_teacher_model(deit_args.teacher_path, deit_args.teacher_model, deit_args.num_classes, device)
	config = get_config(deit_args.model, deit_args)
	deit_model = get_model(deit_args.model, config).to(device)
	param_groups = get_param_groups(deit_model, deit_args.weight_decay)
	optimizer = AdamW(param_groups, lr=deit_args.lr)
	criterion = nn.CrossEntropyLoss(label_smoothing=deit_args.label_smoothing)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=deit_args.epochs)
	deit_trainer = DeiTTrainer(deit_model, teacher_model, optimizer, criterion, device, scheduler=scheduler, scheduler_type="cosineannealing", writer=SummaryWriter(f"runs/deit"), alpha=deit_args.alpha, tau=deit_args.tau)
	deit_trainer.train(deit_args.epochs, train_dl, val_dl, save_path=f"{deit_args.save_path}/deit.pt", config=config, args=vars(deit_args))
	print("DeiT distillation training complete.\n\nAll training phases finished.")

if __name__ == "__main__":
	main()
