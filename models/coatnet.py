import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from layers.vit import ViTBlock
from layers.norm import DyT, RMSNorm
from layers.conv import MBConv

class CoAtNet(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.activation = activation.lower()
		if self.activation == "gelu":
			self.act_fn = nn.GELU
		else:
			self.act_fn = nn.SiLU

		# Convolutional Stem
		self.s0 = nn.Sequential(
			nn.Conv2d(config.chw[0], config.dims[0], kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(dims[0]),
			self.act_fn(),
		)
		
		# Convolutional Stages (MBConv)
		self.s1 = self._make_conv_stage(config.dims[0], config.dims[1], config.depths[0], stride=2)
		self.s2 = self._make_conv_stage(config.dims[1], config.dims[2], config.depths[1], stride=2)

		# Transformer Stages
		self.s3_pe = nn.Conv2d(config.dims[2], config.dims[3], 1) # Patch embedding
		self.s3 = self._make_transformer_stage(config.dims[3], config.depths[2], config.num_heads, cofnig.norm, config.activation)

		self.s4_pe = nn.Conv2d(config.dims[3], config.dims[4], 2, 2) # Patch merging
		self.s4 = self._make_transformer_stage(config.dims[4], config.depths[3], config.num_heads, cofnig.norm, config.activation)

		self.head = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(config.dims[-1], config.num_classes)
		)

	def _make_conv_stage(self, in_chans, out_chans, depth, stride):
		layers = [MBConv(in_chans, out_chans, stride=stride)]

		for _ in range(depth - 1):
			layers.append(MBConv(out_chans, out_chans, stride=1))

		return nn.Sequential(*layers)

	def _make_transformer_stage(self, dim, depth, num_heads, norm=RMSNorm, activation="silu"):
		class MHAConfig:
			d_model, n_head, dropout, use_qk_norm, use_layer_scale = dim, num_heads, 0.1, False, False
			norm, activation, d_ff = norm, activation, dim * 4
		
		blocks = []
		for _ in range(depth):
			blocks.append(ViTBlock(MHAConfig()))

		return nn.ModuleList(blocks)

	def forward(self, x):
		B = x.shape[0]

		# Conv stages
		x = self.s0(x)
		x = self.s1(x)
		x = self.s2(x)
		
		# Transformer stage 1
		x = self.s3_pe(x)
		_, C, H, W = x.shape
		x = rearrange(x, 'b c h w -> b (h w) c')
		for blk in self.s3:
			x = blk(x)
		x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

		# Transformer stage 2
		x = self.s4_pe(x)
		_, C, H, W = x.shape
		x = rearrange(x, 'b c h w -> b (h w) c')
		for blk in self.s4:
			x = blk(x)
		x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
		
		# Head
		x = self.head(x)

		return x
