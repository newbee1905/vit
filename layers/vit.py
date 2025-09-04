import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import MHA
from layers.transformers import FeedForward

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
