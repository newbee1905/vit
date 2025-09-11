import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.utils import get_positional_embeddings
from layers.embeddings import patchify
from layers.vit import ViTBlock
from layers.attention import QuadrangleAttention

class ViT(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.chw = config.chw # (C, H, W)

		self.d_model = config.d_model
		self.n_patch = config.n_patch
		self.n_block = config.n_block

		assert self.chw[1] % self.n_patch == 0, "Input shape not entirely divisible by number of patch"
		assert self.chw[2] % self.n_patch == 0, "Input shape not entirely divisible by number of patch"

		self.patch_size = (self.chw[1] // self.n_patch, self.chw[2] // self.n_patch)
		self.h_grid = self.chw[1] // self.patch_size[0]
		self.w_grid = self.chw[2] // self.patch_size[1]

		self.input_d = self.chw[0] * self.patch_size[0] * self.patch_size[1]
		self.linear_mapper = nn.Linear(self.input_d, self.d_model)

		self.class_token = nn.Parameter(torch.rand(1, self.d_model))	

		# self.register_buffer('pos_embed', get_positional_embeddings(n_patch ** 2 + 1, d_model), persistent=False)
		self.pos_embed = nn.Parameter(torch.randn(1, self.n_patch ** 2 + 1, self.d_model))

		self.blocks = nn.ModuleList([ViTBlock(config) for _ in range(self.n_block)])

		self.mlp = nn.Linear(self.d_model, config.out_d)

		self.is_quadrangle_attention = config.attention == QuadrangleAttention

	def forward(self, images):
		bsz = images.size(0)

		patch = patchify(images, self.n_patch)
		tokens = self.linear_mapper(patch)

		tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

		pos_embed = self.pos_embed.repeat(bsz, 1, 1)
		out = tokens + pos_embed

		total_reg_loss = 0.0
		for block in self.blocks:
			if self.is_quadrangle_attention:
				out, regularization_loss = block(out, self.h_grid, self.w_grid)
				total_reg_loss += regularization_loss
			else:
				out, _ = block(out)

		# [CLS] token
		out = out[:, 0]

		output = self.mlp(out)
		if self.is_quadrangle_attention:
			return output, total_reg_loss
		else:
			return output
