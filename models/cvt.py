import torch
import torch.nn as nn
from einops import rearrange

from layers.embeddings import ConvEmbedding
from layers.cvt import CvTBlock

class CvT(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.distillation = config.distillation
		
		# Stage 1
		self.stage1_embedding = ConvEmbedding(
			in_channels=config.chw[0],
			out_channels=config.s1_emb_dim,
			kernel_size=config.s1_emb_kernel,
			stride=config.s1_emb_stride,
			padding=config.s1_emb_pad,
		)
		self.stage1_blocks = nn.ModuleList([
			CvTBlock(
				config=self.get_stage_config(config.s1_emb_dim, config.s1_heads, config.s1_mlp_ratio),
				kernel_size=config.s1_qkv_kernel,
				qkv_stride=config.s1_qkv_stride,
				padding=config.s1_qkv_pad,
			) for _ in range(config.s1_depth)
		])

		# Stage 2
		self.stage2_embedding = ConvEmbedding(
			in_channels=config.s1_emb_dim,
			out_channels=config.s2_emb_dim,
			kernel_size=config.s2_emb_kernel,
			stride=config.s2_emb_stride,
			padding=config.s2_emb_pad,
		)
		self.stage2_blocks = nn.ModuleList([
			CvTBlock(
				config=self.get_stage_config(config.s2_emb_dim, config.s2_heads, config.s2_mlp_ratio),
				kernel_size=config.s2_qkv_kernel,
				qkv_stride=config.s2_qkv_stride,
				padding=config.s2_qkv_pad,
			) for _ in range(config.s2_depth)
		])
		
		# Stage 3
		self.stage3_embedding = ConvEmbedding(
			in_channels=config.s2_emb_dim,
			out_channels=config.s3_emb_dim,
			kernel_size=config.s3_emb_kernel,
			stride=config.s3_emb_stride,
			padding=config.s3_emb_pad,
		)
		self.stage3_blocks = nn.ModuleList([
			CvTBlock(
				config=self.get_stage_config(config.s3_emb_dim, config.s3_heads, config.s3_mlp_ratio),
				kernel_size=config.s3_qkv_kernel,
				qkv_stride=config.s3_qkv_stride,
				padding=config.s3_qkv_pad,
			) for _ in range(config.s3_depth)
		])

		self.norm = config.norm(config.s3_emb_dim)
		self.head = nn.Linear(config.s3_emb_dim, config.out_d)
		if self.distillation:
			self.distill_head = nn.Linear(config.s3_emb_dim, config.out_d)


	def get_stage_config(self, d_model, n_head, mlp_ratio):
		class StageConfig:
			pass
		s_config = StageConfig()
		s_config.d_model = d_model
		s_config.d_ff = int(d_model * mlp_ratio)
		s_config.n_head = n_head
		s_config.norm = self.config.norm
		s_config.activation = self.config.activation
		s_config.dropout = self.config.dropout
		s_config.use_layer_scale = self.config.use_layer_scale
		s_config.activation = self.config.activation
		return s_config
		
	def forward(self, x):
		h, w = self.config.chw[1], self.config.chw[2]
		
		# Stage 1
		x = self.stage1_embedding(x)
		h, w = h // self.config.s1_emb_stride, w // self.config.s1_emb_stride
		for blk in self.stage1_blocks:
			x, h, w = blk(x, h, w)
		
		# Stage 2
		x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
		x = self.stage2_embedding(x)
		h, w = h // self.config.s2_emb_stride, w // self.config.s2_emb_stride
		for blk in self.stage2_blocks:
			x, h, w = blk(x, h, w)

		# Stage 3
		x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
		x = self.stage3_embedding(x)
		h, w = h // self.config.s3_emb_stride, w // self.config.s3_emb_stride
		for blk in self.stage3_blocks:
			x, h, w = blk(x, h, w)
		
		# Head
		x = x.mean(dim=1) # Average pooling
		x = self.norm(x)
		
		if self.distillation:
			return self.head(x), self.distill_head(x)
		
		return self.head(x)
