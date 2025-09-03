import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from layers.norm import QKNorm

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
		
		# Apply conv projection
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

