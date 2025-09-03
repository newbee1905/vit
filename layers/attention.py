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

