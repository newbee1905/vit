import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

def get_positional_embeddings(seq_len, d_model, theta=10000.0):
	"""Sinusoidal positional embeddings."""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(-1, 1)
	dim = torch.arange(d_model, dtype=torch.float, device=device).reshape(1, -1)
	
	div_term = theta ** (torch.div(dim, 2, rounding_mode='floor') * 2 / d_model)
	
	embeddings = torch.zeros(seq_len, d_model, device=device)
	embeddings[:, 0::2] = torch.sin(pos / div_term[0, 0::2])
	embeddings[:, 1::2] = torch.cos(pos / div_term[0, 1::2])
	return embeddings
