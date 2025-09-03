import torch
import torch.nn as nn
import torch.nn.functional as F

# RMSNorm from gemma
# https://github.com/google/gemma_pytorch/blob/main/gemma/model.py
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
		# Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
		# See https://github.com/huggingface/transformers/pull/29402
		output = self._norm(x.float())
		if self.add_unit_offset:
			output = output * (1 + self.weight.float())
		else:
			output = output * self.weight.float()

		return output.type_as(x)

# TODO: recheck the paper again cause currently
# it is slower in training compare to RMSNorm
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

# Based on https://github.com/CyndxAI/QKNorm/blob/main/QKNorm/layers.py
class QKNorm(nn.Module):
	def __init__(self, d_head: int, max_seq_len: int, eps: float = 1e-6):
		super().__init__()
		self.eps = eps

		# QK Norm paper will apply the gain scale after matrix multiplication
		# between q and k
		# Due to the usage of optimised scaled_dot_production from torch
		# using our gain would be not possible, by making the gain to be
		# sqrt_q we would have new formula
		# (sqrt_q * Q) * (sqrt_q * K)_T
		# = (sqrt_q * Q) * (sqrt_q * K_T)
		# = q * (Q . K_T)

		# gain_ = np.log2(max_seq_len ** 2 - max_seq_len)
		val = float(max_seq_len * max_seq_len - max_seq_len)

		# in case of numerically invalid values
		if val < 1.0:
			val = 2.0

		init_gain = math.sqrt(math.log2(val))
		self.gain = nn.Parameter(torch.tensor(init_gain))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		l2_norm_sq = x.pow(2).sum(dim=-1, keepdim=True)
		inv_norm = torch.rsqrt(l2_norm_sq + self.eps)

		out = x * inv_norm * self.gain

		return out
