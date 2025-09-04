import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
	"""Feedforward block with configurable activation.

	Supports:
	- 'swiglu': uses SiLU on the first half and multiplies with the second half.
	- 'geglu': uses GELU on the first half and multiplies with the second half.
	- 'gelu': standard feedforward with GELU.
	- 'silu': standard feedforward with SiLU.
	"""
	def __init__(self, config):
		super().__init__()

		self.activation = config.activation.lower()
		supported_activations = (
			'swiglu', 'silu', 'geglu', 'gelu', 'reglu', 'relu', 'reglu6', 'relu6'
		)
		if self.activation not in supported_activations:
			raise ValueError(f"Unknown activation type: {activation}")

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
	def __init__(self, config):
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
		self.drop = nn.Dropout(self.dropout)

	def forward(self, x, H, W):
		B, N, C = x.shape
		x = self.linear1(x)

		if self.activation in ('swiglu', 'geglu'):
			gate, x = x.chunk(2, dim=-1)
			x = gate * self.act_fn(x)
		else:
			x = self.act_fn(x)
		
		x_conv = x.transpose(1, 2).reshape(B, C, H, W)
		x_conv = self.bn(self.conv(x_conv))
		x_conv = x_conv.reshape(B, C, N).transpose(1, 2)
		
		x = x + x_conv
		
		x = self.linear2(x)
		x = self.drop(x)

		return x
