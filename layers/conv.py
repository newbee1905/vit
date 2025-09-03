import torch
import torch.nn as nn
import torch.nn.functional as F

class MBConv(nn.Module):
	def __init__(self, in_chans, out_chans, stride=1, expand_ratio=4, activation="gelu"):
		super().__init__()

		self.stride = stride
		hidden_dim = in_chans * expand_ratio

		self.activation = activation.lower()
		if self.activation == "gelu":
			self.act_fn = nn.GELU
		if self.activation == "silu":
			self.act_fn = nn.SiLU
		else:
			self.act_fn = nn.ReLU
		
		self.use_res_connect = self.stride == 1 and in_chans == out_chans

		layers = []

		# Point-wise expansion
		layers.append(nn.Conv2d(in_chans, hidden_dim, 1, 1, 0, bias=False))
		layers.append(nn.BatchNorm2d(hidden_dim))
		layers.append(self.act_fn())

		# Depth-wise convolution
		layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
		layers.append(nn.BatchNorm2d(hidden_dim))
		layers.append(self.act_fn())

		# Point-wise projection
		layers.append(nn.Conv2d(hidden_dim, out_chans, 1, 1, 0, bias=False))
		layers.append(nn.BatchNorm2d(out_chans))
		
		self.conv = nn.Sequential(*layers)

	def forward(self, x):
		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)
