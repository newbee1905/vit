import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

def patchify(images, n_patch):
	_, _, h, w = images.shape

	assert h == w, "Patchify method is implemented for square images only"

	return rearrange(
		images,
		'b c (p1 h_patch) (p2 w_patch) -> b (p1 p2) (c h_patch w_patch)',
		p1=n_patch, p2=n_patch, h_patch=h // n_patch, w_patch=w // n_patch,
	)

class ConvEmbedding(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
		self.norm = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		x = self.conv(x)
		x = self.norm(x)
		x = rearrange(x, 'b c h w -> b (h w) c')
		return x

class PatchMerging(nn.Module):
	"""
	Patch Merging Layer for the Swin Transformer.
	"""
	def __init__(self, config):
		super().__init__()

		self.d_model = config.d_model
		self.reduction = nn.Linear(4 * self.d_model, 2 * self.d_model, bias=False)
		self.norm = config.norm(4 * self.d_model)

	def forward(self, x, h, w):
		bsz, seq_len, _ = x.shape
		assert seq_len == h * w, "input feature has wrong size"

		x = x.view(bsz, h, w, -1)

		# Pad if H or W is not divisible by 2
		pad_input = (h % 2 == 1) or (w % 2 == 1)
		if pad_input:
			x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))

		# bsz h/2 w/2 -1 
		x0 = x[:, 0::2, 0::2, :]
		x1 = x[:, 1::2, 0::2, :]
		x2 = x[:, 0::2, 1::2, :]
		x3 = x[:, 1::2, 1::2, :]

		x = torch.cat([x0, x1, x2, x3], -1)
		x = x.view(B, h * w / 4, -1)

		x = self.norm(x)
		x = self.reduction(x)

		return x, h // 2, w // 2
