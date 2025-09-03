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
