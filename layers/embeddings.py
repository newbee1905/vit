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
