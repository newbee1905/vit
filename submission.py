# File: config.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.norm import DyT, RMSNorm
from layers.attention import MHA, QuadrangleAttention

# from dataclasses import dataclass

class ViTConfig:
	d_model=192
	d_ff=768

	n_head=16
	n_block=4
	# n_patch=16
	n_patch=4
	out_d = 100

	attention=QuadrangleAttention
	window_size=[2, 2]
	norm=RMSNorm
	activation="swiglu"

	dropout=0.3

	use_layer_scale=True
	use_qk_norm=False

	# chw=(3, 224, 224)
	chw=(3, 32, 32)

	kernel_size=16
	stride=16

class CoAtNetConfig:
	dims=[32, 64, 128, 192, 192]
	depths=[2, 2, 2, 2]
	num_heads=16

	# chw=(3, 224, 224)
	chw=(3, 32, 32)

	activation="silu"
	norm=RMSNorm

	dropout=0.3

	out_d = 100 

	window_size = [2, 2]
	use_layer_scale=True

class CvTConfig:
	# Based on CvT-13
	chw = (3, 32, 32)
	out_d = 100 
	
	# Stage 1
	s1_emb_kernel = 7
	s1_emb_stride = 4
	s1_emb_pad = 2
	s1_emb_dim = 64
	s1_depth = 1
	s1_heads = 1
	s1_mlp_ratio = 4.0
	s1_qkv_kernel = 3
	s1_qkv_stride = 1
	s1_qkv_pad = 1

	# Stage 2
	s2_emb_kernel = 3
	s2_emb_stride = 2
	s2_emb_pad = 1
	s2_emb_dim = 192
	s2_depth = 2
	s2_heads = 3
	s2_mlp_ratio = 4.0
	s2_qkv_kernel = 3
	s2_qkv_stride = 2
	s2_qkv_pad = 1

	# Stage 3
	s3_emb_kernel = 3
	s3_emb_stride = 2
	s3_emb_pad = 1
	s3_emb_dim = 384
	s3_depth = 10
	s3_heads = 6
	s3_mlp_ratio = 4.0
	s3_qkv_kernel = 3
	s3_qkv_stride = 2
	s3_qkv_pad = 1

	dropout=0.3
	norm=RMSNorm
	activation="silu"
	use_layer_scale=False
# File: dataset.py
import os
import requests
import zipfile
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from tqdm import tqdm

class TinyImageNet(Dataset):
	url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
	filename = 'tiny-imagenet-200.zip'
	folder_name = 'tiny-imagenet-200'
	
	def __init__(self, root, split='train', transform=None, download=False, test_size=0.1, random_state=0):
		self.root = os.path.expanduser(root)
		self.split = split
		self.transform = transform
		self.test_size = test_size
		self.random_state = random_state
		
		if download:
			self.download()
			
		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')
		
		self.data = []
		self.targets = []
		
		self.class_to_idx = {}
		with open(os.path.join(self.root, self.folder_name, 'wnids.txt'), 'r') as f:
			for i, line in enumerate(f):
				self.class_to_idx[line.strip()] = i
		
		self._load_data()
	
	def _load_data(self):
		"""Load and split the data based on the specified split"""
		if self.split == 'train':
			self._load_from_folder('train')
		elif self.split == 'val':
			self._load_from_folder('val')
		elif self.split == 'test':
			self._load_from_folder('test')
		else:
			raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")

	def _load_from_folder(self, folder):
		"""Load images and labels directly from the specified subfolder"""
		if folder == 'train':
			split_dir = os.path.join(self.root, self.folder_name, folder)
			for class_folder in os.listdir(split_dir):
				class_idx = self.class_to_idx[class_folder]
				class_dir = os.path.join(split_dir, class_folder, 'images')
				for img_name in os.listdir(class_dir):
					img_path = os.path.join(class_dir, img_name)
					self.data.append(img_path)
					self.targets.append(class_idx)
		elif folder == 'val':
			split_dir = os.path.join(self.root, self.folder_name, folder)
			with open(os.path.join(split_dir, 'val_annotations.txt'), 'r') as f:
				for line in f:
					parts = line.strip().split('\t')
					img_name, class_id = parts[0], parts[1]
					img_path = os.path.join(split_dir, 'images', img_name)
					self.data.append(img_path)
					self.targets.append(self.class_to_idx[class_id])
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		img_path = self.data[idx]
		target = self.targets[idx]
		
		with open(img_path, 'rb') as f:
			img = Image.open(f).convert('RGB')
		
		if self.transform:
			img = self.transform(img)
			
		return img, target
	
	def _check_integrity(self):
		return os.path.isdir(os.path.join(self.root, self.folder_name))
	
	def download(self):
		if self._check_integrity():
			print('Files already downloaded and verified')
			return
		
		download_url(self.url, self.root, self.filename)
		with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zip_ref:
			zip_ref.extractall(self.root)
	
	def get_split_info(self):
		"""Return information about the dataset splits"""
		class_counts = {}
		for target in self.targets:
			class_counts[target] = class_counts.get(target, 0) + 1
			
		return {
			'split': self.split,
			'total_samples': len(self.data),
			'num_classes': len(self.class_to_idx),
			'samples_per_class': class_counts
		}

# File: deit_trainer.py
import torch
import torch.nn.functional as F
from train import Trainer

import torch
import torch.nn.functional as F

class DeiTTrainer(Trainer):
	def __init__(self, model, teacher_model, optimizer, criterion, device, scheduler=None, scheduler_type=None, writer=None, alpha=0.5, tau=1.0):
		super().__init__(
			model, optimizer, criterion, device,
			scheduler=scheduler, scheduler_type=scheduler_type, writer=writer
		)
		self.teacher_model = teacher_model
		self.teacher_model.eval()

		for param in self.teacher_model.parameters():
			param.requires_grad = False

		self.alpha = alpha
		self.tau = tau

	def compute_loss(self, y_hat, y, x=None):
		if isinstance(y_hat, tuple):
			student_pred, regularization_loss = y_hat
		else:
			student_pred = y_hat
			regularization_loss = 0

		base_loss = self.criterion(student_pred, y)
		
		if x is not None and self.teacher_model is not None:
			with torch.no_grad():
				teacher_pred = self.teacher_model(x)

				if isinstance(teacher_pred, tuple):
					teacher_pred = teacher_pred[0]
			
			distill_loss = F.kl_div(
				F.log_softmax(student_pred / self.tau, dim=1),
				F.softmax(teacher_pred / self.tau, dim=1),
				reduction='batchmean'
			) * (self.tau * self.tau)

			total_loss = (1 - self.alpha) * base_loss + self.alpha * distill_loss
		else:
			total_loss = base_loss
		
		if isinstance(y_hat, tuple):
			total_loss += regularization_loss

		return total_loss
# File: get_resnet_weights.py
import torch
import os

os.makedirs('checkpoints', exist_ok=True)

model_urls = {
	'resnet20': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20-12fca82f.th',
	'resnet32': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet32-d509ac18.th',
}

for model_name, model_url in model_urls.items():
	local_path = f"checkpoints/{model_name}_cifar10.pth"
	state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True, map_location=torch.device("cpu"))["state_dict"]

	new_state_dict = {}
	for k, v in state_dict.items():
		if 'linear' in k:
			k = k.replace('linear', 'fc')

		if k.startswith('module.'):
			# Remove the "module." prefix
			new_state_dict[k[7:]] = v
		else:
			new_state_dict[k] = v

	torch.save(new_state_dict, local_path)

	print(f"{model_name} CIFAR-10 weights saved to {local_path}")
# File: layers/attention.py
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

class QuadrangleAttention(nn.Module):
	def __init__(self, config, lambda_reg=1.0):
		super().__init__()

		self.d_model = config.d_model
		self.n_head = config.n_head

		self.d_head = self.d_model // self.n_head

		self.window_size = config.window_size
		self.lambda_reg = lambda_reg

		self.qkv_proj = nn.Linear(self.d_model, self.d_model * 3, bias=False)
		self.out_proj = nn.Linear(self.d_model, self.d_model)

		self.dropout = nn.Dropout(config.dropout)

		self.use_qk_norm = getattr(config, 'use_qk_norm', False)
		if self.use_qk_norm:
			self.q_norm = QKNorm(self.d_head)
			self.k_norm = QKNorm(self.d_head)

		# Quadrangle Prediction Module (QPM)
		self.qpm = nn.Sequential(
			nn.AvgPool2d(kernel_size=self.window_size, stride=self.window_size),
			nn.SiLU(),
			nn.Conv2d(self.d_model, 9 * self.n_head, kernel_size=1)
		)
		
		self.qpm[-1].weight.data.zero_()
		self.qpm[-1].bias.data.zero_()

	def _get_transform_matrix(self, t_list):
		scale_x, scale_y, shear_x, shear_y, rotation, translate_x, translate_y, perspective_x, perspective_y = t_list

		bsz, num_windows, n_head = scale_x.shape

		# Scaling transformation matrix
		t_scale = torch.stack([
			(scale_x + 1), torch.zeros_like(scale_x), torch.zeros_like(scale_x),
			torch.zeros_like(scale_x), (scale_y + 1), torch.zeros_like(scale_x),
			torch.zeros_like(scale_x), torch.zeros_like(scale_x), torch.ones_like(scale_x)
		], dim=-1).view(bsz, num_windows, n_head, 3, 3)

		# Shearing transformation matrix
		t_shear = torch.stack([
			torch.ones_like(shear_x), shear_x, torch.zeros_like(shear_x),
			shear_y, torch.ones_like(shear_x), torch.zeros_like(shear_x),
			torch.zeros_like(shear_x), torch.zeros_like(shear_x), torch.ones_like(shear_x)
		], dim=-1).view(bsz, num_windows, n_head, 3, 3)

		# Rotation transformation matrix
		t_rotate = torch.stack([
			torch.cos(rotation), -torch.sin(rotation), torch.zeros_like(rotation),
			torch.sin(rotation), torch.cos(rotation), torch.zeros_like(rotation),
			torch.zeros_like(rotation), torch.zeros_like(rotation), torch.ones_like(rotation)
		], dim=-1).view(bsz, num_windows, n_head, 3, 3)

		# Translation transformation matrix
		t_translate = torch.stack([
			torch.ones_like(translate_x), torch.zeros_like(translate_x), translate_x,
			torch.zeros_like(translate_x), torch.ones_like(translate_x), translate_y,
			torch.zeros_like(translate_x), torch.zeros_like(translate_x), torch.ones_like(translate_x)
		], dim=-1).view(bsz, num_windows, n_head, 3, 3)

		# Perspective (or projection) transformation matrix
		t_perspective = torch.stack([
			torch.ones_like(perspective_x), torch.zeros_like(perspective_x), torch.zeros_like(perspective_x),
			torch.zeros_like(perspective_x), torch.ones_like(perspective_x), torch.zeros_like(perspective_x),
			perspective_x, perspective_y, torch.ones_like(perspective_x)
		], dim=-1).view(bsz, num_windows, n_head, 3, 3)

		transform_matrix = t_scale @ t_shear @ t_rotate @ t_translate @ t_perspective
		return transform_matrix

	def forward(self, x, h, w):
		bsz, seq_len, d_model = x.shape
		x_reshaped = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
		
		qkv_proj = self.qkv_proj(x)
		q_proj, k_proj, v_proj = qkv_proj.chunk(3, dim=-1)

		t = self.qpm(x_reshaped)
		
		num_windows_h, num_windows_w = t.shape[-2:]
		n_win = num_windows_h * num_windows_w
		
		t_per_head = rearrange(
			t, 'b (t_dim n_head) num_win_h num_win_w -> b (num_win_h num_win_w) n_head t_dim', 
			n_head=self.n_head, t_dim=9
		)
		t_unpacked = t_per_head.chunk(9, dim=-1)
		# Remove the extra dimension from chunking
		t_unpacked = [param.squeeze(-1) for param in t_unpacked]

		transform_matrix_per_head = self._get_transform_matrix(t_unpacked)

		# Generate default window coordinates
		window_h, window_w = self.window_size
		coords = torch.stack(
			torch.meshgrid(
				torch.arange(window_h, device=x.device), 
				torch.arange(window_w, device=x.device), 
				indexing='ij'
			), dim=-1
		).float()

		# Center coordinates around (0, 0)
		coords = coords - torch.tensor([window_h/2, window_w/2], device=x.device)
		coords = coords.reshape(-1, 2)
		coords = F.pad(coords, (0, 1), value=1.0) # Add Z=1 for projective transform

		# Expand coordinates for all batch, heads, and windows
		coords = coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)
		coords = coords.repeat(bsz, n_win, self.n_head, 1, 1)
		
		# Apply transformation to get new coordinates
		new_coords_hom = transform_matrix_per_head @ coords.transpose(-1, -2)
		new_coords = new_coords_hom[:, :, :, :2, :] / (new_coords_hom[:, :, :, 2:, :] + 1e-6)
		
		# [bsz, n_win, n_head, 2, num_points] -> [bsz, n_head, n_win, num_points, 2]
		new_coords_norm = new_coords.permute(0, 2, 1, 4, 3)
		
		# Calculate window centers
		center_y = torch.arange(num_windows_h, device=x.device).float() * window_h + window_h/2
		center_x = torch.arange(num_windows_w, device=x.device).float() * window_w + window_w/2
		center_grid = torch.stack(torch.meshgrid(center_y, center_x, indexing='ij'), dim=-1)
		center_grid = center_grid.view(-1, 2)  # [n_win, 2] in (y, x) order
		
		# Add window centers to get absolute coordinates
		new_coords_abs = new_coords_norm + center_grid.view(1, 1, -1, 1, 2).to(x.device)
		
		new_coords_normed = (new_coords_abs / torch.tensor([h, w], device=x.device).view(1, 1, 1, 1, 2)) * 2 - 1
		new_coords_normed = new_coords_normed.flip(-1)

		# regularization Loss - penalize coordinates outside [-1, 1]
		regularization_loss = torch.sum(
			torch.relu(torch.abs(new_coords_normed) - 1) ** 2
		)

		q = rearrange(q_proj, 'b n (h d) -> b h n d', h=self.n_head)
		k = rearrange(k_proj, 'b n (h d) -> b h n d', h=self.n_head)
		v = rearrange(v_proj, 'b n (h d) -> b h n d', h=self.n_head)
		
		k_reshaped = rearrange(k, 'b h (H W) d -> b h d H W', H=h, W=w)
		v_reshaped = rearrange(v, 'b h (H W) d -> b h d H W', H=h, W=w)
		
		new_coords_sample = rearrange(new_coords_normed, 'b h n_win num_samples c -> (b h) n_win num_samples c')
		
		k_flat = rearrange(k_reshaped, 'b h d H W -> (b h) d H W')
		v_flat = rearrange(v_reshaped, 'b h d H W -> (b h) d H W')
		
		k_sampled_list = []
		v_sampled_list = []
		
		for win_idx in range(n_win):
			k_sampled_win = F.grid_sample(
				k_flat,
				new_coords_sample[:, win_idx:win_idx+1, :, :].view(-1, window_h, window_w, 2),
				mode='bilinear', padding_mode='zeros', align_corners=True
			)
			v_sampled_win = F.grid_sample(
				v_flat,
				new_coords_sample[:, win_idx:win_idx+1, :, :].view(-1, window_h, window_w, 2),
				mode='bilinear', padding_mode='zeros', align_corners=True
			)
			k_sampled_list.append(k_sampled_win)
			v_sampled_list.append(v_sampled_win)
		
		k_sampled = torch.stack(k_sampled_list, dim=1)  # [b*h, n_win, d, window_h, window_w]
		v_sampled = torch.stack(v_sampled_list, dim=1)
		
		k_sampled = rearrange(k_sampled, '(b h) n_win d wh ww -> b h (n_win wh ww) d', b=bsz, h=self.n_head)
		v_sampled = rearrange(v_sampled, '(b h) n_win d wh ww -> b h (n_win wh ww) d', b=bsz, h=self.n_head)
		
		if self.use_qk_norm:
			q = self.q_norm(q)
			k_sampled = self.k_norm(k_sampled)

		attn_output = F.scaled_dot_product_attention(
			q, k_sampled, v_sampled,
			dropout_p=self.dropout.p if self.training else 0.0,
			is_causal=False,
		)
		
		attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
		output = self.out_proj(attn_output)

		return output, self.lambda_reg * regularization_loss
# File: layers/conv.py
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
# File: layers/cvt.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import ConvAttention
from layers.transformers import FeedForward

class CvTBlock(nn.Module):
	def __init__(self, config, kernel_size, qkv_stride, padding):
		super().__init__()

		self.d_model = config.d_model
		
		self.attn_norm = config.norm(self.d_model)
		self.attn = ConvAttention(
			dim=self.d_model, 
			n_head=config.n_head,
			kernel_size=kernel_size,
			qkv_stride=qkv_stride,
			padding=padding,
			dropout=config.dropout
		)
		self.attn_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.attn_dropout = nn.Dropout(config.dropout)

		self.ff_norm = config.norm(self.d_model)
		self.ff = FeedForward(config)

		self.ff_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.ff_dropout = nn.Dropout(config.dropout)

	def forward(self, x, h, w):
		norm_x = self.attn_norm(x)
		attn_out = self.attn(norm_x, h, w)
		attn_out = self.attn_dropout(attn_out)
		
		if self.attn_layer_scale is not None:
			x = x + self.attn_layer_scale * attn_out
		else:
			x = x + attn_out

		norm_x = self.ff_norm(x)
		ff_out = self.ff(norm_x)
		ff_out = self.ff_dropout(ff_out)
		if self.ff_layer_scale is not None:
			x = x + self.ff_layer_scale * ff_out
		else:
			x = x + ff_out
		
		return x, h, w
# File: layers/embeddings.py
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
# File: layers/norm.py
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
# File: layers/transformers.py
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
# File: layers/vit.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import MHA, QuadrangleAttention
from layers.transformers import FeedForward

class ViTBlock(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.d_model = config.d_model
		self.n_head = config.n_head
		
		self.attn_norm = config.norm(self.d_model)
		self.attn = config.attention(config)
		self.attn_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.attn_dropout = nn.Dropout(config.dropout)

		self.ff_norm = config.norm(self.d_model)
		self.ff = FeedForward(config)
		self.ff_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.ff_dropout = nn.Dropout(config.dropout)

		self.is_quadrangle_attention = isinstance(self.attn, QuadrangleAttention)

	def forward(self, x, h=None, w=None):
		cls_token, patch_tokens = x[:, :1, :], x[:, 1:, :]

		if self.is_quadrangle_attention:
			norm_patch_tokens = self.attn_norm(patch_tokens) 
			attn_out_patches, regularization_loss = self.attn(norm_patch_tokens, h, w)
			
			attn_out_cls = self.attn_norm(cls_token)
			
			attn_out = torch.cat([attn_out_cls, attn_out_patches], dim=1)
		else:
			norm_x = self.attn_norm(x)
			attn_out = self.attn(norm_x)

			regularization_loss = 0.0

		attn_out = self.attn_dropout(attn_out)
		if self.attn_layer_scale is not None:
			x = x + self.attn_layer_scale * attn_out
		else:
			x = x + attn_out

		norm_x = self.ff_norm(x)
		ff_out = self.ff(norm_x)
		ff_out = self.ff_dropout(ff_out)
		if self.ff_layer_scale is not None:
			x = x + self.ff_layer_scale * ff_out
		else:
			x = x + ff_out

		return x, regularization_loss
# File: main.py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import CIFAR100

from dataset import TinyImageNet
from train import Trainer
from deit_trainer import DeiTTrainer

from utils import parse_args, get_config, get_model, get_param_groups, set_seed, load_teacher_model
from schedulers import WarmupScheduler

args = parse_args()
set_seed(args.seed)

writer = SummaryWriter(f"runs/{args.model}")

print("=" * 60)
print("Training Configuration:")
print("=" * 60)
for arg in vars(args):
	print(f"{arg:20}: {getattr(args, arg)}")
print("=" * 60)

print(f"\nTraining {args.model.upper()} model")
print(f"Distillation: {'Enabled' if args.distillation else 'Disabled'}")
if args.distillation:
	print(f"Teacher model: {args.teacher_model.upper()}")
	if args.teacher_path:
		print(f"Teacher path: {args.teacher_path}")
	print(f"Alpha (distillation weight): {args.alpha}")
	print(f"Temperature: {args.tau}")

if args.device == 'auto':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
	device = torch.device(args.device)

print(f"\nUsing device: {device}")
if device.type == 'cuda':
	print(f"GPU: {torch.cuda.get_device_name(device)}")
	print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

train_transform = v2.Compose([
	v2.ToImage(),
	v2.TrivialAugmentWide(),
	v2.RandomResizedCrop(32, scale=(0.7, 1.0)),
	v2.RandomHorizontalFlip(),
	v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
	v2.ToDtype(torch.float32, scale=True),
	v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
	v2.RandomErasing(p=0.25, scale=(0.02, 0.2)),
])

test_transform = v2.Compose([
	v2.ToImage(),
	v2.Resize(32),
	v2.ToDtype(torch.float32, scale=True),
	v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

print(f"\nLoading dataset from {args.data_root}")
train_ds = CIFAR100(root='./datasets', train=True, download=True, transform=train_transform)
val_ds = CIFAR100(root='./datasets', train=False, transform=test_transform)

print(f"Train samples: {len(train_ds):,}")
print(f"Val samples: {len(val_ds):,}")
print(f"Test samples: {len(val_ds):,}")

train_dl = DataLoader(
	train_ds, shuffle=True, batch_size=args.batch_size, 
	num_workers=args.num_workers, pin_memory=True
)
val_dl = DataLoader(
	val_ds, shuffle=False, batch_size=args.batch_size, 
	num_workers=args.num_workers, pin_memory=True,
)
test_dl = DataLoader(
	val_ds, shuffle=False, batch_size=args.batch_size, 
	num_workers=args.num_workers, pin_memory=True,
)

config = get_config(args.model, args)
model = get_model(args.model, config).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: {args.model.upper()}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

param_groups = get_param_groups(model, args.weight_decay)
if args.optimizer == 'adamw':
	optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
elif args.optimizer == 'adam':
	optimizer = torch.optim.Adam(param_groups, lr=args.lr)
elif args.optimizer == 'sgd':
	optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum)

if args.label_smoothing > 0:
	criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
else:
	criterion = nn.CrossEntropyLoss()

base_scheduler = None
if args.scheduler == "cosineannealing":
	base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
elif args.scheduler == "reduceonplateau":
	base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
elif args.scheduler == "onecycle":
	base_scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer, max_lr=args.lr, steps_per_epoch=len(train_dl), epochs=args.epochs
	)

if args.warmup_steps > 0:
	scheduler = WarmupScheduler(optimizer, warmup_steps=args.warmup_steps, after_scheduler=base_scheduler)
else:
	scheduler = base_scheduler

if args.distillation:
	print(f"\nSetting up distillation training...")
	
	if not args.teacher_path:
		raise ValueError("Must provide --teacher-path for distillation")

	print(f"Loading teacher from {args.teacher_path}")
	teacher_model = load_teacher_model(
		args.teacher_path,
		args.teacher_model,
		args.num_classes,
		device
	)
	teacher_model.eval()
	teacher_params = sum(p.numel() for p in teacher_model.parameters())
	print(f"Teacher parameters: {teacher_params:,}")
	
	trainer = DeiTTrainer(
		model,
		teacher_model,
		optimizer,
		criterion,
		device,
		scheduler=scheduler,
		scheduler_type=args.scheduler,
		writer=writer,
		alpha=args.alpha,
		tau=args.tau
	)
else:
	print(f"\nSetting up standard training...")
	trainer = Trainer(
		model,
		optimizer,
		criterion,
		device,
		scheduler=scheduler,
		scheduler_type=args.scheduler,
		writer=writer,
	)

if args.resume:
	print(f"\nResuming from checkpoint: {args.resume}")
	checkpoint = torch.load(args.resume, map_location=device)
	trainer.load_checkpoint(checkpoint)

print(f"\nStarting training for {args.epochs} epochs...")
trainer.train(
	args.epochs, train_dl, val_dl,
	save_path=f"{args.save_path}/{args.model}.pt",
	config=config,
	args=vars(args)
)

print(f"\n{'='*60}")
print("FINAL EVALUATION")
print(f"{'='*60}")
test_loss, test_acc = trainer.run_one_epoch(test_dl, state='eval')
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.1%}")
print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
print(f"{'='*60}")
# File: models/coatnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from layers.vit import ViTBlock
from layers.norm import DyT, RMSNorm
from layers.conv import MBConv

class CoAtNet(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.activation = activation.lower()
		if self.activation == "gelu":
			self.act_fn = nn.GELU
		else:
			self.act_fn = nn.SiLU

		# Convolutional Stem
		self.s0 = nn.Sequential(
			nn.Conv2d(config.chw[0], config.dims[0], kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(dims[0]),
			self.act_fn(),
		)
		
		# Convolutional Stages (MBConv)
		self.s1 = self._make_conv_stage(config.dims[0], config.dims[1], config.depths[0], stride=2)
		self.s2 = self._make_conv_stage(config.dims[1], config.dims[2], config.depths[1], stride=2)

		# Transformer Stages
		self.s3_pe = nn.Conv2d(config.dims[2], config.dims[3], 1) # Patch embedding
		self.s3 = self._make_transformer_stage(config.dims[3], config.depths[2], config.num_heads, cofnig.norm, config.activation)

		self.s4_pe = nn.Conv2d(config.dims[3], config.dims[4], 2, 2) # Patch merging
		self.s4 = self._make_transformer_stage(config.dims[4], config.depths[3], config.num_heads, cofnig.norm, config.activation)

		self.head = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(config.dims[-1], config.num_classes)
		)

	def _make_conv_stage(self, in_chans, out_chans, depth, stride):
		layers = [MBConv(in_chans, out_chans, stride=stride)]

		for _ in range(depth - 1):
			layers.append(MBConv(out_chans, out_chans, stride=1))

		return nn.Sequential(*layers)

	def _make_transformer_stage(self, dim, depth, num_heads, norm=RMSNorm, activation="silu"):
		class MHAConfig:
			d_model, n_head, dropout, use_qk_norm, use_layer_scale = dim, num_heads, 0.1, False, False
			norm, activation, d_ff = norm, activation, dim * 4
		
		blocks = []
		for _ in range(depth):
			blocks.append(ViTBlock(MHAConfig()))

		return nn.ModuleList(blocks)

	def forward(self, x):
		B = x.shape[0]

		# Conv stages
		x = self.s0(x)
		x = self.s1(x)
		x = self.s2(x)
		
		# Transformer stage 1
		x = self.s3_pe(x)
		_, C, H, W = x.shape
		x = rearrange(x, 'b c h w -> b (h w) c')
		for blk in self.s3:
			x = blk(x)
		x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

		# Transformer stage 2
		x = self.s4_pe(x)
		_, C, H, W = x.shape
		x = rearrange(x, 'b c h w -> b (h w) c')
		for blk in self.s4:
			x = blk(x)
		x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
		
		# Head
		x = self.head(x)

		return x
# File: models/cvt.py
import torch
import torch.nn as nn
from einops import rearrange

from layers.embeddings import ConvEmbedding
from layers.cvt import CvTBlock

class CvT(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		
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
		
		return self.head(x)
# File: models/resnet.py
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
import torch.hub

# Using the code from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
def _weights_init(m):
	classname = m.__class__.__name__
	if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
		torch.nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
	def __init__(self, lambd):
		super(LambdaLayer, self).__init__()
		self.lambd = lambd

	def forward(self, x):
		return self.lambd(x)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, option='A'):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != planes:
			if option == 'A':
				"""
				For CIFAR10 ResNet paper uses option A.
				"""
				self.shortcut = LambdaLayer(lambda x:
											F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
			elif option == 'B':
				self.shortcut = nn.Sequential(
					 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
					 nn.BatchNorm2d(self.expansion * planes)
				)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(ResNet, self).__init__()
		self.in_planes = 16

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
		self.fc = nn.Linear(64, num_classes)

		self.apply(_weights_init)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion

		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = F.avg_pool2d(out, out.size()[3])
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		return out

def resnet_cifar(depth, num_classes=100):
	if depth == 20:
		return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)
	elif depth == 32:
		return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)
	else:
		raise ValueError('Invalid ResNet depth for CIFAR: {}'.format(depth))

def get_model(model_name: str, num_classes=100):
	if model_name == "resnet20":
		model = resnet_cifar(20, num_classes=10)
	elif model_name == "resnet32":
		model = resnet_cifar(32, num_classes=10)
	else:
		raise ValueError(f"Unknown model {model_name}")
		
	local_path = f"checkpoints/{model_name}_cifar10.pth"
	state_dict = torch.load(local_path)
	model.load_state_dict(state_dict)

	model.fc = nn.Linear(model.fc.in_features, num_classes)
	
	return model
# File: models/utils.py
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
# File: models/vit.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.utils import get_positional_embeddings
from layers.embeddings import patchify
from layers.vit import ViTBlock
from layers.attention import QuadrangleAttention

class ViT(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.chw = config.chw # (C, H, W)

		self.d_model = config.d_model
		self.n_patch = config.n_patch
		self.n_block = config.n_block

		assert self.chw[1] % self.n_patch == 0, "Input shape not entirely divisible by number of patch"
		assert self.chw[2] % self.n_patch == 0, "Input shape not entirely divisible by number of patch"

		self.patch_size = (self.chw[1] // self.n_patch, self.chw[2] // self.n_patch)
		self.h_grid = self.chw[1] // self.patch_size[0]
		self.w_grid = self.chw[2] // self.patch_size[1]

		self.input_d = self.chw[0] * self.patch_size[0] * self.patch_size[1]
		self.linear_mapper = nn.Linear(self.input_d, self.d_model)

		self.class_token = nn.Parameter(torch.rand(1, self.d_model))	

		# self.register_buffer('pos_embed', get_positional_embeddings(n_patch ** 2 + 1, d_model), persistent=False)
		self.pos_embed = nn.Parameter(torch.randn(1, self.n_patch ** 2 + 1, self.d_model))

		self.blocks = nn.ModuleList([ViTBlock(config) for _ in range(self.n_block)])

		self.mlp = nn.Linear(self.d_model, config.out_d)

		self.is_quadrangle_attention = config.attention == QuadrangleAttention

	def forward(self, images):
		bsz = images.size(0)

		patch = patchify(images, self.n_patch)
		tokens = self.linear_mapper(patch)

		tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

		pos_embed = self.pos_embed.repeat(bsz, 1, 1)
		out = tokens + pos_embed

		total_reg_loss = 0.0
		for block in self.blocks:
			if self.is_quadrangle_attention:
				out, regularization_loss = block(out, self.h_grid, self.w_grid)
				total_reg_loss += regularization_loss
			else:
				out, _ = block(out)

		# [CLS] token
		out = out[:, 0]

		output = self.mlp(out)
		if self.is_quadrangle_attention:
			return output, total_reg_loss
		else:
			return output
# File: plot.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os

def load_tensorboard_logs(path: str, tag: str) -> pd.DataFrame:
	"""
	Loads scalar data from a TensorBoard log file.
	"""
	ea = event_accumulator.EventAccumulator(path,
		size_guidance={event_accumulator.SCALARS: 0})
	ea.Reload()

	if tag not in ea.Tags()['scalars']:
		print(f"Warning: Tag '{tag}' not found in {path}. Skipping.")
		return pd.DataFrame()

	scalar_events = ea.Scalars(tag)
	steps = [event.step for event in scalar_events]
	values = [event.value for event in scalar_events]

	return pd.DataFrame({'Epoch': steps, 'Value': values})

base_log_dir = './runs'
runs = {
	'deit': os.path.join(base_log_dir, 'deit'),
	'resnet32_finetune': os.path.join(base_log_dir, 'resnet32_finetune'),
	'resnet32_transfer': os.path.join(base_log_dir, 'resnet32_transfer'),
	'vit': os.path.join(base_log_dir, 'vit')
}
tags = {
	'Accuracy/train': ('Accuracy', 'Train'),
	'Accuracy/val': ('Accuracy', 'Validation'),
	'Loss/train': ('Loss', 'Train'),
	'Loss/val': ('Loss', 'Validation')
}

all_data_dfs = []
for model_name, path in runs.items():
	if not os.path.exists(path):
		print(f"Warning: Directory not found at {path}. Skipping model '{model_name}'.")
		continue
	for tag_name, (metric, split) in tags.items():
		df_run = load_tensorboard_logs(path, tag_name)
		if not df_run.empty:
			df_run['Model'] = model_name
			df_run['Metric'] = metric
			df_run['Split'] = split
			all_data_dfs.append(df_run)

if not all_data_dfs:
	print("Error: No data was loaded. Please check your paths and tag names.")
else:
	temp_df = pd.concat(all_data_dfs)
	
	if 'resnet32_transfer' in temp_df['Model'].unique():
		transfer_max_step = temp_df[temp_df['Model'] == 'resnet32_transfer']['Epoch'].max()
		print(f"Offset determined from resnet32_transfer's final step: {transfer_max_step}")

		for i, df in enumerate(all_data_dfs):
			if not df.empty and df['Model'].iloc[0] == 'resnet32_finetune':
				df['Epoch'] += transfer_max_step
				all_data_dfs[i] = df 
	else:
		print("Warning: 'resnet32_transfer' data not found. Cannot apply offset.")


	df_combined = pd.concat(all_data_dfs, ignore_index=True)

	sns.set_theme(style="darkgrid")

	g = sns.relplot(
		data=df_combined,
		x="Epoch",
		y="Value",
		hue="Model",
		col="Split",
		row="Metric",
		kind="line",
		height=4,
		aspect=1.2,
		linewidth=1.5,
		facet_kws={'sharey': False, 'sharex': True}
	)


	g.fig.suptitle("Model Metrics", y=1.03, fontsize=16)
	g.set_titles("{row_name} / {col_name}", size=12)
	g.set_axis_labels("Epoch", "Value", size=12)
	sns.move_legend(g, "lower center", bbox_to_anchor=(.5, -0.08), ncol=4, title=None, frameon=False)
	
	plt.savefig("model_comparison_plot.png", dpi=300, bbox_inches='tight')
	plt.show()
# File: schedulers.py
from torch.optim.lr_scheduler import _LRScheduler

class WarmupScheduler(_LRScheduler):
	def __init__(self, optimizer, warmup_steps, after_scheduler=None, last_epoch=-1):
		self.warmup_steps = warmup_steps
		self.after_scheduler = after_scheduler
		self.finished_warmup = False
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		if self.last_epoch < self.warmup_steps:
			# Linear warmup
			return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]

		if self.after_scheduler:
			if not self.finished_warmup:
				# reset after warmup
				self.after_scheduler.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
				self.finished_warmup = True

			return self.after_scheduler.get_last_lr()

		return [group["lr"] for group in self.optimizer.param_groups]

	def step(self, epoch=None, metrics=None):
		if self.last_epoch < self.warmup_steps:
			return super(WarmupScheduler, self).step(epoch)
		if self.after_scheduler:
			if isinstance(self.after_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
				self.after_scheduler.step(metrics)
			else:
				if epoch is None:
					self.after_scheduler.step(None)
				else:
					self.after_scheduler.step(epoch - self.warmup_steps)
# File: submission.py
# config.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.norm import DyT, RMSNorm
from layers.attention import MHA, QuadrangleAttention

# from dataclasses import dataclass

class ViTConfig:
	d_model=192
	d_ff=768

	n_head=16
	n_block=4
	# n_patch=16
	n_patch=4
	out_d = 100

	attention=QuadrangleAttention
	window_size=[2, 2]
	norm=RMSNorm
	activation="swiglu"

	dropout=0.3

	use_layer_scale=True
	use_qk_norm=False

	# chw=(3, 224, 224)
	chw=(3, 32, 32)

	kernel_size=16
	stride=16

class CoAtNetConfig:
	dims=[32, 64, 128, 192, 192]
	depths=[2, 2, 2, 2]
	num_heads=16

	# chw=(3, 224, 224)
	chw=(3, 32, 32)

	activation="silu"
	norm=RMSNorm

	dropout=0.3

	out_d = 100 

	window_size = [2, 2]
	use_layer_scale=True

class CvTConfig:
	# Based on CvT-13
	chw = (3, 32, 32)
	out_d = 100 
	
	# Stage 1
	s1_emb_kernel = 7
	s1_emb_stride = 4
	s1_emb_pad = 2
	s1_emb_dim = 64
	s1_depth = 1
	s1_heads = 1
	s1_mlp_ratio = 4.0
	s1_qkv_kernel = 3
	s1_qkv_stride = 1
	s1_qkv_pad = 1

	# Stage 2
	s2_emb_kernel = 3
	s2_emb_stride = 2
	s2_emb_pad = 1
	s2_emb_dim = 192
	s2_depth = 2
	s2_heads = 3
	s2_mlp_ratio = 4.0
	s2_qkv_kernel = 3
	s2_qkv_stride = 2
	s2_qkv_pad = 1

	# Stage 3
	s3_emb_kernel = 3
	s3_emb_stride = 2
	s3_emb_pad = 1
	s3_emb_dim = 384
	s3_depth = 10
	s3_heads = 6
	s3_mlp_ratio = 4.0
	s3_qkv_kernel = 3
	s3_qkv_stride = 2
	s3_qkv_pad = 1

	dropout=0.3
	norm=RMSNorm
	activation="silu"
	use_layer_scale=False
# dataset.py
import os
import requests
import zipfile
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from tqdm import tqdm

class TinyImageNet(Dataset):
	url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
	filename = 'tiny-imagenet-200.zip'
	folder_name = 'tiny-imagenet-200'
	
	def __init__(self, root, split='train', transform=None, download=False, test_size=0.1, random_state=0):
		self.root = os.path.expanduser(root)
		self.split = split
		self.transform = transform
		self.test_size = test_size
		self.random_state = random_state
		
		if download:
			self.download()
			
		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')
		
		self.data = []
		self.targets = []
		
		self.class_to_idx = {}
		with open(os.path.join(self.root, self.folder_name, 'wnids.txt'), 'r') as f:
			for i, line in enumerate(f):
				self.class_to_idx[line.strip()] = i
		
		self._load_data()
	
	def _load_data(self):
		"""Load and split the data based on the specified split"""
		if self.split == 'train':
			self._load_from_folder('train')
		elif self.split == 'val':
			self._load_from_folder('val')
		elif self.split == 'test':
			self._load_from_folder('test')
		else:
			raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")

	def _load_from_folder(self, folder):
		"""Load images and labels directly from the specified subfolder"""
		if folder == 'train':
			split_dir = os.path.join(self.root, self.folder_name, folder)
			for class_folder in os.listdir(split_dir):
				class_idx = self.class_to_idx[class_folder]
				class_dir = os.path.join(split_dir, class_folder, 'images')
				for img_name in os.listdir(class_dir):
					img_path = os.path.join(class_dir, img_name)
					self.data.append(img_path)
					self.targets.append(class_idx)
		elif folder == 'val':
			split_dir = os.path.join(self.root, self.folder_name, folder)
			with open(os.path.join(split_dir, 'val_annotations.txt'), 'r') as f:
				for line in f:
					parts = line.strip().split('\t')
					img_name, class_id = parts[0], parts[1]
					img_path = os.path.join(split_dir, 'images', img_name)
					self.data.append(img_path)
					self.targets.append(self.class_to_idx[class_id])
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		img_path = self.data[idx]
		target = self.targets[idx]
		
		with open(img_path, 'rb') as f:
			img = Image.open(f).convert('RGB')
		
		if self.transform:
			img = self.transform(img)
			
		return img, target
	
	def _check_integrity(self):
		return os.path.isdir(os.path.join(self.root, self.folder_name))
	
	def download(self):
		if self._check_integrity():
			print('Files already downloaded and verified')
			return
		
		download_url(self.url, self.root, self.filename)
		with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zip_ref:
			zip_ref.extractall(self.root)
	
	def get_split_info(self):
		"""Return information about the dataset splits"""
		class_counts = {}
		for target in self.targets:
			class_counts[target] = class_counts.get(target, 0) + 1
			
		return {
			'split': self.split,
			'total_samples': len(self.data),
			'num_classes': len(self.class_to_idx),
			'samples_per_class': class_counts
		}

# deit_trainer.py
import torch
import torch.nn.functional as F
from train import Trainer

import torch
import torch.nn.functional as F

class DeiTTrainer(Trainer):
	def __init__(self, model, teacher_model, optimizer, criterion, device, scheduler=None, scheduler_type=None, writer=None, alpha=0.5, tau=1.0):
		super().__init__(
			model, optimizer, criterion, device,
			scheduler=scheduler, scheduler_type=scheduler_type, writer=writer
		)
		self.teacher_model = teacher_model
		self.teacher_model.eval()

		for param in self.teacher_model.parameters():
			param.requires_grad = False

		self.alpha = alpha
		self.tau = tau

	def compute_loss(self, y_hat, y, x=None):
		if isinstance(y_hat, tuple):
			student_pred, regularization_loss = y_hat
		else:
			student_pred = y_hat
			regularization_loss = 0

		base_loss = self.criterion(student_pred, y)
		
		if x is not None and self.teacher_model is not None:
			with torch.no_grad():
				teacher_pred = self.teacher_model(x)

				if isinstance(teacher_pred, tuple):
					teacher_pred = teacher_pred[0]
			
			distill_loss = F.kl_div(
				F.log_softmax(student_pred / self.tau, dim=1),
				F.softmax(teacher_pred / self.tau, dim=1),
				reduction='batchmean'
			) * (self.tau * self.tau)

			total_loss = (1 - self.alpha) * base_loss + self.alpha * distill_loss
		else:
			total_loss = base_loss
		
		if isinstance(y_hat, tuple):
			total_loss += regularization_loss

		return total_loss
# get_resnet_weights.py
import torch
import os

os.makedirs('checkpoints', exist_ok=True)

model_urls = {
	'resnet20': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20-12fca82f.th',
	'resnet32': 'https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet32-d509ac18.th',
}

for model_name, model_url in model_urls.items():
	local_path = f"checkpoints/{model_name}_cifar10.pth"
	state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True, map_location=torch.device("cpu"))["state_dict"]

	new_state_dict = {}
	for k, v in state_dict.items():
		if 'linear' in k:
			k = k.replace('linear', 'fc')

		if k.startswith('module.'):
			# Remove the "module." prefix
			new_state_dict[k[7:]] = v
		else:
			new_state_dict[k] = v

	torch.save(new_state_dict, local_path)

	print(f"{model_name} CIFAR-10 weights saved to {local_path}")
# layers/attention.py
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

class QuadrangleAttention(nn.Module):
	def __init__(self, config, lambda_reg=1.0):
		super().__init__()

		self.d_model = config.d_model
		self.n_head = config.n_head

		self.d_head = self.d_model // self.n_head

		self.window_size = config.window_size
		self.lambda_reg = lambda_reg

		self.qkv_proj = nn.Linear(self.d_model, self.d_model * 3, bias=False)
		self.out_proj = nn.Linear(self.d_model, self.d_model)

		self.dropout = nn.Dropout(config.dropout)

		self.use_qk_norm = getattr(config, 'use_qk_norm', False)
		if self.use_qk_norm:
			self.q_norm = QKNorm(self.d_head)
			self.k_norm = QKNorm(self.d_head)

		# Quadrangle Prediction Module (QPM)
		self.qpm = nn.Sequential(
			nn.AvgPool2d(kernel_size=self.window_size, stride=self.window_size),
			nn.SiLU(),
			nn.Conv2d(self.d_model, 9 * self.n_head, kernel_size=1)
		)
		
		self.qpm[-1].weight.data.zero_()
		self.qpm[-1].bias.data.zero_()

	def _get_transform_matrix(self, t_list):
		scale_x, scale_y, shear_x, shear_y, rotation, translate_x, translate_y, perspective_x, perspective_y = t_list

		bsz, num_windows, n_head = scale_x.shape

		# Scaling transformation matrix
		t_scale = torch.stack([
			(scale_x + 1), torch.zeros_like(scale_x), torch.zeros_like(scale_x),
			torch.zeros_like(scale_x), (scale_y + 1), torch.zeros_like(scale_x),
			torch.zeros_like(scale_x), torch.zeros_like(scale_x), torch.ones_like(scale_x)
		], dim=-1).view(bsz, num_windows, n_head, 3, 3)

		# Shearing transformation matrix
		t_shear = torch.stack([
			torch.ones_like(shear_x), shear_x, torch.zeros_like(shear_x),
			shear_y, torch.ones_like(shear_x), torch.zeros_like(shear_x),
			torch.zeros_like(shear_x), torch.zeros_like(shear_x), torch.ones_like(shear_x)
		], dim=-1).view(bsz, num_windows, n_head, 3, 3)

		# Rotation transformation matrix
		t_rotate = torch.stack([
			torch.cos(rotation), -torch.sin(rotation), torch.zeros_like(rotation),
			torch.sin(rotation), torch.cos(rotation), torch.zeros_like(rotation),
			torch.zeros_like(rotation), torch.zeros_like(rotation), torch.ones_like(rotation)
		], dim=-1).view(bsz, num_windows, n_head, 3, 3)

		# Translation transformation matrix
		t_translate = torch.stack([
			torch.ones_like(translate_x), torch.zeros_like(translate_x), translate_x,
			torch.zeros_like(translate_x), torch.ones_like(translate_x), translate_y,
			torch.zeros_like(translate_x), torch.zeros_like(translate_x), torch.ones_like(translate_x)
		], dim=-1).view(bsz, num_windows, n_head, 3, 3)

		# Perspective (or projection) transformation matrix
		t_perspective = torch.stack([
			torch.ones_like(perspective_x), torch.zeros_like(perspective_x), torch.zeros_like(perspective_x),
			torch.zeros_like(perspective_x), torch.ones_like(perspective_x), torch.zeros_like(perspective_x),
			perspective_x, perspective_y, torch.ones_like(perspective_x)
		], dim=-1).view(bsz, num_windows, n_head, 3, 3)

		transform_matrix = t_scale @ t_shear @ t_rotate @ t_translate @ t_perspective
		return transform_matrix

	def forward(self, x, h, w):
		bsz, seq_len, d_model = x.shape
		x_reshaped = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
		
		qkv_proj = self.qkv_proj(x)
		q_proj, k_proj, v_proj = qkv_proj.chunk(3, dim=-1)

		t = self.qpm(x_reshaped)
		
		num_windows_h, num_windows_w = t.shape[-2:]
		n_win = num_windows_h * num_windows_w
		
		t_per_head = rearrange(
			t, 'b (t_dim n_head) num_win_h num_win_w -> b (num_win_h num_win_w) n_head t_dim', 
			n_head=self.n_head, t_dim=9
		)
		t_unpacked = t_per_head.chunk(9, dim=-1)
		# Remove the extra dimension from chunking
		t_unpacked = [param.squeeze(-1) for param in t_unpacked]

		transform_matrix_per_head = self._get_transform_matrix(t_unpacked)

		# Generate default window coordinates
		window_h, window_w = self.window_size
		coords = torch.stack(
			torch.meshgrid(
				torch.arange(window_h, device=x.device), 
				torch.arange(window_w, device=x.device), 
				indexing='ij'
			), dim=-1
		).float()

		# Center coordinates around (0, 0)
		coords = coords - torch.tensor([window_h/2, window_w/2], device=x.device)
		coords = coords.reshape(-1, 2)
		coords = F.pad(coords, (0, 1), value=1.0) # Add Z=1 for projective transform

		# Expand coordinates for all batch, heads, and windows
		coords = coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)
		coords = coords.repeat(bsz, n_win, self.n_head, 1, 1)
		
		# Apply transformation to get new coordinates
		new_coords_hom = transform_matrix_per_head @ coords.transpose(-1, -2)
		new_coords = new_coords_hom[:, :, :, :2, :] / (new_coords_hom[:, :, :, 2:, :] + 1e-6)
		
		# [bsz, n_win, n_head, 2, num_points] -> [bsz, n_head, n_win, num_points, 2]
		new_coords_norm = new_coords.permute(0, 2, 1, 4, 3)
		
		# Calculate window centers
		center_y = torch.arange(num_windows_h, device=x.device).float() * window_h + window_h/2
		center_x = torch.arange(num_windows_w, device=x.device).float() * window_w + window_w/2
		center_grid = torch.stack(torch.meshgrid(center_y, center_x, indexing='ij'), dim=-1)
		center_grid = center_grid.view(-1, 2)  # [n_win, 2] in (y, x) order
		
		# Add window centers to get absolute coordinates
		new_coords_abs = new_coords_norm + center_grid.view(1, 1, -1, 1, 2).to(x.device)
		
		new_coords_normed = (new_coords_abs / torch.tensor([h, w], device=x.device).view(1, 1, 1, 1, 2)) * 2 - 1
		new_coords_normed = new_coords_normed.flip(-1)

		# regularization Loss - penalize coordinates outside [-1, 1]
		regularization_loss = torch.sum(
			torch.relu(torch.abs(new_coords_normed) - 1) ** 2
		)

		q = rearrange(q_proj, 'b n (h d) -> b h n d', h=self.n_head)
		k = rearrange(k_proj, 'b n (h d) -> b h n d', h=self.n_head)
		v = rearrange(v_proj, 'b n (h d) -> b h n d', h=self.n_head)
		
		k_reshaped = rearrange(k, 'b h (H W) d -> b h d H W', H=h, W=w)
		v_reshaped = rearrange(v, 'b h (H W) d -> b h d H W', H=h, W=w)
		
		new_coords_sample = rearrange(new_coords_normed, 'b h n_win num_samples c -> (b h) n_win num_samples c')
		
		k_flat = rearrange(k_reshaped, 'b h d H W -> (b h) d H W')
		v_flat = rearrange(v_reshaped, 'b h d H W -> (b h) d H W')
		
		k_sampled_list = []
		v_sampled_list = []
		
		for win_idx in range(n_win):
			k_sampled_win = F.grid_sample(
				k_flat,
				new_coords_sample[:, win_idx:win_idx+1, :, :].view(-1, window_h, window_w, 2),
				mode='bilinear', padding_mode='zeros', align_corners=True
			)
			v_sampled_win = F.grid_sample(
				v_flat,
				new_coords_sample[:, win_idx:win_idx+1, :, :].view(-1, window_h, window_w, 2),
				mode='bilinear', padding_mode='zeros', align_corners=True
			)
			k_sampled_list.append(k_sampled_win)
			v_sampled_list.append(v_sampled_win)
		
		k_sampled = torch.stack(k_sampled_list, dim=1)  # [b*h, n_win, d, window_h, window_w]
		v_sampled = torch.stack(v_sampled_list, dim=1)
		
		k_sampled = rearrange(k_sampled, '(b h) n_win d wh ww -> b h (n_win wh ww) d', b=bsz, h=self.n_head)
		v_sampled = rearrange(v_sampled, '(b h) n_win d wh ww -> b h (n_win wh ww) d', b=bsz, h=self.n_head)
		
		if self.use_qk_norm:
			q = self.q_norm(q)
			k_sampled = self.k_norm(k_sampled)

		attn_output = F.scaled_dot_product_attention(
			q, k_sampled, v_sampled,
			dropout_p=self.dropout.p if self.training else 0.0,
			is_causal=False,
		)
		
		attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
		output = self.out_proj(attn_output)

		return output, self.lambda_reg * regularization_loss
# layers/conv.py
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
# layers/cvt.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import ConvAttention
from layers.transformers import FeedForward

class CvTBlock(nn.Module):
	def __init__(self, config, kernel_size, qkv_stride, padding):
		super().__init__()

		self.d_model = config.d_model
		
		self.attn_norm = config.norm(self.d_model)
		self.attn = ConvAttention(
			dim=self.d_model, 
			n_head=config.n_head,
			kernel_size=kernel_size,
			qkv_stride=qkv_stride,
			padding=padding,
			dropout=config.dropout
		)
		self.attn_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.attn_dropout = nn.Dropout(config.dropout)

		self.ff_norm = config.norm(self.d_model)
		self.ff = FeedForward(config)

		self.ff_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.ff_dropout = nn.Dropout(config.dropout)

	def forward(self, x, h, w):
		norm_x = self.attn_norm(x)
		attn_out = self.attn(norm_x, h, w)
		attn_out = self.attn_dropout(attn_out)
		
		if self.attn_layer_scale is not None:
			x = x + self.attn_layer_scale * attn_out
		else:
			x = x + attn_out

		norm_x = self.ff_norm(x)
		ff_out = self.ff(norm_x)
		ff_out = self.ff_dropout(ff_out)
		if self.ff_layer_scale is not None:
			x = x + self.ff_layer_scale * ff_out
		else:
			x = x + ff_out
		
		return x, h, w
# layers/embeddings.py
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
# layers/norm.py
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
# layers/transformers.py
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
# layers/vit.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import MHA, QuadrangleAttention
from layers.transformers import FeedForward

class ViTBlock(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.d_model = config.d_model
		self.n_head = config.n_head
		
		self.attn_norm = config.norm(self.d_model)
		self.attn = config.attention(config)
		self.attn_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.attn_dropout = nn.Dropout(config.dropout)

		self.ff_norm = config.norm(self.d_model)
		self.ff = FeedForward(config)
		self.ff_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.ff_dropout = nn.Dropout(config.dropout)

		self.is_quadrangle_attention = isinstance(self.attn, QuadrangleAttention)

	def forward(self, x, h=None, w=None):
		cls_token, patch_tokens = x[:, :1, :], x[:, 1:, :]

		if self.is_quadrangle_attention:
			norm_patch_tokens = self.attn_norm(patch_tokens) 
			attn_out_patches, regularization_loss = self.attn(norm_patch_tokens, h, w)
			
			attn_out_cls = self.attn_norm(cls_token)
			
			attn_out = torch.cat([attn_out_cls, attn_out_patches], dim=1)
		else:
			norm_x = self.attn_norm(x)
			attn_out = self.attn(norm_x)

			regularization_loss = 0.0

		attn_out = self.attn_dropout(attn_out)
		if self.attn_layer_scale is not None:
			x = x + self.attn_layer_scale * attn_out
		else:
			x = x + attn_out

		norm_x = self.ff_norm(x)
		ff_out = self.ff(norm_x)
		ff_out = self.ff_dropout(ff_out)
		if self.ff_layer_scale is not None:
			x = x + self.ff_layer_scale * ff_out
		else:
			x = x + ff_out

		return x, regularization_loss
# main.py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import CIFAR100

from dataset import TinyImageNet
from train import Trainer
from deit_trainer import DeiTTrainer

from utils import parse_args, get_config, get_model, get_param_groups, set_seed, load_teacher_model
from schedulers import WarmupScheduler

args = parse_args()
set_seed(args.seed)

writer = SummaryWriter(f"runs/{args.model}")

print("=" * 60)
print("Training Configuration:")
print("=" * 60)
for arg in vars(args):
	print(f"{arg:20}: {getattr(args, arg)}")
print("=" * 60)

print(f"\nTraining {args.model.upper()} model")
print(f"Distillation: {'Enabled' if args.distillation else 'Disabled'}")
if args.distillation:
	print(f"Teacher model: {args.teacher_model.upper()}")
	if args.teacher_path:
		print(f"Teacher path: {args.teacher_path}")
	print(f"Alpha (distillation weight): {args.alpha}")
	print(f"Temperature: {args.tau}")

if args.device == 'auto':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
	device = torch.device(args.device)

print(f"\nUsing device: {device}")
if device.type == 'cuda':
	print(f"GPU: {torch.cuda.get_device_name(device)}")
	print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

train_transform = v2.Compose([
	v2.ToImage(),
	v2.TrivialAugmentWide(),
	v2.RandomResizedCrop(32, scale=(0.7, 1.0)),
	v2.RandomHorizontalFlip(),
	v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
	v2.ToDtype(torch.float32, scale=True),
	v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
	v2.RandomErasing(p=0.25, scale=(0.02, 0.2)),
])

test_transform = v2.Compose([
	v2.ToImage(),
	v2.Resize(32),
	v2.ToDtype(torch.float32, scale=True),
	v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

print(f"\nLoading dataset from {args.data_root}")
train_ds = CIFAR100(root='./datasets', train=True, download=True, transform=train_transform)
val_ds = CIFAR100(root='./datasets', train=False, transform=test_transform)

print(f"Train samples: {len(train_ds):,}")
print(f"Val samples: {len(val_ds):,}")
print(f"Test samples: {len(val_ds):,}")

train_dl = DataLoader(
	train_ds, shuffle=True, batch_size=args.batch_size, 
	num_workers=args.num_workers, pin_memory=True
)
val_dl = DataLoader(
	val_ds, shuffle=False, batch_size=args.batch_size, 
	num_workers=args.num_workers, pin_memory=True,
)
test_dl = DataLoader(
	val_ds, shuffle=False, batch_size=args.batch_size, 
	num_workers=args.num_workers, pin_memory=True,
)

config = get_config(args.model, args)
model = get_model(args.model, config).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: {args.model.upper()}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

param_groups = get_param_groups(model, args.weight_decay)
if args.optimizer == 'adamw':
	optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
elif args.optimizer == 'adam':
	optimizer = torch.optim.Adam(param_groups, lr=args.lr)
elif args.optimizer == 'sgd':
	optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum)

if args.label_smoothing > 0:
	criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
else:
	criterion = nn.CrossEntropyLoss()

base_scheduler = None
if args.scheduler == "cosineannealing":
	base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
elif args.scheduler == "reduceonplateau":
	base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
elif args.scheduler == "onecycle":
	base_scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer, max_lr=args.lr, steps_per_epoch=len(train_dl), epochs=args.epochs
	)

if args.warmup_steps > 0:
	scheduler = WarmupScheduler(optimizer, warmup_steps=args.warmup_steps, after_scheduler=base_scheduler)
else:
	scheduler = base_scheduler

if args.distillation:
	print(f"\nSetting up distillation training...")
	
	if not args.teacher_path:
		raise ValueError("Must provide --teacher-path for distillation")

	print(f"Loading teacher from {args.teacher_path}")
	teacher_model = load_teacher_model(
		args.teacher_path,
		args.teacher_model,
		args.num_classes,
		device
	)
	teacher_model.eval()
	teacher_params = sum(p.numel() for p in teacher_model.parameters())
	print(f"Teacher parameters: {teacher_params:,}")
	
	trainer = DeiTTrainer(
		model,
		teacher_model,
		optimizer,
		criterion,
		device,
		scheduler=scheduler,
		scheduler_type=args.scheduler,
		writer=writer,
		alpha=args.alpha,
		tau=args.tau
	)
else:
	print(f"\nSetting up standard training...")
	trainer = Trainer(
		model,
		optimizer,
		criterion,
		device,
		scheduler=scheduler,
		scheduler_type=args.scheduler,
		writer=writer,
	)

if args.resume:
	print(f"\nResuming from checkpoint: {args.resume}")
	checkpoint = torch.load(args.resume, map_location=device)
	trainer.load_checkpoint(checkpoint)

print(f"\nStarting training for {args.epochs} epochs...")
trainer.train(
	args.epochs, train_dl, val_dl,
	save_path=f"{args.save_path}/{args.model}.pt",
	config=config,
	args=vars(args)
)

print(f"\n{'='*60}")
print("FINAL EVALUATION")
print(f"{'='*60}")
test_loss, test_acc = trainer.run_one_epoch(test_dl, state='eval')
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.1%}")
print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
print(f"{'='*60}")
# models/coatnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from layers.vit import ViTBlock
from layers.norm import DyT, RMSNorm
from layers.conv import MBConv

class CoAtNet(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.activation = activation.lower()
		if self.activation == "gelu":
			self.act_fn = nn.GELU
		else:
			self.act_fn = nn.SiLU

		# Convolutional Stem
		self.s0 = nn.Sequential(
			nn.Conv2d(config.chw[0], config.dims[0], kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(dims[0]),
			self.act_fn(),
		)
		
		# Convolutional Stages (MBConv)
		self.s1 = self._make_conv_stage(config.dims[0], config.dims[1], config.depths[0], stride=2)
		self.s2 = self._make_conv_stage(config.dims[1], config.dims[2], config.depths[1], stride=2)

		# Transformer Stages
		self.s3_pe = nn.Conv2d(config.dims[2], config.dims[3], 1) # Patch embedding
		self.s3 = self._make_transformer_stage(config.dims[3], config.depths[2], config.num_heads, cofnig.norm, config.activation)

		self.s4_pe = nn.Conv2d(config.dims[3], config.dims[4], 2, 2) # Patch merging
		self.s4 = self._make_transformer_stage(config.dims[4], config.depths[3], config.num_heads, cofnig.norm, config.activation)

		self.head = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(config.dims[-1], config.num_classes)
		)

	def _make_conv_stage(self, in_chans, out_chans, depth, stride):
		layers = [MBConv(in_chans, out_chans, stride=stride)]

		for _ in range(depth - 1):
			layers.append(MBConv(out_chans, out_chans, stride=1))

		return nn.Sequential(*layers)

	def _make_transformer_stage(self, dim, depth, num_heads, norm=RMSNorm, activation="silu"):
		class MHAConfig:
			d_model, n_head, dropout, use_qk_norm, use_layer_scale = dim, num_heads, 0.1, False, False
			norm, activation, d_ff = norm, activation, dim * 4
		
		blocks = []
		for _ in range(depth):
			blocks.append(ViTBlock(MHAConfig()))

		return nn.ModuleList(blocks)

	def forward(self, x):
		B = x.shape[0]

		# Conv stages
		x = self.s0(x)
		x = self.s1(x)
		x = self.s2(x)
		
		# Transformer stage 1
		x = self.s3_pe(x)
		_, C, H, W = x.shape
		x = rearrange(x, 'b c h w -> b (h w) c')
		for blk in self.s3:
			x = blk(x)
		x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

		# Transformer stage 2
		x = self.s4_pe(x)
		_, C, H, W = x.shape
		x = rearrange(x, 'b c h w -> b (h w) c')
		for blk in self.s4:
			x = blk(x)
		x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
		
		# Head
		x = self.head(x)

		return x
# models/cvt.py
import torch
import torch.nn as nn
from einops import rearrange

from layers.embeddings import ConvEmbedding
from layers.cvt import CvTBlock

class CvT(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		
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
		
		return self.head(x)
# models/resnet.py
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
import torch.hub

# Using the code from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
def _weights_init(m):
	classname = m.__class__.__name__
	if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
		torch.nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
	def __init__(self, lambd):
		super(LambdaLayer, self).__init__()
		self.lambd = lambd

	def forward(self, x):
		return self.lambd(x)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, option='A'):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != planes:
			if option == 'A':
				"""
				For CIFAR10 ResNet paper uses option A.
				"""
				self.shortcut = LambdaLayer(lambda x:
											F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
			elif option == 'B':
				self.shortcut = nn.Sequential(
					 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
					 nn.BatchNorm2d(self.expansion * planes)
				)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(ResNet, self).__init__()
		self.in_planes = 16

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
		self.fc = nn.Linear(64, num_classes)

		self.apply(_weights_init)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion

		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = F.avg_pool2d(out, out.size()[3])
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		return out

def resnet_cifar(depth, num_classes=100):
	if depth == 20:
		return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)
	elif depth == 32:
		return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)
	else:
		raise ValueError('Invalid ResNet depth for CIFAR: {}'.format(depth))

def get_model(model_name: str, num_classes=100):
	if model_name == "resnet20":
		model = resnet_cifar(20, num_classes=10)
	elif model_name == "resnet32":
		model = resnet_cifar(32, num_classes=10)
	else:
		raise ValueError(f"Unknown model {model_name}")
		
	local_path = f"checkpoints/{model_name}_cifar10.pth"
	state_dict = torch.load(local_path)
	model.load_state_dict(state_dict)

	model.fc = nn.Linear(model.fc.in_features, num_classes)
	
	return model
# models/utils.py
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
# models/vit.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.utils import get_positional_embeddings
from layers.embeddings import patchify
from layers.vit import ViTBlock
from layers.attention import QuadrangleAttention

class ViT(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.chw = config.chw # (C, H, W)

		self.d_model = config.d_model
		self.n_patch = config.n_patch
		self.n_block = config.n_block

		assert self.chw[1] % self.n_patch == 0, "Input shape not entirely divisible by number of patch"
		assert self.chw[2] % self.n_patch == 0, "Input shape not entirely divisible by number of patch"

		self.patch_size = (self.chw[1] // self.n_patch, self.chw[2] // self.n_patch)
		self.h_grid = self.chw[1] // self.patch_size[0]
		self.w_grid = self.chw[2] // self.patch_size[1]

		self.input_d = self.chw[0] * self.patch_size[0] * self.patch_size[1]
		self.linear_mapper = nn.Linear(self.input_d, self.d_model)

		self.class_token = nn.Parameter(torch.rand(1, self.d_model))	

		# self.register_buffer('pos_embed', get_positional_embeddings(n_patch ** 2 + 1, d_model), persistent=False)
		self.pos_embed = nn.Parameter(torch.randn(1, self.n_patch ** 2 + 1, self.d_model))

		self.blocks = nn.ModuleList([ViTBlock(config) for _ in range(self.n_block)])

		self.mlp = nn.Linear(self.d_model, config.out_d)

		self.is_quadrangle_attention = config.attention == QuadrangleAttention

	def forward(self, images):
		bsz = images.size(0)

		patch = patchify(images, self.n_patch)
		tokens = self.linear_mapper(patch)

		tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

		pos_embed = self.pos_embed.repeat(bsz, 1, 1)
		out = tokens + pos_embed

		total_reg_loss = 0.0
		for block in self.blocks:
			if self.is_quadrangle_attention:
				out, regularization_loss = block(out, self.h_grid, self.w_grid)
				total_reg_loss += regularization_loss
			else:
				out, _ = block(out)

		# [CLS] token
		out = out[:, 0]

		output = self.mlp(out)
		if self.is_quadrangle_attention:
			return output, total_reg_loss
		else:
			return output
# plot.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os

def load_tensorboard_logs(path: str, tag: str) -> pd.DataFrame:
	"""
	Loads scalar data from a TensorBoard log file.
	"""
	ea = event_accumulator.EventAccumulator(path,
		size_guidance={event_accumulator.SCALARS: 0})
	ea.Reload()

	if tag not in ea.Tags()['scalars']:
		print(f"Warning: Tag '{tag}' not found in {path}. Skipping.")
		return pd.DataFrame()

	scalar_events = ea.Scalars(tag)
	steps = [event.step for event in scalar_events]
	values = [event.value for event in scalar_events]

	return pd.DataFrame({'Epoch': steps, 'Value': values})

base_log_dir = './runs'
runs = {
	'deit': os.path.join(base_log_dir, 'deit'),
	'resnet32_finetune': os.path.join(base_log_dir, 'resnet32_finetune'),
	'resnet32_transfer': os.path.join(base_log_dir, 'resnet32_transfer'),
	'vit': os.path.join(base_log_dir, 'vit')
}
tags = {
	'Accuracy/train': ('Accuracy', 'Train'),
	'Accuracy/val': ('Accuracy', 'Validation'),
	'Loss/train': ('Loss', 'Train'),
	'Loss/val': ('Loss', 'Validation')
}

all_data_dfs = []
for model_name, path in runs.items():
	if not os.path.exists(path):
		print(f"Warning: Directory not found at {path}. Skipping model '{model_name}'.")
		continue
	for tag_name, (metric, split) in tags.items():
		df_run = load_tensorboard_logs(path, tag_name)
		if not df_run.empty:
			df_run['Model'] = model_name
			df_run['Metric'] = metric
			df_run['Split'] = split
			all_data_dfs.append(df_run)

if not all_data_dfs:
	print("Error: No data was loaded. Please check your paths and tag names.")
else:
	temp_df = pd.concat(all_data_dfs)
	
	if 'resnet32_transfer' in temp_df['Model'].unique():
		transfer_max_step = temp_df[temp_df['Model'] == 'resnet32_transfer']['Epoch'].max()
		print(f"Offset determined from resnet32_transfer's final step: {transfer_max_step}")

		for i, df in enumerate(all_data_dfs):
			if not df.empty and df['Model'].iloc[0] == 'resnet32_finetune':
				df['Epoch'] += transfer_max_step
				all_data_dfs[i] = df 
	else:
		print("Warning: 'resnet32_transfer' data not found. Cannot apply offset.")


	df_combined = pd.concat(all_data_dfs, ignore_index=True)

	sns.set_theme(style="darkgrid")

	g = sns.relplot(
		data=df_combined,
		x="Epoch",
		y="Value",
		hue="Model",
		col="Split",
		row="Metric",
		kind="line",
		height=4,
		aspect=1.2,
		linewidth=1.5,
		facet_kws={'sharey': False, 'sharex': True}
	)


	g.fig.suptitle("Model Metrics", y=1.03, fontsize=16)
	g.set_titles("{row_name} / {col_name}", size=12)
	g.set_axis_labels("Epoch", "Value", size=12)
	sns.move_legend(g, "lower center", bbox_to_anchor=(.5, -0.08), ncol=4, title=None, frameon=False)
	
	plt.savefig("model_comparison_plot.png", dpi=300, bbox_inches='tight')
	plt.show()
# schedulers.py
from torch.optim.lr_scheduler import _LRScheduler

class WarmupScheduler(_LRScheduler):
	def __init__(self, optimizer, warmup_steps, after_scheduler=None, last_epoch=-1):
		self.warmup_steps = warmup_steps
		self.after_scheduler = after_scheduler
		self.finished_warmup = False
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		if self.last_epoch < self.warmup_steps:
			# Linear warmup
			return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]

		if self.after_scheduler:
			if not self.finished_warmup:
				# reset after warmup
				self.after_scheduler.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
				self.finished_warmup = True

			return self.after_scheduler.get_last_lr()

		return [group["lr"] for group in self.optimizer.param_groups]

	def step(self, epoch=None, metrics=None):
		if self.last_epoch < self.warmup_steps:
			return super(WarmupScheduler, self).step(epoch)
		if self.after_scheduler:
			if isinstance(self.after_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
				self.after_scheduler.step(metrics)
			else:
				if epoch is None:
					self.after_scheduler.step(None)
				else:
					self.after_scheduler.step(epoch - self.warmup_steps)
# tmp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.norm import DyT, RMSNorm

# from dataclasses import dataclass

class ViTConfig:
	d_model=512
	d_ff=2048

	n_head=16
	n_block=4
	# n_patch=16
	n_patch=8
	out_d = 200 # TinyImageNet

	norm=RMSNorm
	activation="swiglu"

	dropout=0.1

	use_layer_scale=True
	use_qk_norm=False

	# chw=(3, 224, 224)
	chw=(3, 64, 64)

	kernel_size=16
	stride=16

class CoAtNetConfig:
	dims=[64, 96, 192, 384, 768]
	depths=[2, 2, 3, 5, 2]
	num_heads=32

	# chw=(3, 224, 224)
	chw=(3, 64, 64)

	activation="silu"
	norm=RMSNorm

	dropout=0.1

	out_d = 200 # TinyImageNet

class CvTConfig:
	# Based on CvT-13, adjusted for TinyImageNet
	chw = (3, 64, 64)
	out_d = 200 # TinyImageNet
	
	# Stage 1
	s1_emb_kernel = 7
	s1_emb_stride = 4
	s1_emb_pad = 2
	s1_emb_dim = 64
	s1_depth = 1
	s1_heads = 1
	s1_mlp_ratio = 4.0
	s1_qkv_kernel = 3
	s1_qkv_stride = 1
	s1_qkv_pad = 1

	# Stage 2
	s2_emb_kernel = 3
	s2_emb_stride = 2
	s2_emb_pad = 1
	s2_emb_dim = 192
	s2_depth = 2
	s2_heads = 3
	s2_mlp_ratio = 4.0
	s2_qkv_kernel = 3
	s2_qkv_stride = 2
	s2_qkv_pad = 1

	# Stage 3
	s3_emb_kernel = 3
	s3_emb_stride = 2
	s3_emb_pad = 1
	s3_emb_dim = 384
	s3_depth = 10
	s3_heads = 6
	s3_mlp_ratio = 4.0
	s3_qkv_kernel = 3
	s3_qkv_stride = 2
	s3_qkv_pad = 1

	dropout=0.1
	norm=RMSNorm
	activation="gelu"
	use_layer_scale=False
import os
import requests
import zipfile
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from tqdm import tqdm

class TinyImageNet(Dataset):
	url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
	filename = 'tiny-imagenet-200.zip'
	folder_name = 'tiny-imagenet-200'
	
	def __init__(self, root, split='train', transform=None, download=False, test_size=0.1, random_state=0):
		self.root = os.path.expanduser(root)
		self.split = split
		self.transform = transform
		self.test_size = test_size
		self.random_state = random_state
		
		if download:
			self.download()
			
		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')
		
		self.data = []
		self.targets = []
		
		self.class_to_idx = {}
		with open(os.path.join(self.root, self.folder_name, 'wnids.txt'), 'r') as f:
			for i, line in enumerate(f):
				self.class_to_idx[line.strip()] = i
		
		self._load_data()
	
	def _load_data(self):
		"""Load and split the data based on the specified split"""
		if self.split == 'train':
			self._load_from_folder('train')
		elif self.split == 'val':
			self._load_from_folder('val')
		elif self.split == 'test':
			self._load_from_folder('test')
		else:
			raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")

	def _load_from_folder(self, folder):
		"""Load images and labels directly from the specified subfolder"""
		if folder == 'train':
			split_dir = os.path.join(self.root, self.folder_name, folder)
			for class_folder in os.listdir(split_dir):
				class_idx = self.class_to_idx[class_folder]
				class_dir = os.path.join(split_dir, class_folder, 'images')
				for img_name in os.listdir(class_dir):
					img_path = os.path.join(class_dir, img_name)
					self.data.append(img_path)
					self.targets.append(class_idx)
		elif folder == 'val':
			split_dir = os.path.join(self.root, self.folder_name, folder)
			with open(os.path.join(split_dir, 'val_annotations.txt'), 'r') as f:
				for line in f:
					parts = line.strip().split('\t')
					img_name, class_id = parts[0], parts[1]
					img_path = os.path.join(split_dir, 'images', img_name)
					self.data.append(img_path)
					self.targets.append(self.class_to_idx[class_id])
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		img_path = self.data[idx]
		target = self.targets[idx]
		
		with open(img_path, 'rb') as f:
			img = Image.open(f).convert('RGB')
		
		if self.transform:
			img = self.transform(img)
			
		return img, target
	
	def _check_integrity(self):
		return os.path.isdir(os.path.join(self.root, self.folder_name))
	
	def download(self):
		if self._check_integrity():
			print('Files already downloaded and verified')
			return
		
		download_url(self.url, self.root, self.filename)
		with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zip_ref:
			zip_ref.extractall(self.root)
	
	def get_split_info(self):
		"""Return information about the dataset splits"""
		class_counts = {}
		for target in self.targets:
			class_counts[target] = class_counts.get(target, 0) + 1
			
		return {
			'split': self.split,
			'total_samples': len(self.data),
			'num_classes': len(self.class_to_idx),
			'samples_per_class': class_counts
		}

import torch
import torch.nn.functional as F
from train import Trainer

import torch
import torch.nn.functional as F

class DeiTTrainer(Trainer):
	def __init__(self, model, teacher_model, optimizer, criterion, device, alpha=0.5, tau=1.0):
		super().__init__(model, optimizer, criterion, device)
		self.teacher_model = teacher_model
		self.teacher_model.eval()

		for param in self.teacher_model.parameters():
			param.requires_grad = False

		self.alpha = alpha
		self.tau = tau
	
	def compute_loss(self, y_hat, y, x=None):
		if isinstance(y_hat, tuple):
			student_pred, student_distill = y_hat
		else:
			student_pred = y_hat
			student_distill = y_hat
		
		base_loss = self.criterion(student_pred, y)
		
		if x is not None:
			with torch.no_grad():
				teacher_pred = self.teacher_model(x)

				if isinstance(teacher_pred, tuple):
					teacher_pred = teacher_pred[0]
		else:
			return base_loss
		
		distill_loss = F.kl_div(
			F.log_softmax(student_distill / self.tau, dim=1),
			F.softmax(teacher_pred / self.tau, dim=1),
			reduction='batchmean'
		) * (self.tau * self.tau)
		
		return (1 - self.alpha) * base_loss + self.alpha * distill_loss
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
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import ConvAttention
from layers.transformers import FeedForward

class CvTBlock(nn.Module):
	def __init__(self, config, kernel_size, qkv_stride, padding):
		super().__init__()

		self.d_model = config.d_model
		
		self.attn_norm = config.norm(self.d_model)
		self.attn = ConvAttention(
			dim=self.d_model, 
			n_head=config.n_head,
			kernel_size=kernel_size,
			qkv_stride=qkv_stride,
			padding=padding,
			dropout=config.dropout
		)
		self.attn_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.attn_dropout = nn.Dropout(config.dropout)

		self.ff_norm = config.norm(self.d_model)
		self.ff = FeedForward(config)

		self.ff_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.ff_dropout = nn.Dropout(config.dropout)

	def forward(self, x, h, w):
		norm_x = self.attn_norm(x)
		attn_out = self.attn(norm_x, h, w)
		attn_out = self.attn_dropout(attn_out)
		
		if self.attn_layer_scale is not None:
			x = x + self.attn_layer_scale * attn_out
		else:
			x = x + attn_out

		norm_x = self.ff_norm(x)
		ff_out = self.ff(norm_x)
		ff_out = self.ff_dropout(ff_out)
		if self.ff_layer_scale is not None:
			x = x + self.ff_layer_scale * ff_out
		else:
			x = x + ff_out
		
		return x, h, w
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

		# TODO: checking out parallel Linear from llama
		# fc_in can be column parallel
		# fc_out can be row parallel

		if self.activation in ('swiglu', 'geglu'):
			# default scaling down by 2/3 since normal 
			# d_ff is 4xd_model
			# Should be ~2.667 scalling now
			# based on Llama SwiGLU FeedForward
			# https://github.com/meta-llama/llama
			d_ff = int(2 * config.d_ff // 3)
			self.fc_in = nn.Linear(config.d_model, d_ff * 2)
		else:
			self.fc_in = nn.Linear(config.d_model, config.d_ff)

		self.fc_out = nn.Linear(config.d_ff, config.d_model) # can be row parallel

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
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import MHA
from layers.transformers import FeedForward

class ViTBlock(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.d_model = config.d_model
		self.n_head = config.n_head
		
		self.attn_norm = config.norm(self.d_model)
		self.attn = MHA(config)
		self.attn_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.attn_dropout = nn.Dropout(config.dropout)

		self.ff_norm = config.norm(self.d_model)
		self.ff = FeedForward(config)
		self.ff_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.ff_dropout = nn.Dropout(config.dropout)

	def forward(self, x):
		norm_x = self.attn_norm(x)
		attn_out = self.attn(norm_x)
		attn_out = self.attn_dropout(attn_out)
		if self.attn_layer_scale is not None:
			x = x + self.attn_layer_scale * attn_out
		else:
			x = x + attn_out

		norm_x = self.ff_norm(x)
		ff_out = self.ff(norm_x)
		ff_out = self.ff_dropout(ff_out)
		if self.ff_layer_scale is not None:
			x = x + self.ff_layer_scale * ff_out
		else:
			x = x + ff_out

		return x
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.models import resnet18, ResNet18_Weights

from dataset import TinyImageNet
from train import Trainer
from deit_trainer import DeiTTrainer

from utils import parse_args, get_config, get_model, get_param_groups, set_seed
from schedulers import WarmupScheduler

def main():
	args = parse_args()
	set_seed(0)

	print("=" * 60)
	print("Training Configuration:")
	print("=" * 60)
	for arg in vars(args):
		print(f"{arg:20}: {getattr(args, arg)}")
	print("=" * 60)

	print(f"\nTraining {args.model.upper()} model")
	print(f"Distillation: {'Enabled' if args.distillation else 'Disabled'}")
	if args.distillation:
		print(f"Teacher model: {args.teacher_model.upper()}")
		if args.teacher_path:
			print(f"Teacher path: {args.teacher_path}")
		print(f"Alpha (distillation weight): {args.alpha}")
		print(f"Temperature: {args.tau}")

	if args.device == 'auto':
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	else:
		device = torch.device(args.device)

	print(f"\nUsing device: {device}")
	if device.type == 'cuda':
		print(f"GPU: {torch.cuda.get_device_name(device)}")
		print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

	train_transform = v2.Compose([
		v2.ToImage(),
		v2.TrivialAugmentWide(),
		v2.RandomResizedCrop(64, scale=(0.7, 1.0)),
		v2.RandomHorizontalFlip(),
		v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		v2.RandomErasing(p=0.25, scale=(0.02, 0.2)),
	])

	test_transform = v2.Compose([
		v2.ToImage(),
		v2.Resize(72),
		v2.CenterCrop(64),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	print(f"\nLoading dataset from {args.data_root}")
	train_ds = TinyImageNet(root='./datasets', split='train', download=True, transform=train_transform)
	val_ds = TinyImageNet(root='./datasets', split='val', transform=test_transform)
	test_ds = TinyImageNet(root='./datasets', split='test', transform=test_transform)

	print(f"Train samples: {len(train_ds):,}")
	print(f"Val samples: {len(val_ds):,}")
	print(f"Test samples: {len(test_ds):,}")

	train_dl = DataLoader(
		train_ds, shuffle=True, batch_size=args.batch_size, 
		num_workers=args.num_workers, pin_memory=True
	)
	val_dl = DataLoader(
		val_ds, shuffle=False, batch_size=args.batch_size, 
		num_workers=args.num_workers, pin_memory=True,
	)
	test_dl = DataLoader(
		test_ds, shuffle=False, batch_size=args.batch_size, 
		num_workers=args.num_workers, pin_memory=True,
	)

	config = get_config(args.model, args)
	model = get_model(args.model, config).to(device)

	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"\nModel: {args.model.upper()}")
	print(f"Total parameters: {total_params:,}")
	print(f"Trainable parameters: {trainable_params:,}")

	param_groups = get_param_groups(model, args.weight_decay)
	if args.optimizer == 'adamw':
		optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
	elif args.optimizer == 'adam':
		optimizer = torch.optim.Adam(param_groups, lr=args.lr)
	elif args.optimizer == 'sgd':
		optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum)

	if args.label_smoothing > 0:
		criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
	else:
		criterion = nn.CrossEntropyLoss()

	base_scheduler = None
	if args.scheduler == "cosineannealing":
		base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	elif args.scheduler == "reduceonplateau":
		base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
	elif args.scheduler == "onecycle":
		base_scheduler = torch.optim.lr_scheduler.OneCycleLR(
			optimizer, max_lr=args.lr, steps_per_epoch=len(train_dl), epochs=args.epochs
		)

	if args.warmup_steps > 0:
		scheduler = WarmupScheduler(optimizer, warmup_steps=args.warmup_steps, after_scheduler=base_scheduler)
	else:
		scheduler = base_scheduler

	if args.resume:
		print(f"\nResuming from checkpoint: {args.resume}")
		checkpoint = torch.load(args.resume, map_location=device)
		trainer.load_checkpoint(checkpoint)
	
	if args.distillation:
		print(f"\nSetting up distillation training...")
		
		if args.teacher_path:
			print(f"Loading teacher from {args.teacher_path}")
			teacher_model = load_teacher_model(
				args.teacher_path, 
				args.teacher_model, 
				args.num_classes, 
				device
			)
		else:
			print("Creating pretrained teacher model")
			teacher_config = get_config(args.teacher_model, args)
			teacher_model = create_model(args.teacher_model, teacher_config).to(device)
			
			if args.teacher_model == 'resnet18':
				teacher_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
				teacher_model.fc = nn.Linear(teacher_model.fc.in_features, args.num_classes)
				teacher_model = teacher_model.to(device)

				nn.init.xavier_uniform_(teacher_model.fc.weight)
				nn.init.zeros_(teacher_model.fc.bias)
		
		teacher_model.eval()
		teacher_params = sum(p.numel() for p in teacher_model.parameters())
		print(f"Teacher parameters: {teacher_params:,}")
		
		trainer = DeiTTrainer(
			model,
			teacher_model,
			optimizer,
			criterion,
			device,
			scheduler=scheduler,
			scheduler_type=args.scheduler,
			alpha=args.alpha,
			tau=args.tau
		)
	else:
		print(f"\nSetting up standard training...")
		trainer = Trainer(
			model,
			optimizer,
			criterion,
			device,
			scheduler=scheduler,
			scheduler_type=args.scheduler,
		)
	
	print(f"\nStarting training for {args.epochs} epochs...")
	trainer.train(
		args.epochs, train_dl, val_dl,
		save_path=f"{args.save_path}/{args.model}",
		config=config,
		args=vars(args)
	)
	
	print(f"\n{'='*60}")
	print("FINAL EVALUATION")
	print(f"{'='*60}")
	test_loss, test_acc = trainer.run_one_epoch(test_dl, state='eval')
	print(f"Test Loss: {test_loss:.4f}")
	print(f"Test Accuracy: {test_acc:.1%}")
	print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
	print(f"{'='*60}")

if __name__ == "__main__":
	main()
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from layers.vit import ViTBlock
from layers.norm import DyT, RMSNorm
from layers.conv import MBConv

class CoAtNet(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.activation = activation.lower()
		if self.activation == "gelu":
			self.act_fn = nn.GELU
		else:
			self.act_fn = nn.SiLU

		# Convolutional Stem
		self.s0 = nn.Sequential(
			nn.Conv2d(config.chw[0], config.dims[0], kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(dims[0]),
			self.act_fn(),
		)
		
		# Convolutional Stages (MBConv)
		self.s1 = self._make_conv_stage(config.dims[0], config.dims[1], config.depths[0], stride=2)
		self.s2 = self._make_conv_stage(config.dims[1], config.dims[2], config.depths[1], stride=2)

		# Transformer Stages
		self.s3_pe = nn.Conv2d(config.dims[2], config.dims[3], 1) # Patch embedding
		self.s3 = self._make_transformer_stage(config.dims[3], config.depths[2], config.num_heads, cofnig.norm, config.activation)

		self.s4_pe = nn.Conv2d(config.dims[3], config.dims[4], 2, 2) # Patch merging
		self.s4 = self._make_transformer_stage(config.dims[4], config.depths[3], config.num_heads, cofnig.norm, config.activation)

		self.head = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(config.dims[-1], config.num_classes)
		)

	def _make_conv_stage(self, in_chans, out_chans, depth, stride):
		layers = [MBConv(in_chans, out_chans, stride=stride)]

		for _ in range(depth - 1):
			layers.append(MBConv(out_chans, out_chans, stride=1))

		return nn.Sequential(*layers)

	def _make_transformer_stage(self, dim, depth, num_heads, norm=RMSNorm, activation="silu"):
		class MHAConfig:
			d_model, n_head, dropout, use_qk_norm, use_layer_scale = dim, num_heads, 0.1, False, False
			norm, activation, d_ff = norm, activation, dim * 4
		
		blocks = []
		for _ in range(depth):
			blocks.append(ViTBlock(MHAConfig()))

		return nn.ModuleList(blocks)

	def forward(self, x):
		B = x.shape[0]

		# Conv stages
		x = self.s0(x)
		x = self.s1(x)
		x = self.s2(x)
		
		# Transformer stage 1
		x = self.s3_pe(x)
		_, C, H, W = x.shape
		x = rearrange(x, 'b c h w -> b (h w) c')
		for blk in self.s3:
			x = blk(x)
		x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

		# Transformer stage 2
		x = self.s4_pe(x)
		_, C, H, W = x.shape
		x = rearrange(x, 'b c h w -> b (h w) c')
		for blk in self.s4:
			x = blk(x)
		x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
		
		# Head
		x = self.head(x)

		return x
import torch
import torch.nn as nn
from einops import rearrange

from layers.embeddings import ConvEmbedding
from layers.cvt import CvTBlock

class CvT(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		
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
		
		return self.head(x)
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
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.utils import get_positional_embeddings
from layers.embeddings import patchify
from layers.vit import ViTBlock

class ViT(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.chw = config.chw # (C, H, W)

		self.d_model = config.d_model
		self.n_patch = config.n_patch
		self.n_block = config.n_block

		print(self.chw)
		assert self.chw[1] % self.n_patch == 0, "Input shape not entirely divisible by number of patch"
		assert self.chw[2] % self.n_patch == 0, "Input shape not entirely divisible by number of patch"

		self.patch_size = (self.chw[1] // self.n_patch, self.chw[2] // self.n_patch)

		self.input_d = self.chw[0] * self.patch_size[0] * self.patch_size[1]
		self.linear_mapper = nn.Linear(self.input_d, self.d_model)

		self.class_token = nn.Parameter(torch.rand(1, self.d_model))	

		# self.register_buffer('pos_embed', get_positional_embeddings(n_patch ** 2 + 1, d_model), persistent=False)
		self.pos_embed = nn.Parameter(torch.randn(1, self.n_patch ** 2 + 1, self.d_model))

		self.blocks = nn.ModuleList([ViTBlock(config) for _ in range(self.n_block)])

		self.mlp = nn.Linear(self.d_model, config.out_d)

	def forward(self, images):
		bsz = images.size(0)

		patch = patchify(images, self.n_patch)
		tokens = self.linear_mapper(patch)

		tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

		pos_embed = self.pos_embed.repeat(bsz, 1, 1)
		out = tokens + pos_embed

		for block in self.blocks:
			out = block(out)

		# [CLS] token
		out = out[:, 0]

		return self.mlp(out)
from torch.optim.lr_scheduler import _LRScheduler

class WarmupScheduler(_LRScheduler):
	def __init__(self, optimizer, warmup_steps, after_scheduler=None, last_epoch=-1):
		self.warmup_steps = warmup_steps
		self.after_scheduler = after_scheduler
		self.finished_warmup = False
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		if self.last_epoch < self.warmup_steps:
			# Linear warmup
			return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]

		if self.after_scheduler:
			if not self.finished_warmup:
				# reset after warmup
				self.after_scheduler.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
				self.finished_warmup = True

			return self.after_scheduler.get_last_lr()

		return [group["lr"] for group in self.optimizer.param_groups]

	def step(self, epoch=None, metrics=None):
		if self.last_epoch < self.warmup_steps:
			return super(WarmupScheduler, self).step(epoch)
		if self.after_scheduler:
			if isinstance(self.after_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
				self.after_scheduler.step(metrics)
			else:
				if epoch is None:
					self.after_scheduler.step(None)
				else:
					self.after_scheduler.step(epoch - self.warmup_steps)
from tqdm import tqdm
import torch

class Trainer:
	def __init__(self, model, optimizer, criterion, device, scheduler=None, scheduler_type=None, patience=10, min_delta=1e-4):
		self.model = model
		self.optimizer = optimizer
		self.criterion = criterion
		self.device = device

		self.scheduler = scheduler
		self.scheduler_type = scheduler_type

		# Early stopping
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.early_stop = False

		# Tracking
		self.start_epoch = 0
		self.best_val_loss = float('inf')
	
	def run_one_epoch(self, dataloader, state='train'):
		"""
		Run one epoch of training or evaluation
		"""
		is_training = (state == 'train')
		
		if is_training:
			self.model.train()
		else:
			self.model.eval()
		
		total_loss = 0.0
		correct = 0
		total = 0
		
		with torch.set_grad_enabled(is_training):
			for batch in dataloader:
				x, y = batch
				x, y = x.to(self.device), y.to(self.device)
				
				if is_training:
					self.optimizer.zero_grad()
				
				y_hat = self.model(x)
				loss = self.compute_loss(y_hat, y, x)
				
				if is_training:
					loss.backward()
					self.optimizer.step()
				
				total_loss += loss.detach().cpu().item()
				
				pred_for_acc = y_hat
				if isinstance(y_hat, tuple):
					pred_for_acc = y_hat[0]
				
				correct += torch.sum(torch.argmax(pred_for_acc, dim=1) == y).detach().cpu().item()
				total += len(x)
		
		avg_loss = total_loss / len(dataloader)
		accuracy = correct / total
		
		return avg_loss, accuracy
	
	def compute_loss(self, y_hat, y, x=None):
		return self.criterion(y_hat, y)
	
	def train(self, n_epochs, train_dl, val_dl, save_path=None, config=None, args=None):
		with tqdm(range(self.start_epoch, n_epochs), desc="Training Progress") as pbar:
			for epoch in pbar:
				train_loss, train_acc = self.run_one_epoch(train_dl, state='train')
				val_loss, val_acc = self.run_one_epoch(val_dl, state='eval')
				
				if val_loss < self.best_val_loss:
					self.best_val_loss = val_loss

				if self.scheduler:
					if self.scheduler_type == "reduceonplateau":
						self.scheduler.step(val_loss)
					else:
						self.scheduler.step()

				if save_path:
					self.save_checkpoint(
						save_path,
						epoch=epoch,
						config=config,
						args=args,
						best=(val_loss < self.best_val_loss)
					)

				if val_loss + self.min_delta < self.best_val_loss:
					self.best_val_loss = val_loss
					self.counter = 0
				else:
					self.counter += 1
					if self.counter >= self.patience:
						print(f"\nEarly stopping triggered at epoch {epoch+1}")
						self.early_stop = True
						break
				
				pbar.set_postfix({
					'epoch': epoch + 1,
					'train_loss': f'{train_loss:.4f}',
					'train_acc': f'{train_acc:.3f}',
					'val_loss': f'{val_loss:.4f}',
					'val_acc': f'{val_acc:.3f}',
					'best_val_loss': f'{self.best_val_loss:.4f}'
				})

	def save_checkpoint(self, path, epoch, config=None, args=None, best=False):
		"""Save training checkpoint."""

		checkpoint = {
			'epoch': epoch + 1,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'best_val_loss': self.best_val_loss,
		}
		if self.scheduler:
			checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
		if config:
			checkpoint['config'] = config
		if args:
			checkpoint['args'] = args

		torch.save(checkpoint, path)

		if best:
			best_path = path.replace(".pt", "_best.pt")
			torch.save(checkpoint, best_path)
			print(f"Best model updated and saved to {best_path}")

	def load_checkpoint(self, checkpoint):
		"""Load model/optimizer state for resume."""

		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		if self.scheduler and 'scheduler_state_dict' in checkpoint:
			self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		self.start_epoch = checkpoint.get('epoch', 0)
		self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

		print(f"Resumed from epoch {self.start_epoch}, best val loss {self.best_val_loss:.4f}")
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights
from models.cvt import CvT
from models.vit import ViT
from models.coatnet import CoAtNet
from config import ViTConfig, CoAtNetConfig, CvTConfig

import random
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser(description="Train CvT / ResNet / DeiT on TinyImageNet")

	# General
	parser.add_argument("--data-root", type=str, default="./tiny-imagenet-200", help="Path to TinyImageNet dataset")
	parser.add_argument("--download", action="store_true", help="Download dataset if not found")
	parser.add_argument("--save-path", type=str, default="checkpoints", help="Path to save the trained model checkpoint")
	parser.add_argument("--resume", type=str, default="", help="Path to resume from checkpoint")
	parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', or 'cuda'")

	# Model
	parser.add_argument("--model", type=str, default="vit", choices=["vit", "cvt", "resnet18", "deit"], help="Model type")
	parser.add_argument("--num-classes", type=int, default=200, help="Number of classes")

	# Optimizer
	parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
	parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
	parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")

	# Scheduler
	parser.add_argument(
		"--scheduler", type=str, default="cosineannealing",
		choices=["cosineannealing", "reduceonplateau", "onecycle"],
		help="Learning rate scheduler",
  )
	parser.add_argument(
		"--warmup-steps", type=int, default=0,
		help="Number of warmup steps before main scheduler kicks in"
	)

	# Training
	parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
	parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
	parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for CE loss")

	# Distillation
	parser.add_argument("--distillation", action="store_true", help="Enable knowledge distillation")
	parser.add_argument("--teacher-model", type=str, default="resnet18", choices=["resnet18", "cvt", "deit"])
	parser.add_argument("--teacher-path", type=str, default="", help="Path to teacher checkpoint")
	parser.add_argument("--alpha", type=float, default=0.5, help="Weight for distillation loss")
	parser.add_argument("--tau", type=float, default=1.0, help="Temperature for distillation")

	return parser.parse_args()

def get_config(model_name: str, args=None):
	if model_name.lower() == "cvt":
		return CvTConfig()
	elif model_name.lower() == "vit":
		return ViTConfig()
	elif model_name.lower() == "coatnet":
		return CoAtNetConfig()
	elif model_name.lower() == "resnet18":
		return None
	else:
		raise ValueError(f"Unknown model {model_name}")

def get_model(model_name: str, config=None, num_classes=200):
	model_name = model_name.lower()

	if model_name == "cvt":
		return CvT(config)
	elif model_name == "vit":
		return ViT(config)
	elif model_name == "coatnet":
		return CoAtNet(config)
	elif model_name == "resnet18":
		model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
		model.fc = nn.Linear(model.fc.in_features, num_classes)
		return model
	else:
		raise ValueError(f"Unknown model {model_name}")

def get_param_groups(model, weight_decay):
	decay, no_decay = [], []

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue

		if (
			name.endswith("bias")
			or "norm" in name.lower()
			or "layerscale" in name.lower()
		):
			no_decay.append(param)
		else:
			decay.append(param)

	return [
		{"params": decay, "weight_decay": weight_decay},
		{"params": no_decay, "weight_decay": 0.0},
	]

def set_seed(seed=0):
	"""Sets the seed for reproducibility."""
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # For multi-GPU setups

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

# train.py
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:
	def __init__(self, model, optimizer, criterion, device, scheduler=None, scheduler_type=None, patience=15, min_delta=1e-6, writer=None):
		self.model = model
		self.optimizer = optimizer
		self.criterion = criterion
		self.device = device
		self.writer = writer

		self.scheduler = scheduler
		self.scheduler_type = scheduler_type

		# Early stopping
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.early_stop = False

		# Tracking
		self.start_epoch = 0
		self.best_val_loss = float('inf')
	
	def run_one_epoch(self, dataloader, state='train'):
		"""
		Run one epoch of training or evaluation
		"""
		is_training = (state == 'train')
		
		if is_training:
			self.model.train()
		else:
			self.model.eval()
		
		total_loss = 0.0
		correct = 0
		total = 0
		
		with torch.set_grad_enabled(is_training):
			for batch in dataloader:
				x, y = batch
				x, y = x.to(self.device), y.to(self.device)
				
				if is_training:
					self.optimizer.zero_grad()
				
				y_hat = self.model(x)

				loss = self.compute_loss(y_hat, y, x)
				total_loss += loss.detach().cpu().item()
				
				pred_for_acc = y_hat
				if isinstance(y_hat, tuple):
					pred_for_acc = y_hat[0]
				
				correct += torch.sum(torch.argmax(pred_for_acc, dim=1) == y).detach().cpu().item()
				total += len(x)
		
		avg_loss = total_loss / len(dataloader)
		accuracy = correct / total
		
		return avg_loss, accuracy
	
	def compute_loss(self, y_hat, y, x=None):
		if isinstance(y_hat, tuple):
			classification_output, regularization_loss = y_hat
			classification_loss = self.criterion(classification_output, y)
			return classification_loss + regularization_loss
		else:
			return self.criterion(y_hat, y)
	
	def train(self, n_epochs, train_dl, val_dl, save_path=None, config=None, args=None):
		with tqdm(range(self.start_epoch, n_epochs), desc="Training Progress") as pbar:
			for epoch in pbar:
				train_loss, train_acc = self.run_one_epoch(train_dl, state='train')
				val_loss, val_acc = self.run_one_epoch(val_dl, state='eval')
				
				if self.writer:
					self.writer.add_scalar('Loss/train', train_loss, epoch)
					self.writer.add_scalar('Accuracy/train', train_acc, epoch)
					self.writer.add_scalar('Loss/val', val_loss, epoch)
					self.writer.add_scalar('Accuracy/val', val_acc, epoch)
					self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

				if self.scheduler:
					if self.scheduler_type == "reduceonplateau":
						self.scheduler.step(val_loss)
					else:
						self.scheduler.step()

				if save_path:
					self.save_checkpoint(
						save_path,
						epoch=epoch,
						config=config,
						args=args,
						best=(val_loss <= self.best_val_loss - self.min_delta)
					)

				if val_loss <= self.best_val_loss - self.min_delta:
					self.best_val_loss = val_loss
					self.counter = 0
				else:
					self.counter += 1
					if self.counter >= self.patience:
						print(f"\nEarly stopping triggered at epoch {epoch+1}")
						self.early_stop = True
						break
				
				pbar.set_postfix({
					'epoch': epoch + 1,
					'train_loss': f'{train_loss:.4f}',
					'train_acc': f'{train_acc:.3f}',
					'val_loss': f'{val_loss:.4f}',
					'val_acc': f'{val_acc:.3f}',
					'best_val_loss': f'{self.best_val_loss:.4f}'
				})

	def save_checkpoint(self, path, epoch, config=None, args=None, best=False):
		"""Save training checkpoint."""

		checkpoint = {
			'epoch': epoch + 1,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'best_val_loss': self.best_val_loss,
		}

		if best:
			best_path = path.replace(".pt", "_best.pt")
			torch.save(checkpoint, best_path)
			print(f"Best model updated and saved to {best_path}")

			checkpoint["best_model_state_dict"] = self.model.state_dict()

		if self.scheduler:
			checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
		if config:
			checkpoint['config'] = config
		if args:
			checkpoint['args'] = args

		torch.save(checkpoint, path)

	def load_checkpoint(self, checkpoint):
		"""Load model/optimizer state for resume."""

		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		if self.scheduler and 'scheduler_state_dict' in checkpoint:
			self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		self.start_epoch = checkpoint.get('epoch', 0)
		self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

		print(f"Resumed from epoch {self.start_epoch}, best val loss {self.best_val_loss:.4f}")
# train_teacher.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

from torchvision.datasets import CIFAR100
from dataset import TinyImageNet
from train import Trainer
from utils import parse_args, get_model, get_param_groups, set_seed

args = parse_args()
set_seed(args.seed)

transfer_writer = SummaryWriter(f"runs/resnet32_transfer")

print("=" * 60)
print("Training Teacher Model (ResNet32 on TinyImageNet)")
print("=" * 60)
for arg in vars(args):
	print(f"{arg:20}: {getattr(args, arg)}")
print("=" * 60)

if args.device == 'auto':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
	device = torch.device(args.device)

print(f"\nUsing device: {device}")
if device.type == 'cuda':
	print(f"GPU: {torch.cuda.get_device_name(device)}")
	print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

train_transform = v2.Compose([
	v2.ToImage(),
	v2.TrivialAugmentWide(),
	v2.RandomResizedCrop(32, scale=(0.7, 1.0)),
	v2.RandomHorizontalFlip(),
	v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
	v2.ToDtype(torch.float32, scale=True),
	v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
	v2.RandomErasing(p=0.25, scale=(0.02, 0.2)),
])

test_transform = v2.Compose([
	v2.ToImage(),
	v2.Resize(32),
	v2.ToDtype(torch.float32, scale=True),
	v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

print(f"\nLoading dataset from {args.data_root}")
train_ds = CIFAR100(root='./datasets', train=True, download=True, transform=train_transform)
val_ds = CIFAR100(root='./datasets', train=False, transform=test_transform)

print(f"Train samples: {len(train_ds):,}")
print(f"Val samples: {len(val_ds):,}")
print(f"Test samples: {len(val_ds):,}")

train_dl = DataLoader(
	train_ds, shuffle=True, batch_size=args.batch_size,
	num_workers=args.num_workers, pin_memory=True
)
val_dl = DataLoader(
	val_ds, shuffle=False, batch_size=args.batch_size,
	num_workers=args.num_workers, pin_memory=True,
)
test_dl = DataLoader(
	val_ds, shuffle=False, batch_size=args.batch_size,
	num_workers=args.num_workers, pin_memory=True,
)

model = get_model("resnet32", num_classes=args.num_classes).to(device)

# --- Phase 1: Transfer Learning (Training Classifier Head) ---
print("\n--- Phase 1: Transfer Learning (Training Classifier Head) ---")

for param in model.parameters():
	param.requires_grad = False

for param in model.fc.parameters():
	param.requires_grad = True

nn.init.xavier_uniform_(model.fc.weight)
nn.init.zeros_(model.fc.bias)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: RESNET32")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters (head only): {trainable_params:,}")

optimizer = AdamW(model.fc.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.transfer_epochs)

trainer = Trainer(
	model,
	optimizer,
	criterion,
	device,
	scheduler=scheduler,
	scheduler_type="cosineannealing",
	writer=transfer_writer,
)

if args.resume:
	print(f"\nResuming from checkpoint: {args.resume}")
	checkpoint = torch.load(args.resume, map_location=device)
	trainer.load_checkpoint(checkpoint)

if args.transfer_epochs > 0:
	print(f"\nStarting transfer learning for {args.transfer_epochs} epochs...")
	trainer.train(
		args.transfer_epochs, train_dl, val_dl,
		save_path=f"{args.save_path}/_resnet32_teacher.pt",
		args=vars(args)
	)

# --- Phase 2: Fine-tuning (Training Full Model) ---
if args.finetune_epochs > 0:
	print("\n--- Phase 2: Fine-tuning (Training Full Model) ---")

	ft_writer = SummaryWriter(f"runs/resnet32_finetune")

	for param in model.parameters():
		param.requires_grad = True

	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Trainable parameters (full model): {trainable_params:,}")

	param_groups = get_param_groups(model, args.weight_decay)
	optimizer = AdamW(param_groups, lr=args.lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)

	finetune_trainer = Trainer(
		model,
		optimizer,
		criterion,
		device,
		scheduler=scheduler,
		scheduler_type="cosineannealing",
		writer=ft_writer,
	)

	print(f"\nStarting fine-tuning for {args.finetune_epochs} epochs...")
	finetune_trainer.train(
		args.finetune_epochs, train_dl, val_dl,
		save_path=f"{args.save_path}/resnet32_teacher.pt",
		args=vars(args)
	)

print(f"\n{'='*60}")
print("FINAL EVALUATION")
print(f"{'='*60}")
test_loss, test_acc = trainer.run_one_epoch(test_dl, state='eval')
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.1%}")
print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
print(f"{'='*60}")
# utils.py
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights
from models.cvt import CvT
from models.vit import ViT
from models.coatnet import CoAtNet
from models.resnet import get_model as get_resnet_cifar
from config import ViTConfig, CoAtNetConfig, CvTConfig

import random
import numpy as np


def parse_args():
	parser = argparse.ArgumentParser(description="Train CvT / ResNet / DeiT on TinyImageNet")

	# General
	parser.add_argument("--data-root", type=str, default="./tiny-imagenet-200", help="Path to TinyImageNet dataset")
	parser.add_argument("--download", action="store_true", help="Download dataset if not found")
	parser.add_argument("--save-path", type=str, default="checkpoints", help="Path to save the trained model checkpoint")
	parser.add_argument("--resume", type=str, default="", help="Path to resume from checkpoint")
	parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', or 'cuda'")

	# Model
	parser.add_argument("--model", type=str, default="vit", choices=["vit", "cvt", "resnet18", "deit"], help="Model type")
	parser.add_argument("--num-classes", type=int, default=100, help="Number of classes")

	# Optimizer
	parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
	parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
	parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")

	# Scheduler
	parser.add_argument(
		"--scheduler", type=str, default="cosineannealing",
		choices=["cosineannealing", "reduceonplateau", "onecycle"],
		help="Learning rate scheduler",
  )
	parser.add_argument(
		"--warmup-steps", type=int, default=0,
		help="Number of warmup steps before main scheduler kicks in"
	)

	# Training
	parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
	parser.add_argument("--transfer-epochs", type=int, default=20, help="Number of epochs for transfer learning (head only)")
	parser.add_argument("--finetune-epochs", type=int, default=80, help="Number of epochs for fine-tuning (full model)")
	parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
	parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for CE loss")

	# Distillation
	parser.add_argument("--distillation", action="store_true", help="Enable knowledge distillation")
	parser.add_argument("--teacher-model", type=str, default="resnet20", choices=["resnet18", "resnet20", "resnet32"])
	parser.add_argument("--teacher-path", type=str, default="", help="Path to teacher checkpoint")
	parser.add_argument("--alpha", type=float, default=0.5, help="Weight for distillation loss")
	parser.add_argument("--tau", type=float, default=1.0, help="Temperature for distillation")

	return parser.parse_args()

def get_config(model_name: str, args=None):
	if model_name.lower() == "cvt":
		return CvTConfig()
	elif model_name.lower() == "vit":
		return ViTConfig()
	elif model_name.lower() == "coatnet":
		return CoAtNetConfig()
	elif model_name.lower() == "resnet18":
		return None
	elif model_name.lower() == "resnet20":
		return None
	else:
		raise ValueError(f"Unknown model {model_name}")

def get_model(model_name: str, config=None, num_classes=100):
	model_name = model_name.lower()

	if model_name == "cvt":
		return CvT(config)
	elif model_name == "vit":
		return ViT(config)
	elif model_name == "coatnet":
		return CoAtNet(config)
	elif model_name == "resnet18":
		model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
		model.fc = nn.Linear(model.fc.in_features, num_classes)

		return model
	elif model_name == "resnet20" or model_name == "resnet32":
		model = get_resnet_cifar(model_name, num_classes=num_classes)

		return model
	else:
		raise ValueError(f"Unknown model {model_name}")

def get_param_groups(model, weight_decay):
	decay, no_decay = [], []

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue

		if (
			name.endswith("bias")
			or "norm" in name.lower()
			or "layerscale" in name.lower()
		):
			no_decay.append(param)
		else:
			decay.append(param)

	return [
		{"params": decay, "weight_decay": weight_decay},
		{"params": no_decay, "weight_decay": 0.0},
	]

def set_seed(seed=0):
	"""Sets the seed for reproducibility."""
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # For multi-GPU setups

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def load_teacher_model(path, model_name, num_classes, device):
	checkpoint = torch.load(path, map_location=device)
	config = checkpoint.get('config')
	
	model = get_model(model_name, config, num_classes=num_classes)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.to(device)
	model.eval()
	
	return model

# File: tmp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.norm import DyT, RMSNorm

# from dataclasses import dataclass

class ViTConfig:
	d_model=512
	d_ff=2048

	n_head=16
	n_block=4
	# n_patch=16
	n_patch=8
	out_d = 200 # TinyImageNet

	norm=RMSNorm
	activation="swiglu"

	dropout=0.1

	use_layer_scale=True
	use_qk_norm=False

	# chw=(3, 224, 224)
	chw=(3, 64, 64)

	kernel_size=16
	stride=16

class CoAtNetConfig:
	dims=[64, 96, 192, 384, 768]
	depths=[2, 2, 3, 5, 2]
	num_heads=32

	# chw=(3, 224, 224)
	chw=(3, 64, 64)

	activation="silu"
	norm=RMSNorm

	dropout=0.1

	out_d = 200 # TinyImageNet

class CvTConfig:
	# Based on CvT-13, adjusted for TinyImageNet
	chw = (3, 64, 64)
	out_d = 200 # TinyImageNet
	
	# Stage 1
	s1_emb_kernel = 7
	s1_emb_stride = 4
	s1_emb_pad = 2
	s1_emb_dim = 64
	s1_depth = 1
	s1_heads = 1
	s1_mlp_ratio = 4.0
	s1_qkv_kernel = 3
	s1_qkv_stride = 1
	s1_qkv_pad = 1

	# Stage 2
	s2_emb_kernel = 3
	s2_emb_stride = 2
	s2_emb_pad = 1
	s2_emb_dim = 192
	s2_depth = 2
	s2_heads = 3
	s2_mlp_ratio = 4.0
	s2_qkv_kernel = 3
	s2_qkv_stride = 2
	s2_qkv_pad = 1

	# Stage 3
	s3_emb_kernel = 3
	s3_emb_stride = 2
	s3_emb_pad = 1
	s3_emb_dim = 384
	s3_depth = 10
	s3_heads = 6
	s3_mlp_ratio = 4.0
	s3_qkv_kernel = 3
	s3_qkv_stride = 2
	s3_qkv_pad = 1

	dropout=0.1
	norm=RMSNorm
	activation="gelu"
	use_layer_scale=False
import os
import requests
import zipfile
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from tqdm import tqdm

class TinyImageNet(Dataset):
	url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
	filename = 'tiny-imagenet-200.zip'
	folder_name = 'tiny-imagenet-200'
	
	def __init__(self, root, split='train', transform=None, download=False, test_size=0.1, random_state=0):
		self.root = os.path.expanduser(root)
		self.split = split
		self.transform = transform
		self.test_size = test_size
		self.random_state = random_state
		
		if download:
			self.download()
			
		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')
		
		self.data = []
		self.targets = []
		
		self.class_to_idx = {}
		with open(os.path.join(self.root, self.folder_name, 'wnids.txt'), 'r') as f:
			for i, line in enumerate(f):
				self.class_to_idx[line.strip()] = i
		
		self._load_data()
	
	def _load_data(self):
		"""Load and split the data based on the specified split"""
		if self.split == 'train':
			self._load_from_folder('train')
		elif self.split == 'val':
			self._load_from_folder('val')
		elif self.split == 'test':
			self._load_from_folder('test')
		else:
			raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")

	def _load_from_folder(self, folder):
		"""Load images and labels directly from the specified subfolder"""
		if folder == 'train':
			split_dir = os.path.join(self.root, self.folder_name, folder)
			for class_folder in os.listdir(split_dir):
				class_idx = self.class_to_idx[class_folder]
				class_dir = os.path.join(split_dir, class_folder, 'images')
				for img_name in os.listdir(class_dir):
					img_path = os.path.join(class_dir, img_name)
					self.data.append(img_path)
					self.targets.append(class_idx)
		elif folder == 'val':
			split_dir = os.path.join(self.root, self.folder_name, folder)
			with open(os.path.join(split_dir, 'val_annotations.txt'), 'r') as f:
				for line in f:
					parts = line.strip().split('\t')
					img_name, class_id = parts[0], parts[1]
					img_path = os.path.join(split_dir, 'images', img_name)
					self.data.append(img_path)
					self.targets.append(self.class_to_idx[class_id])
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		img_path = self.data[idx]
		target = self.targets[idx]
		
		with open(img_path, 'rb') as f:
			img = Image.open(f).convert('RGB')
		
		if self.transform:
			img = self.transform(img)
			
		return img, target
	
	def _check_integrity(self):
		return os.path.isdir(os.path.join(self.root, self.folder_name))
	
	def download(self):
		if self._check_integrity():
			print('Files already downloaded and verified')
			return
		
		download_url(self.url, self.root, self.filename)
		with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zip_ref:
			zip_ref.extractall(self.root)
	
	def get_split_info(self):
		"""Return information about the dataset splits"""
		class_counts = {}
		for target in self.targets:
			class_counts[target] = class_counts.get(target, 0) + 1
			
		return {
			'split': self.split,
			'total_samples': len(self.data),
			'num_classes': len(self.class_to_idx),
			'samples_per_class': class_counts
		}

import torch
import torch.nn.functional as F
from train import Trainer

import torch
import torch.nn.functional as F

class DeiTTrainer(Trainer):
	def __init__(self, model, teacher_model, optimizer, criterion, device, alpha=0.5, tau=1.0):
		super().__init__(model, optimizer, criterion, device)
		self.teacher_model = teacher_model
		self.teacher_model.eval()

		for param in self.teacher_model.parameters():
			param.requires_grad = False

		self.alpha = alpha
		self.tau = tau
	
	def compute_loss(self, y_hat, y, x=None):
		if isinstance(y_hat, tuple):
			student_pred, student_distill = y_hat
		else:
			student_pred = y_hat
			student_distill = y_hat
		
		base_loss = self.criterion(student_pred, y)
		
		if x is not None:
			with torch.no_grad():
				teacher_pred = self.teacher_model(x)

				if isinstance(teacher_pred, tuple):
					teacher_pred = teacher_pred[0]
		else:
			return base_loss
		
		distill_loss = F.kl_div(
			F.log_softmax(student_distill / self.tau, dim=1),
			F.softmax(teacher_pred / self.tau, dim=1),
			reduction='batchmean'
		) * (self.tau * self.tau)
		
		return (1 - self.alpha) * base_loss + self.alpha * distill_loss
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
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import ConvAttention
from layers.transformers import FeedForward

class CvTBlock(nn.Module):
	def __init__(self, config, kernel_size, qkv_stride, padding):
		super().__init__()

		self.d_model = config.d_model
		
		self.attn_norm = config.norm(self.d_model)
		self.attn = ConvAttention(
			dim=self.d_model, 
			n_head=config.n_head,
			kernel_size=kernel_size,
			qkv_stride=qkv_stride,
			padding=padding,
			dropout=config.dropout
		)
		self.attn_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.attn_dropout = nn.Dropout(config.dropout)

		self.ff_norm = config.norm(self.d_model)
		self.ff = FeedForward(config)

		self.ff_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.ff_dropout = nn.Dropout(config.dropout)

	def forward(self, x, h, w):
		norm_x = self.attn_norm(x)
		attn_out = self.attn(norm_x, h, w)
		attn_out = self.attn_dropout(attn_out)
		
		if self.attn_layer_scale is not None:
			x = x + self.attn_layer_scale * attn_out
		else:
			x = x + attn_out

		norm_x = self.ff_norm(x)
		ff_out = self.ff(norm_x)
		ff_out = self.ff_dropout(ff_out)
		if self.ff_layer_scale is not None:
			x = x + self.ff_layer_scale * ff_out
		else:
			x = x + ff_out
		
		return x, h, w
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

		# TODO: checking out parallel Linear from llama
		# fc_in can be column parallel
		# fc_out can be row parallel

		if self.activation in ('swiglu', 'geglu'):
			# default scaling down by 2/3 since normal 
			# d_ff is 4xd_model
			# Should be ~2.667 scalling now
			# based on Llama SwiGLU FeedForward
			# https://github.com/meta-llama/llama
			d_ff = int(2 * config.d_ff // 3)
			self.fc_in = nn.Linear(config.d_model, d_ff * 2)
		else:
			self.fc_in = nn.Linear(config.d_model, config.d_ff)

		self.fc_out = nn.Linear(config.d_ff, config.d_model) # can be row parallel

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
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import MHA
from layers.transformers import FeedForward

class ViTBlock(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.d_model = config.d_model
		self.n_head = config.n_head
		
		self.attn_norm = config.norm(self.d_model)
		self.attn = MHA(config)
		self.attn_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.attn_dropout = nn.Dropout(config.dropout)

		self.ff_norm = config.norm(self.d_model)
		self.ff = FeedForward(config)
		self.ff_layer_scale = nn.Parameter(torch.ones(self.d_model) * 1e-4) if config.use_layer_scale else None
		self.ff_dropout = nn.Dropout(config.dropout)

	def forward(self, x):
		norm_x = self.attn_norm(x)
		attn_out = self.attn(norm_x)
		attn_out = self.attn_dropout(attn_out)
		if self.attn_layer_scale is not None:
			x = x + self.attn_layer_scale * attn_out
		else:
			x = x + attn_out

		norm_x = self.ff_norm(x)
		ff_out = self.ff(norm_x)
		ff_out = self.ff_dropout(ff_out)
		if self.ff_layer_scale is not None:
			x = x + self.ff_layer_scale * ff_out
		else:
			x = x + ff_out

		return x
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.models import resnet18, ResNet18_Weights

from dataset import TinyImageNet
from train import Trainer
from deit_trainer import DeiTTrainer

from utils import parse_args, get_config, get_model, get_param_groups, set_seed
from schedulers import WarmupScheduler

def main():
	args = parse_args()
	set_seed(0)

	print("=" * 60)
	print("Training Configuration:")
	print("=" * 60)
	for arg in vars(args):
		print(f"{arg:20}: {getattr(args, arg)}")
	print("=" * 60)

	print(f"\nTraining {args.model.upper()} model")
	print(f"Distillation: {'Enabled' if args.distillation else 'Disabled'}")
	if args.distillation:
		print(f"Teacher model: {args.teacher_model.upper()}")
		if args.teacher_path:
			print(f"Teacher path: {args.teacher_path}")
		print(f"Alpha (distillation weight): {args.alpha}")
		print(f"Temperature: {args.tau}")

	if args.device == 'auto':
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	else:
		device = torch.device(args.device)

	print(f"\nUsing device: {device}")
	if device.type == 'cuda':
		print(f"GPU: {torch.cuda.get_device_name(device)}")
		print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

	train_transform = v2.Compose([
		v2.ToImage(),
		v2.TrivialAugmentWide(),
		v2.RandomResizedCrop(64, scale=(0.7, 1.0)),
		v2.RandomHorizontalFlip(),
		v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		v2.RandomErasing(p=0.25, scale=(0.02, 0.2)),
	])

	test_transform = v2.Compose([
		v2.ToImage(),
		v2.Resize(72),
		v2.CenterCrop(64),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	print(f"\nLoading dataset from {args.data_root}")
	train_ds = TinyImageNet(root='./datasets', split='train', download=True, transform=train_transform)
	val_ds = TinyImageNet(root='./datasets', split='val', transform=test_transform)
	test_ds = TinyImageNet(root='./datasets', split='test', transform=test_transform)

	print(f"Train samples: {len(train_ds):,}")
	print(f"Val samples: {len(val_ds):,}")
	print(f"Test samples: {len(test_ds):,}")

	train_dl = DataLoader(
		train_ds, shuffle=True, batch_size=args.batch_size, 
		num_workers=args.num_workers, pin_memory=True
	)
	val_dl = DataLoader(
		val_ds, shuffle=False, batch_size=args.batch_size, 
		num_workers=args.num_workers, pin_memory=True,
	)
	test_dl = DataLoader(
		test_ds, shuffle=False, batch_size=args.batch_size, 
		num_workers=args.num_workers, pin_memory=True,
	)

	config = get_config(args.model, args)
	model = get_model(args.model, config).to(device)

	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"\nModel: {args.model.upper()}")
	print(f"Total parameters: {total_params:,}")
	print(f"Trainable parameters: {trainable_params:,}")

	param_groups = get_param_groups(model, args.weight_decay)
	if args.optimizer == 'adamw':
		optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
	elif args.optimizer == 'adam':
		optimizer = torch.optim.Adam(param_groups, lr=args.lr)
	elif args.optimizer == 'sgd':
		optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum)

	if args.label_smoothing > 0:
		criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
	else:
		criterion = nn.CrossEntropyLoss()

	base_scheduler = None
	if args.scheduler == "cosineannealing":
		base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	elif args.scheduler == "reduceonplateau":
		base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
	elif args.scheduler == "onecycle":
		base_scheduler = torch.optim.lr_scheduler.OneCycleLR(
			optimizer, max_lr=args.lr, steps_per_epoch=len(train_dl), epochs=args.epochs
		)

	if args.warmup_steps > 0:
		scheduler = WarmupScheduler(optimizer, warmup_steps=args.warmup_steps, after_scheduler=base_scheduler)
	else:
		scheduler = base_scheduler

	if args.resume:
		print(f"\nResuming from checkpoint: {args.resume}")
		checkpoint = torch.load(args.resume, map_location=device)
		trainer.load_checkpoint(checkpoint)
	
	if args.distillation:
		print(f"\nSetting up distillation training...")
		
		if args.teacher_path:
			print(f"Loading teacher from {args.teacher_path}")
			teacher_model = load_teacher_model(
				args.teacher_path, 
				args.teacher_model, 
				args.num_classes, 
				device
			)
		else:
			print("Creating pretrained teacher model")
			teacher_config = get_config(args.teacher_model, args)
			teacher_model = create_model(args.teacher_model, teacher_config).to(device)
			
			if args.teacher_model == 'resnet18':
				teacher_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
				teacher_model.fc = nn.Linear(teacher_model.fc.in_features, args.num_classes)
				teacher_model = teacher_model.to(device)

				nn.init.xavier_uniform_(teacher_model.fc.weight)
				nn.init.zeros_(teacher_model.fc.bias)
		
		teacher_model.eval()
		teacher_params = sum(p.numel() for p in teacher_model.parameters())
		print(f"Teacher parameters: {teacher_params:,}")
		
		trainer = DeiTTrainer(
			model,
			teacher_model,
			optimizer,
			criterion,
			device,
			scheduler=scheduler,
			scheduler_type=args.scheduler,
			alpha=args.alpha,
			tau=args.tau
		)
	else:
		print(f"\nSetting up standard training...")
		trainer = Trainer(
			model,
			optimizer,
			criterion,
			device,
			scheduler=scheduler,
			scheduler_type=args.scheduler,
		)
	
	print(f"\nStarting training for {args.epochs} epochs...")
	trainer.train(
		args.epochs, train_dl, val_dl,
		save_path=f"{args.save_path}/{args.model}",
		config=config,
		args=vars(args)
	)
	
	print(f"\n{'='*60}")
	print("FINAL EVALUATION")
	print(f"{'='*60}")
	test_loss, test_acc = trainer.run_one_epoch(test_dl, state='eval')
	print(f"Test Loss: {test_loss:.4f}")
	print(f"Test Accuracy: {test_acc:.1%}")
	print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
	print(f"{'='*60}")

if __name__ == "__main__":
	main()
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from layers.vit import ViTBlock
from layers.norm import DyT, RMSNorm
from layers.conv import MBConv

class CoAtNet(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.activation = activation.lower()
		if self.activation == "gelu":
			self.act_fn = nn.GELU
		else:
			self.act_fn = nn.SiLU

		# Convolutional Stem
		self.s0 = nn.Sequential(
			nn.Conv2d(config.chw[0], config.dims[0], kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(dims[0]),
			self.act_fn(),
		)
		
		# Convolutional Stages (MBConv)
		self.s1 = self._make_conv_stage(config.dims[0], config.dims[1], config.depths[0], stride=2)
		self.s2 = self._make_conv_stage(config.dims[1], config.dims[2], config.depths[1], stride=2)

		# Transformer Stages
		self.s3_pe = nn.Conv2d(config.dims[2], config.dims[3], 1) # Patch embedding
		self.s3 = self._make_transformer_stage(config.dims[3], config.depths[2], config.num_heads, cofnig.norm, config.activation)

		self.s4_pe = nn.Conv2d(config.dims[3], config.dims[4], 2, 2) # Patch merging
		self.s4 = self._make_transformer_stage(config.dims[4], config.depths[3], config.num_heads, cofnig.norm, config.activation)

		self.head = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(config.dims[-1], config.num_classes)
		)

	def _make_conv_stage(self, in_chans, out_chans, depth, stride):
		layers = [MBConv(in_chans, out_chans, stride=stride)]

		for _ in range(depth - 1):
			layers.append(MBConv(out_chans, out_chans, stride=1))

		return nn.Sequential(*layers)

	def _make_transformer_stage(self, dim, depth, num_heads, norm=RMSNorm, activation="silu"):
		class MHAConfig:
			d_model, n_head, dropout, use_qk_norm, use_layer_scale = dim, num_heads, 0.1, False, False
			norm, activation, d_ff = norm, activation, dim * 4
		
		blocks = []
		for _ in range(depth):
			blocks.append(ViTBlock(MHAConfig()))

		return nn.ModuleList(blocks)

	def forward(self, x):
		B = x.shape[0]

		# Conv stages
		x = self.s0(x)
		x = self.s1(x)
		x = self.s2(x)
		
		# Transformer stage 1
		x = self.s3_pe(x)
		_, C, H, W = x.shape
		x = rearrange(x, 'b c h w -> b (h w) c')
		for blk in self.s3:
			x = blk(x)
		x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

		# Transformer stage 2
		x = self.s4_pe(x)
		_, C, H, W = x.shape
		x = rearrange(x, 'b c h w -> b (h w) c')
		for blk in self.s4:
			x = blk(x)
		x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
		
		# Head
		x = self.head(x)

		return x
import torch
import torch.nn as nn
from einops import rearrange

from layers.embeddings import ConvEmbedding
from layers.cvt import CvTBlock

class CvT(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		
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
		
		return self.head(x)
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
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.utils import get_positional_embeddings
from layers.embeddings import patchify
from layers.vit import ViTBlock

class ViT(nn.Module):
	def __init__(self, config):
		super().__init__()

		self.chw = config.chw # (C, H, W)

		self.d_model = config.d_model
		self.n_patch = config.n_patch
		self.n_block = config.n_block

		print(self.chw)
		assert self.chw[1] % self.n_patch == 0, "Input shape not entirely divisible by number of patch"
		assert self.chw[2] % self.n_patch == 0, "Input shape not entirely divisible by number of patch"

		self.patch_size = (self.chw[1] // self.n_patch, self.chw[2] // self.n_patch)

		self.input_d = self.chw[0] * self.patch_size[0] * self.patch_size[1]
		self.linear_mapper = nn.Linear(self.input_d, self.d_model)

		self.class_token = nn.Parameter(torch.rand(1, self.d_model))	

		# self.register_buffer('pos_embed', get_positional_embeddings(n_patch ** 2 + 1, d_model), persistent=False)
		self.pos_embed = nn.Parameter(torch.randn(1, self.n_patch ** 2 + 1, self.d_model))

		self.blocks = nn.ModuleList([ViTBlock(config) for _ in range(self.n_block)])

		self.mlp = nn.Linear(self.d_model, config.out_d)

	def forward(self, images):
		bsz = images.size(0)

		patch = patchify(images, self.n_patch)
		tokens = self.linear_mapper(patch)

		tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

		pos_embed = self.pos_embed.repeat(bsz, 1, 1)
		out = tokens + pos_embed

		for block in self.blocks:
			out = block(out)

		# [CLS] token
		out = out[:, 0]

		return self.mlp(out)
from torch.optim.lr_scheduler import _LRScheduler

class WarmupScheduler(_LRScheduler):
	def __init__(self, optimizer, warmup_steps, after_scheduler=None, last_epoch=-1):
		self.warmup_steps = warmup_steps
		self.after_scheduler = after_scheduler
		self.finished_warmup = False
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		if self.last_epoch < self.warmup_steps:
			# Linear warmup
			return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]

		if self.after_scheduler:
			if not self.finished_warmup:
				# reset after warmup
				self.after_scheduler.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
				self.finished_warmup = True

			return self.after_scheduler.get_last_lr()

		return [group["lr"] for group in self.optimizer.param_groups]

	def step(self, epoch=None, metrics=None):
		if self.last_epoch < self.warmup_steps:
			return super(WarmupScheduler, self).step(epoch)
		if self.after_scheduler:
			if isinstance(self.after_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
				self.after_scheduler.step(metrics)
			else:
				if epoch is None:
					self.after_scheduler.step(None)
				else:
					self.after_scheduler.step(epoch - self.warmup_steps)
from tqdm import tqdm
import torch

class Trainer:
	def __init__(self, model, optimizer, criterion, device, scheduler=None, scheduler_type=None, patience=10, min_delta=1e-4):
		self.model = model
		self.optimizer = optimizer
		self.criterion = criterion
		self.device = device

		self.scheduler = scheduler
		self.scheduler_type = scheduler_type

		# Early stopping
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.early_stop = False

		# Tracking
		self.start_epoch = 0
		self.best_val_loss = float('inf')
	
	def run_one_epoch(self, dataloader, state='train'):
		"""
		Run one epoch of training or evaluation
		"""
		is_training = (state == 'train')
		
		if is_training:
			self.model.train()
		else:
			self.model.eval()
		
		total_loss = 0.0
		correct = 0
		total = 0
		
		with torch.set_grad_enabled(is_training):
			for batch in dataloader:
				x, y = batch
				x, y = x.to(self.device), y.to(self.device)
				
				if is_training:
					self.optimizer.zero_grad()
				
				y_hat = self.model(x)
				loss = self.compute_loss(y_hat, y, x)
				
				if is_training:
					loss.backward()
					self.optimizer.step()
				
				total_loss += loss.detach().cpu().item()
				
				pred_for_acc = y_hat
				if isinstance(y_hat, tuple):
					pred_for_acc = y_hat[0]
				
				correct += torch.sum(torch.argmax(pred_for_acc, dim=1) == y).detach().cpu().item()
				total += len(x)
		
		avg_loss = total_loss / len(dataloader)
		accuracy = correct / total
		
		return avg_loss, accuracy
	
	def compute_loss(self, y_hat, y, x=None):
		return self.criterion(y_hat, y)
	
	def train(self, n_epochs, train_dl, val_dl, save_path=None, config=None, args=None):
		with tqdm(range(self.start_epoch, n_epochs), desc="Training Progress") as pbar:
			for epoch in pbar:
				train_loss, train_acc = self.run_one_epoch(train_dl, state='train')
				val_loss, val_acc = self.run_one_epoch(val_dl, state='eval')
				
				if val_loss < self.best_val_loss:
					self.best_val_loss = val_loss

				if self.scheduler:
					if self.scheduler_type == "reduceonplateau":
						self.scheduler.step(val_loss)
					else:
						self.scheduler.step()

				if save_path:
					self.save_checkpoint(
						save_path,
						epoch=epoch,
						config=config,
						args=args,
						best=(val_loss < self.best_val_loss)
					)

				if val_loss + self.min_delta < self.best_val_loss:
					self.best_val_loss = val_loss
					self.counter = 0
				else:
					self.counter += 1
					if self.counter >= self.patience:
						print(f"\nEarly stopping triggered at epoch {epoch+1}")
						self.early_stop = True
						break
				
				pbar.set_postfix({
					'epoch': epoch + 1,
					'train_loss': f'{train_loss:.4f}',
					'train_acc': f'{train_acc:.3f}',
					'val_loss': f'{val_loss:.4f}',
					'val_acc': f'{val_acc:.3f}',
					'best_val_loss': f'{self.best_val_loss:.4f}'
				})

	def save_checkpoint(self, path, epoch, config=None, args=None, best=False):
		"""Save training checkpoint."""

		checkpoint = {
			'epoch': epoch + 1,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'best_val_loss': self.best_val_loss,
		}
		if self.scheduler:
			checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
		if config:
			checkpoint['config'] = config
		if args:
			checkpoint['args'] = args

		torch.save(checkpoint, path)

		if best:
			best_path = path.replace(".pt", "_best.pt")
			torch.save(checkpoint, best_path)
			print(f"Best model updated and saved to {best_path}")

	def load_checkpoint(self, checkpoint):
		"""Load model/optimizer state for resume."""

		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		if self.scheduler and 'scheduler_state_dict' in checkpoint:
			self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		self.start_epoch = checkpoint.get('epoch', 0)
		self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

		print(f"Resumed from epoch {self.start_epoch}, best val loss {self.best_val_loss:.4f}")
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights
from models.cvt import CvT
from models.vit import ViT
from models.coatnet import CoAtNet
from config import ViTConfig, CoAtNetConfig, CvTConfig

import random
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser(description="Train CvT / ResNet / DeiT on TinyImageNet")

	# General
	parser.add_argument("--data-root", type=str, default="./tiny-imagenet-200", help="Path to TinyImageNet dataset")
	parser.add_argument("--download", action="store_true", help="Download dataset if not found")
	parser.add_argument("--save-path", type=str, default="checkpoints", help="Path to save the trained model checkpoint")
	parser.add_argument("--resume", type=str, default="", help="Path to resume from checkpoint")
	parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', or 'cuda'")

	# Model
	parser.add_argument("--model", type=str, default="vit", choices=["vit", "cvt", "resnet18", "deit"], help="Model type")
	parser.add_argument("--num-classes", type=int, default=200, help="Number of classes")

	# Optimizer
	parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
	parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
	parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")

	# Scheduler
	parser.add_argument(
		"--scheduler", type=str, default="cosineannealing",
		choices=["cosineannealing", "reduceonplateau", "onecycle"],
		help="Learning rate scheduler",
  )
	parser.add_argument(
		"--warmup-steps", type=int, default=0,
		help="Number of warmup steps before main scheduler kicks in"
	)

	# Training
	parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
	parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
	parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for CE loss")

	# Distillation
	parser.add_argument("--distillation", action="store_true", help="Enable knowledge distillation")
	parser.add_argument("--teacher-model", type=str, default="resnet18", choices=["resnet18", "cvt", "deit"])
	parser.add_argument("--teacher-path", type=str, default="", help="Path to teacher checkpoint")
	parser.add_argument("--alpha", type=float, default=0.5, help="Weight for distillation loss")
	parser.add_argument("--tau", type=float, default=1.0, help="Temperature for distillation")

	return parser.parse_args()

def get_config(model_name: str, args=None):
	if model_name.lower() == "cvt":
		return CvTConfig()
	elif model_name.lower() == "vit":
		return ViTConfig()
	elif model_name.lower() == "coatnet":
		return CoAtNetConfig()
	elif model_name.lower() == "resnet18":
		return None
	else:
		raise ValueError(f"Unknown model {model_name}")

def get_model(model_name: str, config=None, num_classes=200):
	model_name = model_name.lower()

	if model_name == "cvt":
		return CvT(config)
	elif model_name == "vit":
		return ViT(config)
	elif model_name == "coatnet":
		return CoAtNet(config)
	elif model_name == "resnet18":
		model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
		model.fc = nn.Linear(model.fc.in_features, num_classes)
		return model
	else:
		raise ValueError(f"Unknown model {model_name}")

def get_param_groups(model, weight_decay):
	decay, no_decay = [], []

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue

		if (
			name.endswith("bias")
			or "norm" in name.lower()
			or "layerscale" in name.lower()
		):
			no_decay.append(param)
		else:
			decay.append(param)

	return [
		{"params": decay, "weight_decay": weight_decay},
		{"params": no_decay, "weight_decay": 0.0},
	]

def set_seed(seed=0):
	"""Sets the seed for reproducibility."""
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # For multi-GPU setups

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

# File: train.py
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:
	def __init__(self, model, optimizer, criterion, device, scheduler=None, scheduler_type=None, patience=15, min_delta=1e-6, writer=None):
		self.model = model
		self.optimizer = optimizer
		self.criterion = criterion
		self.device = device
		self.writer = writer

		self.scheduler = scheduler
		self.scheduler_type = scheduler_type

		# Early stopping
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.early_stop = False

		# Tracking
		self.start_epoch = 0
		self.best_val_loss = float('inf')
	
	def run_one_epoch(self, dataloader, state='train'):
		"""
		Run one epoch of training or evaluation
		"""
		is_training = (state == 'train')
		
		if is_training:
			self.model.train()
		else:
			self.model.eval()
		
		total_loss = 0.0
		correct = 0
		total = 0
		
		with torch.set_grad_enabled(is_training):
			for batch in dataloader:
				x, y = batch
				x, y = x.to(self.device), y.to(self.device)
				
				if is_training:
					self.optimizer.zero_grad()
				
				y_hat = self.model(x)

				loss = self.compute_loss(y_hat, y, x)
				total_loss += loss.detach().cpu().item()
				
				pred_for_acc = y_hat
				if isinstance(y_hat, tuple):
					pred_for_acc = y_hat[0]
				
				correct += torch.sum(torch.argmax(pred_for_acc, dim=1) == y).detach().cpu().item()
				total += len(x)
		
		avg_loss = total_loss / len(dataloader)
		accuracy = correct / total
		
		return avg_loss, accuracy
	
	def compute_loss(self, y_hat, y, x=None):
		if isinstance(y_hat, tuple):
			classification_output, regularization_loss = y_hat
			classification_loss = self.criterion(classification_output, y)
			return classification_loss + regularization_loss
		else:
			return self.criterion(y_hat, y)
	
	def train(self, n_epochs, train_dl, val_dl, save_path=None, config=None, args=None):
		with tqdm(range(self.start_epoch, n_epochs), desc="Training Progress") as pbar:
			for epoch in pbar:
				train_loss, train_acc = self.run_one_epoch(train_dl, state='train')
				val_loss, val_acc = self.run_one_epoch(val_dl, state='eval')
				
				if self.writer:
					self.writer.add_scalar('Loss/train', train_loss, epoch)
					self.writer.add_scalar('Accuracy/train', train_acc, epoch)
					self.writer.add_scalar('Loss/val', val_loss, epoch)
					self.writer.add_scalar('Accuracy/val', val_acc, epoch)
					self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

				if self.scheduler:
					if self.scheduler_type == "reduceonplateau":
						self.scheduler.step(val_loss)
					else:
						self.scheduler.step()

				if save_path:
					self.save_checkpoint(
						save_path,
						epoch=epoch,
						config=config,
						args=args,
						best=(val_loss <= self.best_val_loss - self.min_delta)
					)

				if val_loss <= self.best_val_loss - self.min_delta:
					self.best_val_loss = val_loss
					self.counter = 0
				else:
					self.counter += 1
					if self.counter >= self.patience:
						print(f"\nEarly stopping triggered at epoch {epoch+1}")
						self.early_stop = True
						break
				
				pbar.set_postfix({
					'epoch': epoch + 1,
					'train_loss': f'{train_loss:.4f}',
					'train_acc': f'{train_acc:.3f}',
					'val_loss': f'{val_loss:.4f}',
					'val_acc': f'{val_acc:.3f}',
					'best_val_loss': f'{self.best_val_loss:.4f}'
				})

	def save_checkpoint(self, path, epoch, config=None, args=None, best=False):
		"""Save training checkpoint."""

		checkpoint = {
			'epoch': epoch + 1,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'best_val_loss': self.best_val_loss,
		}

		if best:
			best_path = path.replace(".pt", "_best.pt")
			torch.save(checkpoint, best_path)
			print(f"Best model updated and saved to {best_path}")

			checkpoint["best_model_state_dict"] = self.model.state_dict()

		if self.scheduler:
			checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
		if config:
			checkpoint['config'] = config
		if args:
			checkpoint['args'] = args

		torch.save(checkpoint, path)

	def load_checkpoint(self, checkpoint):
		"""Load model/optimizer state for resume."""

		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		if self.scheduler and 'scheduler_state_dict' in checkpoint:
			self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		self.start_epoch = checkpoint.get('epoch', 0)
		self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

		print(f"Resumed from epoch {self.start_epoch}, best val loss {self.best_val_loss:.4f}")
# File: train_teacher.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

from torchvision.datasets import CIFAR100
from dataset import TinyImageNet
from train import Trainer
from utils import parse_args, get_model, get_param_groups, set_seed

args = parse_args()
set_seed(args.seed)

transfer_writer = SummaryWriter(f"runs/resnet32_transfer")

print("=" * 60)
print("Training Teacher Model (ResNet32 on TinyImageNet)")
print("=" * 60)
for arg in vars(args):
	print(f"{arg:20}: {getattr(args, arg)}")
print("=" * 60)

if args.device == 'auto':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
	device = torch.device(args.device)

print(f"\nUsing device: {device}")
if device.type == 'cuda':
	print(f"GPU: {torch.cuda.get_device_name(device)}")
	print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

train_transform = v2.Compose([
	v2.ToImage(),
	v2.TrivialAugmentWide(),
	v2.RandomResizedCrop(32, scale=(0.7, 1.0)),
	v2.RandomHorizontalFlip(),
	v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
	v2.ToDtype(torch.float32, scale=True),
	v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
	v2.RandomErasing(p=0.25, scale=(0.02, 0.2)),
])

test_transform = v2.Compose([
	v2.ToImage(),
	v2.Resize(32),
	v2.ToDtype(torch.float32, scale=True),
	v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

print(f"\nLoading dataset from {args.data_root}")
train_ds = CIFAR100(root='./datasets', train=True, download=True, transform=train_transform)
val_ds = CIFAR100(root='./datasets', train=False, transform=test_transform)

print(f"Train samples: {len(train_ds):,}")
print(f"Val samples: {len(val_ds):,}")
print(f"Test samples: {len(val_ds):,}")

train_dl = DataLoader(
	train_ds, shuffle=True, batch_size=args.batch_size,
	num_workers=args.num_workers, pin_memory=True
)
val_dl = DataLoader(
	val_ds, shuffle=False, batch_size=args.batch_size,
	num_workers=args.num_workers, pin_memory=True,
)
test_dl = DataLoader(
	val_ds, shuffle=False, batch_size=args.batch_size,
	num_workers=args.num_workers, pin_memory=True,
)

model = get_model("resnet32", num_classes=args.num_classes).to(device)

# --- Phase 1: Transfer Learning (Training Classifier Head) ---
print("\n--- Phase 1: Transfer Learning (Training Classifier Head) ---")

for param in model.parameters():
	param.requires_grad = False

for param in model.fc.parameters():
	param.requires_grad = True

nn.init.xavier_uniform_(model.fc.weight)
nn.init.zeros_(model.fc.bias)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: RESNET32")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters (head only): {trainable_params:,}")

optimizer = AdamW(model.fc.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.transfer_epochs)

trainer = Trainer(
	model,
	optimizer,
	criterion,
	device,
	scheduler=scheduler,
	scheduler_type="cosineannealing",
	writer=transfer_writer,
)

if args.resume:
	print(f"\nResuming from checkpoint: {args.resume}")
	checkpoint = torch.load(args.resume, map_location=device)
	trainer.load_checkpoint(checkpoint)

if args.transfer_epochs > 0:
	print(f"\nStarting transfer learning for {args.transfer_epochs} epochs...")
	trainer.train(
		args.transfer_epochs, train_dl, val_dl,
		save_path=f"{args.save_path}/_resnet32_teacher.pt",
		args=vars(args)
	)

# --- Phase 2: Fine-tuning (Training Full Model) ---
if args.finetune_epochs > 0:
	print("\n--- Phase 2: Fine-tuning (Training Full Model) ---")

	ft_writer = SummaryWriter(f"runs/resnet32_finetune")

	for param in model.parameters():
		param.requires_grad = True

	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Trainable parameters (full model): {trainable_params:,}")

	param_groups = get_param_groups(model, args.weight_decay)
	optimizer = AdamW(param_groups, lr=args.lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)

	finetune_trainer = Trainer(
		model,
		optimizer,
		criterion,
		device,
		scheduler=scheduler,
		scheduler_type="cosineannealing",
		writer=ft_writer,
	)

	print(f"\nStarting fine-tuning for {args.finetune_epochs} epochs...")
	finetune_trainer.train(
		args.finetune_epochs, train_dl, val_dl,
		save_path=f"{args.save_path}/resnet32_teacher.pt",
		args=vars(args)
	)

print(f"\n{'='*60}")
print("FINAL EVALUATION")
print(f"{'='*60}")
test_loss, test_acc = trainer.run_one_epoch(test_dl, state='eval')
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.1%}")
print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
print(f"{'='*60}")
# File: utils.py
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights
from models.cvt import CvT
from models.vit import ViT
from models.coatnet import CoAtNet
from models.resnet import get_model as get_resnet_cifar
from config import ViTConfig, CoAtNetConfig, CvTConfig

import random
import numpy as np


def parse_args():
	parser = argparse.ArgumentParser(description="Train CvT / ResNet / DeiT on TinyImageNet")

	# General
	parser.add_argument("--data-root", type=str, default="./tiny-imagenet-200", help="Path to TinyImageNet dataset")
	parser.add_argument("--download", action="store_true", help="Download dataset if not found")
	parser.add_argument("--save-path", type=str, default="checkpoints", help="Path to save the trained model checkpoint")
	parser.add_argument("--resume", type=str, default="", help="Path to resume from checkpoint")
	parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', or 'cuda'")

	# Model
	parser.add_argument("--model", type=str, default="vit", choices=["vit", "cvt", "resnet18", "deit"], help="Model type")
	parser.add_argument("--num-classes", type=int, default=100, help="Number of classes")

	# Optimizer
	parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
	parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
	parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")

	# Scheduler
	parser.add_argument(
		"--scheduler", type=str, default="cosineannealing",
		choices=["cosineannealing", "reduceonplateau", "onecycle"],
		help="Learning rate scheduler",
  )
	parser.add_argument(
		"--warmup-steps", type=int, default=0,
		help="Number of warmup steps before main scheduler kicks in"
	)

	# Training
	parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
	parser.add_argument("--transfer-epochs", type=int, default=20, help="Number of epochs for transfer learning (head only)")
	parser.add_argument("--finetune-epochs", type=int, default=80, help="Number of epochs for fine-tuning (full model)")
	parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
	parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for CE loss")

	# Distillation
	parser.add_argument("--distillation", action="store_true", help="Enable knowledge distillation")
	parser.add_argument("--teacher-model", type=str, default="resnet20", choices=["resnet18", "resnet20", "resnet32"])
	parser.add_argument("--teacher-path", type=str, default="", help="Path to teacher checkpoint")
	parser.add_argument("--alpha", type=float, default=0.5, help="Weight for distillation loss")
	parser.add_argument("--tau", type=float, default=1.0, help="Temperature for distillation")

	return parser.parse_args()

def get_config(model_name: str, args=None):
	if model_name.lower() == "cvt":
		return CvTConfig()
	elif model_name.lower() == "vit":
		return ViTConfig()
	elif model_name.lower() == "coatnet":
		return CoAtNetConfig()
	elif model_name.lower() == "resnet18":
		return None
	elif model_name.lower() == "resnet20":
		return None
	else:
		raise ValueError(f"Unknown model {model_name}")

def get_model(model_name: str, config=None, num_classes=100):
	model_name = model_name.lower()

	if model_name == "cvt":
		return CvT(config)
	elif model_name == "vit":
		return ViT(config)
	elif model_name == "coatnet":
		return CoAtNet(config)
	elif model_name == "resnet18":
		model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
		model.fc = nn.Linear(model.fc.in_features, num_classes)

		return model
	elif model_name == "resnet20" or model_name == "resnet32":
		model = get_resnet_cifar(model_name, num_classes=num_classes)

		return model
	else:
		raise ValueError(f"Unknown model {model_name}")

def get_param_groups(model, weight_decay):
	decay, no_decay = [], []

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue

		if (
			name.endswith("bias")
			or "norm" in name.lower()
			or "layerscale" in name.lower()
		):
			no_decay.append(param)
		else:
			decay.append(param)

	return [
		{"params": decay, "weight_decay": weight_decay},
		{"params": no_decay, "weight_decay": 0.0},
	]

def set_seed(seed=0):
	"""Sets the seed for reproducibility."""
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # For multi-GPU setups

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def load_teacher_model(path, model_name, num_classes, device):
	checkpoint = torch.load(path, map_location=device)
	config = checkpoint.get('config')
	
	model = get_model(model_name, config, num_classes=num_classes)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.to(device)
	model.eval()
	
	return model

