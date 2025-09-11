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
			query = self.q_norm(query)
			k_sampled = self.k_norm(k_sampled)

		attn_output = F.scaled_dot_product_attention(
			query, k_sampled, v_sampled,
			dropout_p=self.dropout.p if self.training else 0.0,
			is_causal=False,
		)
		
		attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
		output = self.out_proj(attn_output)

		return output, self.lambda_reg * regularization_loss
