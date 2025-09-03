import numpy as np

from tqdm import tqdm, trange
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import v2
from torchvision.transforms import transforms
from torchvision.datasets.mnist import MNIST

from einops import rearrange

from vit import ViT
from config import ViTConfig

np.random.seed(0)
torch.manual_seed(0)

def main():
	train_transform = v2.Compose([
		v2.ToImage(),
		v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
		v2.RandomRotation(degrees=10),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize((0.1307,), (0.3081,))
	])
	test_transform = v2.Compose([
		v2.ToImage(),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize((0.1307,), (0.3081,))
	])

	train_ds = MNIST(root='./datasets', train=True, download=True, transform=train_transform)
	test_ds = MNIST(root='./datasets', train=False, download=True, transform=test_transform)

	train_dl = DataLoader(train_ds, shuffle=True, batch_size=128)
	test_dl = DataLoader(test_ds, shuffle=False, batch_size=128)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
	model = ViT(ViTConfig()).to(device)

	N_EPOCHS = 15
	LR = 0.005

	matrix_params = list(p for p in model.blocks.parameters() if p.ndim == 2)
	vector_params = list(p for p in model.blocks.parameters() if p.ndim != 2)

	embed_params  = list(model.linear_mapper.parameters())
	lm_head_params= list(model.mlp.parameters())

	param_groups = [
		dict(params=matrix_params),
		dict(params=vector_params, algorithm="lion"),
		dict(params=embed_params, algorithm="lion"),
		dict(params=lm_head_params, algorithm="lion", lr=LR / math.sqrt(model.d_model))
	]

	main_params = []
	no_decay = []
	for name, p in model.named_parameters():
		if "bias" in name or "norm" in name:
			no_decay.append(p)
		else:
			main_params.append(p)

	optim = AdamW([
		{"params": main_params, "lr": LR, "weight_decay": 0.1},
		{"params": no_decay, "lr": LR, "weight_decay": 0.0},
	], betas=(0.9, 0.999))
	criterion = CrossEntropyLoss()

	model.train()
	for epoch in trange(N_EPOCHS, desc="Training"):
		train_loss = 0.0
		for batch in tqdm(train_dl, desc=f"Epoch {epoch + 1} in training", leave=False):
			x, y = batch
			x, y = x.to(device), y.to(device)
			y_hat = model(x)
			loss = criterion(y_hat, y)

			train_loss += loss.detach().cpu().item() / len(train_dl)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

	model.eval()
	with torch.no_grad():
		correct, total = 0, 0
		test_loss = 0.0
		for batch in tqdm(test_dl, desc="Testing"):
			x, y = batch
			x, y = x.to(device), y.to(device)
			y_hat = model(x)
			loss = criterion(y_hat, y)
			test_loss += loss.detach().cpu().item() / len(test_dl)

			correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
			total += len(x)

		print(f"Test loss: {test_loss:.2f}")
		print(f"Test accuracy: {correct / total * 100:.2f}%")

if __name__ == "__main__":
	main()
