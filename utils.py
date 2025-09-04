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

