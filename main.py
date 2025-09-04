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
