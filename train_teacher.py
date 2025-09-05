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

writer = SummaryWriter(f"runs/resnet18")

print("=" * 60)
print("Training Teacher Model (ResNet18 on TinyImageNet)")
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

model = get_model("resnet18", num_classes=args.num_classes).to(device)

# --- Phase 1: Transfer Learning (Training Classifier Head) ---
print("\n--- Phase 1: Transfer Learning (Training Classifier Head) ---")

# Freeze all layers
for param in model.parameters():
	param.requires_grad = False

# Unfreeze and initialize the new classifier head
for param in model.fc.parameters():
	param.requires_grad = True

nn.init.xavier_uniform_(model.fc.weight)
nn.init.zeros_(model.fc.bias)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: RESNET18")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters (head only): {trainable_params:,}")

# Optimizer for the classifier head only
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
	writer=writer,
)

if args.resume:
	print(f"\nResuming from checkpoint: {args.resume}")
	checkpoint = torch.load(args.resume, map_location=device)
	trainer.load_checkpoint(checkpoint)

if args.transfer_epochs > 0:
	print(f"\nStarting transfer learning for {args.transfer_epochs} epochs...")
	trainer.train(
		args.transfer_epochs, train_dl, val_dl,
		save_path=f"{args.save_path}/resnet18_teacher.pt",
		args=vars(args)
	)

# --- Phase 2: Fine-tuning (Training Full Model) ---
if args.finetune_epochs > 0:
	print("\n--- Phase 2: Fine-tuning (Training Full Model) ---")

	# Unfreeze all layers
	for param in model.parameters():
		param.requires_grad = True

	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Trainable parameters (full model): {trainable_params:,}")

	# New optimizer and scheduler for the full model
	param_groups = get_param_groups(model, args.weight_decay)
	optimizer = AdamW(param_groups, lr=args.lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)

	# Update trainer with new optimizer and scheduler
	trainer.optimizer = optimizer
	trainer.scheduler = scheduler

	total_epochs = args.transfer_epochs + args.finetune_epochs
	print(f"\nStarting fine-tuning for {args.finetune_epochs} epochs...")
	trainer.train(
		total_epochs, train_dl, val_dl,
		save_path=f"{args.save_path}/resnet18_teacher.pt",
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
