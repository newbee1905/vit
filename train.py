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
