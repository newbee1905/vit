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
