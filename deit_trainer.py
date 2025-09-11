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
