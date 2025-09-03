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
	
	def __init__(self, root, split='train', transform=None, download=False, test_size=0.1, val_size=0.1, random_state=0):
		self.root = os.path.expanduser(root)
		self.split = split
		self.transform = transform
		self.test_size = test_size
		self.val_size = val_size
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
		if self.split == 'val':
			self._load_original_val()
		else:
			self._load_train_test_split()
	
	def _load_original_val(self):
		"""Load the original validation set"""
		val_dir = os.path.join(self.root, self.folder_name, 'val')
		with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
			for line in f:
				parts = line.strip().split('\t')
				img_name, class_id = parts[0], parts[1]
				img_path = os.path.join(val_dir, 'images', img_name)
				self.data.append(img_path)
				self.targets.append(self.class_to_idx[class_id])
	
	def _load_train_test_split(self):
		"""Load and split the original training data into train and test"""
		all_data = []
		all_targets = []
		
		train_dir = os.path.join(self.root, self.folder_name, 'train')
		for class_folder in os.listdir(train_dir):
			class_idx = self.class_to_idx[class_folder]
			class_dir = os.path.join(train_dir, class_folder, 'images')
			for img_name in os.listdir(class_dir):
				img_path = os.path.join(class_dir, img_name)
				all_data.append(img_path)
				all_targets.append(class_idx)
		
		train_val_data, test_data, train_val_targets, test_targets = train_test_split(
			all_data, all_targets, 
			test_size=self.test_size, 
			random_state=self.random_state, 
			stratify=all_targets
		)
		
		adjusted_val_size = self.val_size / (1 - self.test_size)
		train_data, val_data, train_targets, val_targets = train_test_split(
			train_val_data, train_val_targets, 
			test_size=adjusted_val_size, 
			random_state=self.random_state, 
			stratify=train_val_targets
		)
		
		if self.split == 'train':
			self.data = train_data
			self.targets = train_targets
		elif self.split == 'test':
			self.data = test_data
			self.targets = test_targets
		else:
			raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'test', or 'val'")
	
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
