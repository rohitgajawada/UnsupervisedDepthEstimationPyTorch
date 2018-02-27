import os
import torch
from torchvision import datasets as dsets
from torchvision import transforms
import utils

"""
Dataloaders for the CIFAR-10 dataset
"""
class LoadCIFAR10():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': opt.workers,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			dsets.CIFAR10('../data', train=True, download=True,
					transform=transforms.Compose([
						transforms.RandomCrop(32, padding=4),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
std=[x/255.0 for x in [63.0, 62.1, 66.7]])
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			dsets.CIFAR10('../data', train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
std=[x/255.0 for x in [63.0, 62.1, 66.7]])
					   ])),
		  **kwargs)
