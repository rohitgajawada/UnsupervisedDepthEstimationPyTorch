import datasets.load_data as ld

def load_data(opt):
	'''
	Loads the dataset file from load_data.py file.
	'''

	if opt.dataset == "cifar10":
		dataloader = ld.LoadCIFAR10(opt)

	elif opt.dataset == "kitti":
		dataloader = ld.LoadKITTI(opt)

	return dataloader
