import numpy as np
from path import Path
from scipy.misc import imread
import torch.utils.data

def crawl(folders_list):
	imgs = []
	depth = []
	for folder in folders_list:
		current_imgs = sorted(folder.files('*.jpg'))
		current_depth = []
		for img in current_imgs:
			d = img.dirname()/(img.name[:-4] + '.npy')
			assert(d.isfile()), "depth file {} not found".format(str(d))
			depth.append(d)
		imgs.extend(current_imgs)
		depth.extend(current_depth)
	return imgs, depth

class ValidationSet(torch.utils.data.Dataset):
	def __init__(self, root, transform=None):
		self.root = Path(root)
		scene_list_path = self.root/'val.txt'
		self.scenes = [self.root/direc[:-1] for direc in open(scene_list_path)]
		self.imgs, self.depth = crawl(self.scenes)
		self.transform = transform

	def __getitem__(self, index):
		img = imread(self.imgs[index]).astype(np.float32)
		depth = np.load(self.depth[index]).astype(np.float32)
		if self.transform is not None:
			img, _ = self.transform([img], None)
			img = img[0]
		return img, depth

	def __len__(self):
		return len(self.imgs)