import numpy as np
from scipy.misc import imread
import random
from path import Path
import torch.utils.data

def load_as_float(path):
    return imread(path).astype(np.float32)

class SequenceFolder(torch.utils.data.Dataset):
    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        random.seed(seed)
        np.random.seed(seed)
        self.root = Path(root)
        if train:
            list_path = self.root/'train.txt'
        else:
            list_path = self.root/'val.txt'
        self.scenes = [self.root/direc[:-1] for direc in open(list_path)]
        self.transform = transform
        self.crawler(sequence_length)

    def crawler(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length - 1)//2
        shifts = range(-demi_length, demi_length + 1)
        shifts = list(shifts)
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt', delimiter=' ').astype(np.float32)
            intrinsics = intrinsics.reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs) - demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i + j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = imread(sample['tgt']).astype(np.float32)
        ref_imgs = []
        for ref_img in sample['ref_imgs']:
            ref_imgs.append(imread(ref_img).astype(np.float32))
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
