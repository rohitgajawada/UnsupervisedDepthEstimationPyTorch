from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import math
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io as sio

class KITTIDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, opt, root_dir, transform=None):
        """
        Args:
            opt (string): Specify train/ test.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.opt = opt
        self.root_dir = root_dir
        self.transform = transform
        self.images1 = pandas.something # TODO
        self.images2 = pandas.something #TODO

    def __getitem__(self, idx)
        img1_name = os.path.join(self.root_dir, self.opt,
                                'image_2', self.images1.iloc[idx, 0])
        img2_name = os.path.join(self.root_dir, self.opt,
        						'image_3', self.images2.iloc[idx, 0])
        image1 = io.imread(img1_name)
        image2 = io.imread(img2_name)
        sample = {'image1': image1, 'image2': image2}

        if self.transform:
            sample = self.transform(sample)

        return sample