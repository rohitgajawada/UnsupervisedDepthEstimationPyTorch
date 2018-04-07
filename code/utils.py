import torch
import torch.nn as nn
from torch.nn import init
import copy
import random
import math
from PIL import Image
from torchvision import transforms

class AverageMeter():
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, opt):
    x = (output == target)
    x = sum(x)
    return x*1.0 / len(output)

def precision(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class RandomRotate(object):
    """
    Rotates images randomly in a specified range
    """
    def __init__(self, rrange):
        self.rrange = rrange

    def __call__(self, img):
        size = img.size
        angle = random.randint(-self.rrange, self.rrange)
        img = img.rotate(angle, resample=Image.BICUBIC)
        img = img.resize(size, Image.ANTIALIAS)
        return img


def adjust_learning_rate(opt, optimizer, epoch):
    """
    Adjusts the learning rate every epoch based on the selected schedule
    """
    epoch = copy.deepcopy(epoch)
    lr = opt.maxlr
    wd = opt.weightDecay
    if opt.learningratescheduler == 'imagenetscheduler':
        if epoch >= 1 and epoch <= 18:
            lr = 1e-3
            wd = 5e-5
        elif epoch >= 19 and epoch <= 29:
            lr = 5e-4
            wd = 5e-5
        elif epoch >= 30 and epoch <= 43:
            lr = 1e-4
            wd = 0
        elif epoch >= 44 and epoch <= 52:
            lr = 5e-5
            wd = 0
        elif epoch >= 53:
            lr = 2e-5
            wd = 0
        if opt.optimType=='sgd':
            lr *= 10
        opt.lr = lr
        opt.weightDecay = wd
    if opt.learningratescheduler == 'decayscheduler':
        while epoch >= opt.decayinterval:
            lr = lr/opt.decaylevel
            epoch = epoch - opt.decayinterval
        lr = max(lr,opt.minlr)
        opt.lr = lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = wd

def get_mean_and_std(dataloader):
    """
    Compute the mean and std value of dataset
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    len_dataset = 0
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        len_dataset += 1
        for i in range(len(inputs[0])):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len_dataset)
    std.div_(len_dataset)
    return mean, std

def weights_init(model, opt):
    """
    Perform weight initializations
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
