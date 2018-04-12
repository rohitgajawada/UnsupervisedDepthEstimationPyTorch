from torch.autograd import Variable
from utils import AverageMeter
from utils import accuracy
from utils import precision
import torch.nn as nn
import utils
import time
from inverse_warp import inverse_warp
from losses import photometric_reconstruction_loss, explainability_loss, smooth_loss, compute_errors
import custom_transforms

class Trainer():
    def __init__(self, disp_model, pose_model, optimizer, opt):
        self.disp_model = disp_model
        self.pose_model = pose_model
        self.optimizer = optimizer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()

    def train(self, trainloader, epoch, opt):
        self.losses.reset()
        self.data_time.reset()
        self.batch_time.reset()
        end = time.time()
        self.disp_model.train()
        self.pose_model.train()
        for i, data in enumerate(trainloader, 0):
            self.optimizer.zero_grad()
            if opt.cuda:
                target_imgs, ref_imgs, intrinsics, intrinsics_inv = data
                target_imgs = Variable(target_imgs.cuda(async=True))
                ref_imgs = [Variable(img.cuda(async=True)) for img in ref_imgs]
                intrinsics = Variable(intrinsics.cuda(async=True))
                intrinsics_inv = Variable(intrinsics_inv.cuda(async=True))

            self.data_time.update(time.time() - end)
            disparities = self.disp_model(target_imgs)
            depths = [1 / disp for disp in disparities]
            explainability_mask, pose = self.pose_model(target_imgs, ref_imgs)

            photoloss = photometric_reconstruction_loss(target_imgs, ref_imgs, intrinsics, intrinsics_inv, depths, explainability_mask, pose, opt.rotation_mode, opt.padding_mode)
            exploss = explainability_loss(explainability_mask)
            smoothloss = smooth_loss(disparities)
            totalloss = opt.p * photoloss + opt.e * exploss + opt.s * smoothloss

            totalloss.backward()
            self.optimizer.step()

            inputs_size = intrinsics.size(0)
            self.losses.update(totalloss.data[0], inputs_size)
            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.printfreq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f} ({batch_time.sum:.3f})\t'
                      'Data {data_time.avg:.3f} ({data_time.sum:.3f})\t'
                      'Loss {loss.avg:.3f}\t'.format(
                       epoch, i, len(trainloader), batch_time=self.batch_time,
                       data_time= self.data_time, loss=self.losses))

        print('Train: [{0}]\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss {loss.avg:.3f}\t'.format(
               epoch, batch_time=self.batch_time,
               data_time= self.data_time, loss=self.losses))
