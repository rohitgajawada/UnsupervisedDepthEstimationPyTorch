from torch.autograd import Variable
from utils import AverageMeter
from utils import accuracy
from utils import precision
from copy import deepcopy
import torch.nn as nn
import utils
import math
import time

class Trainer():
    """
    Defines a trainer class that is used to train and evaluate models on various datasets
    """
    def __init__(self, model, criterion, optimizer, opt, logger):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.acc = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

    def train(self, trainloader, epoch, opt):
        """
        Trains the specified model for a single epoch on the training data
        """
        self.model.train()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()
        self.acc.reset()
        self.data_time.reset()
        self.batch_time.reset()

        end = time.time()
        for i, data in enumerate(trainloader, 0):

            self.optimizer.zero_grad()

            if opt.cuda:
                inputs, targets = data
                inputs = inputs.cuda(async=True)
                targets = targets.cuda(async=True)

            inputs, targets = Variable(inputs), Variable(targets)
            self.data_time.update(time.time() - end)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            prec1, prec5 = precision(outputs.data, targets.data, topk=(1,5))
            acc = accuracy(outputs.data.max(1)[1], targets.data, opt)
            prec1, prec5 = prec1[0], prec5[0]

            loss.backward()
            self.optimizer.step()

            inputs_size = inputs.size(0)
            self.losses.update(loss.data[0], inputs_size)
            self.acc.update(acc, inputs_size)
            self.top1.update(prec1, inputs_size)
            self.top5.update(prec5, inputs_size)

            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.printfreq == 0 and opt.verbose == True:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f} ({batch_time.sum:.3f})\t'
                      'Data {data_time.avg:.3f} ({data_time.sum:.3f})\t'
                      'Loss {loss.avg:.3f}\t'
                      'Accuracy {acc.avg:.4f}\t'
                      'Prec@1 {top1.avg:.4f}\t'
                      'Prec@5 {top5.avg:.4f}'.format(
                       epoch, i, len(trainloader), batch_time=self.batch_time,
                       data_time= self.data_time, loss=self.losses,acc=self.acc,
                       top1=self.top1, top5=self.top5))

        # log to TensorBoard
        #if opt.tensorboard:
            #self.logger.scalar_summary('train_loss', self.losses.avg, epoch)
            #self.logger.scalar_summary('train_acc', self.top1.avg, epoch)

        print('Train: [{0}]\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss {loss.avg:.3f}\t'
              'Accuracy {acc.avg:.4f}\t'
              'Prec@1 {top1.avg:.4f}\t'
              'Prec@5 {top5.avg:.4f}\t'.format(
               epoch, batch_time=self.batch_time,
               data_time= self.data_time, loss=self.losses,
               acc=self.acc, top1=self.top1, top5=self.top5))


class Validator():
    """
    Evaluates the specified model on the validation data
    """
    def __init__(self, model, criterion, opt, logger):

        self.model = model
        self.criterion = criterion
        self.logger = logger
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.acc = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

    def validate(self, valloader, epoch, opt):
        """
        Validates the specified model on the validation data
        """
        self.model.eval()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()
        self.acc.reset()
        self.data_time.reset()
        self.batch_time.reset()
        end = time.time()

        for i, data in enumerate(valloader, 0):

            if opt.cuda:
                inputs, targets = data
                inputs = inputs.cuda(async=True)
                targets = targets.cuda(async=True)

            inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)

            self.data_time.update(time.time() - end)
            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)
            acc = accuracy(outputs.data.max(1)[1], targets.data, opt)
            self.losses.update(loss.data[0], inputs[0].size(0))
            prec1, prec5 = precision(outputs.data, targets.data, topk=(1,5))
            prec1, prec5 = prec1[0], prec5[0]
            inputs_size = inputs.size(0)
            acc = accuracy(outputs.data.max(1)[1], targets.data, opt)

            self.acc.update(acc, inputs_size)
            self.top1.update(prec1, inputs_size)
            self.top5.update(prec5, inputs_size)

            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.printfreq == 0 and opt.verbose == True:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                       epoch, i, len(valloader), batch_time=self.batch_time,
                       data_time= self.data_time, loss=self.losses,
                       acc=self.acc, top1=self.top1, top5=self.top5))

        finalacc = self.acc.avg

        #if opt.tensorboard:
            #self.logger.scalar_summary('val_loss', self.losses.avg, epoch)
            #self.logger.scalar_summary('val_acc', self.top1.avg, epoch)

        print('Val: [{0}]\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss {loss.avg:.3f}\t'
              'Accuracy {acc:.4f}\t'
              'Prec@1 {top1.avg:.4f}\t'
              'Prec@5 {top5.avg:.4f}\t'.format(
               epoch, batch_time=self.batch_time,
               data_time= self.data_time, loss=self.losses,
               acc=finalacc, top1=self.top1, top5=self.top5))

        return self.top1.avg
