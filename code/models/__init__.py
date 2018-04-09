import torch
import torch.nn as nn
import torch.optim as optim
import utils
import os
from itertools import chain

def setup(disp_model, pose_model, opt):

    parameters = chain(disp_model.parameters(), pose_model.parameters())
    if opt.optimType == 'sgd':
        optimizer = optim.SGD(parameters, lr = opt.lr, momentum = opt.momentum, nesterov = opt.nesterov, weight_decay = opt.weightDecay)
    elif opt.optimType == 'adam':
        optimizer = optim.Adam(parameters, lr = opt.maxlr, weight_decay = opt.weightDecay)

    if opt.weight_init:
        utils.weights_init(disp_model, opt)
        utils.weights_init(pose_model, opt)

    return disp_model, pose_model, optimizer

def save_checkpoint(opt, model, optimizer, best_acc, epoch):
    state = {
        'epoch': epoch + 1,
        'arch': opt.model_def,
        'state_dict': model.state_dict(),
        'best_prec1': best_acc,
        'optimizer' : optimizer.state_dict(),
    }
    filename = "savedmodels/" + opt.name + ".pth.tar"
    torch.save(state, filename)

def resumer(opt, model, optimizer):
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))
        return model, optimizer, opt, best_prec1
