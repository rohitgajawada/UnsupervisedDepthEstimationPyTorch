import torch
import torch.nn as nn
import torch.optim as optim
import models.stereonet as stereonet

import utils
import os

"""
Initialize and return the model, criterion and the optimizer
"""
def setup(model, opt):

    if opt.criterion == "nllLoss":
        criterion = nn.NLLLoss().cuda()

    if opt.optimType == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = opt.lr, momentum = opt.momentum, nesterov = opt.nesterov, weight_decay = opt.weightDecay)
    elif opt.optimType == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = opt.maxlr, weight_decay = opt.weightDecay)

    if opt.weight_init:
        utils.weights_init(model, opt)

    return model, criterion, optimizer

"""
Save the current state as a checkpoint
"""
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

"""
Resume from a given checkpoint
"""
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

"""
Load a model from the specified opts
"""
def load_model(opt):
    if opt.pretrained_file != "":
        model = torch.load(opt.pretrained_file)
    else:
        if opt.model_def == 'stereonet':
            model = stereonet.Net()
            if opt.cuda:
                model = model.cuda()


    return model
