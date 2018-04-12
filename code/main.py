import os
import torch
import torch.backends.cudnn as cudnn
import opts
import train
import utils
import models.__init__ as init
import models.dispnet as dispnet
import models.posenet as posenet
from datasets.validation_folders import ValidationSet
from datasets.folders import SequenceFolder
import custom_transforms

parser = opts.myargparser()

def main():
    global opt, best_prec1
    opt = parser.parse_args()
    print(opt)

    # Data loading
    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    ])
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(),
    custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    print('Loading scenes in', opt.data_dir)
    train_set = SequenceFolder(opt.data_dir, transform=train_transform,
                                seed=opt.seed, train=True,
                                sequence_length=opt.sequence_length)

    val_set = ValidationSet(opt.data_dir, transform=valid_transform)

    print(len(train_set), 'samples found')
    print(len(val_set), 'samples found')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                                                shuffle=True, num_workers=opt.workers,
                                                pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=opt.batch_size,
    #                                             shuffle=False, num_workers=opt.workers,
    #                                             pin_memory=True)
    if opt.epoch == 0:
        opt.epoch_size = len(train_loader)
    # Done loading

    disp_model = dispnet.DispNet().cuda()
    pose_model = posenet.PoseNet().cuda()
    disp_model, pose_model, optimizer = init.setup(disp_model, pose_model, opt)
    print(disp_model, pose_model)
    trainer = train.Trainer(disp_model, pose_model, optimizer, opt)
    if opt.resume:
        if os.path.isfile(opt.resume):
            disp_model, pose_model, optimizer, opt, best_prec1 = init.resumer(opt, disp_model, pose_model, optimizer)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    cudnn.benchmark = True
    for epoch in range(opt.start_epoch, opt.epochs):
        utils.adjust_learning_rate(opt, optimizer, epoch)
        print("Starting epoch number:",epoch+1,"Learning rate:", optimizer.param_groups[0]["lr"])
        if opt.testOnly == False:
            trainer.train(train_loader, epoch, opt)
        init.save_checkpoint(opt, disp_model, pose_model, optimizer, best_prec1, epoch)

if __name__ == '__main__':
    main()
