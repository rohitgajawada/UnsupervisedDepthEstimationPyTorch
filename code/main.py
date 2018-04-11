import os
import torch.backends.cudnn as cudnn
import opts
import train
import utils
import models.__init__ as init
import datasets.__datainit__ as init_data
import models.dispnet as dispnet
import models.posenet as posenet
from datasets.validation_folders import ValidationSet
from datasets.folders import SequenceFolder

parser = opts.myargparser()

def main():
    global opt, best_prec1

    # Data loading
    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    ])
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(),
                                                    custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                std=[0.5, 0.5, 0.5])])
    print('Loading scenes in', args.data)
    train_set = SequenceFolder(args.data, transform=train_transform,
                                seed=args.seed, train=True,
                                sequence_length=args.sequence_length)

    val_set = ValidationSet(args.data, transform=valid_transform)

    print(len(train_set), 'samples found')
    print(len(val_set), 'samples found')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.workers,
                                                pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.workers,
                                                pin_memory=True)
    if args.epoch == 0:
        args.epoch_size = len(train_loader)

    # Done loading

    opt = parser.parse_args()
    print(opt)
    disp_model = dispnet.Net().cuda()
    pose_model = posenet.Net().cuda()
    disp_model, pose_model, optimizer = init.setup(disp_model, pose_model, opt)
    print(disp_model, pose_model)
    trainer = train.Trainer(dispnet, posenet, optimizer, opt)
    if opt.resume:
        if os.path.isfile(opt.resume):
            disp_model, pose_model, optimizer, opt, best_prec1 = init.resumer(opt, disp_model, pose_model, optimizer)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    cudnn.benchmark = True
    dataloader = init_data.load_data(opt)
    train_loader = dataloader.train_loader

    for epoch in range(opt.start_epoch, opt.epochs):
        utils.adjust_learning_rate(opt, optimizer, epoch)
        print("Starting epoch number:",epoch+1,"Learning rate:", optimizer.param_groups[0]["lr"])
        if opt.testOnly == False:
            trainer.train(train_loader, epoch, opt)
        init.save_checkpoint(opt, disp_model, pose_model, optimizer, best_prec1, epoch)

if __name__ == '__main__':
    main()
