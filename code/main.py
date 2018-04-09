import os
import torch.backends.cudnn as cudnn
import opts
import train
import utils
import models.__init__ as init
import datasets.__datainit__ as init_data
import models.dispnet as dispnet
import models.posenet as posenet

parser = opts.myargparser()

def main():
    global opt, best_prec1
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
