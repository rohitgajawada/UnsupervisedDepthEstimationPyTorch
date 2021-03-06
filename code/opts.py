import argparse
# Reporting
reporttype_choices = ['acc', 'error']
# Optimizer choices
optim_choices = ['sgd','adam','adagrad', 'adamax', 'adadelta']
# Learning rate schedules
lrscheduler_choices = ['decayscheduler', 'imagenetscheduler']

def myargparser():
    parser = argparse.ArgumentParser(description='PyTorch Core Training')

    parser.add_argument('--dataset', type=str, default='kitti', help='chosen dataset')
    parser.add_argument('--data_dir', type=str, default='../data/new_clean/', help='chosen data directory')
    parser.add_argument('--dataset-format', default='sequential', metavar='STR')
    parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')

    parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler', help='rotation mode for PoseNet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
    parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros', help='padding mode for image warping')
    parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation')

    parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1.0)
    parser.add_argument('-e', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0.2)
    parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)

    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--epoch', default=0, type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', default=4, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--testbatchsize', default=4, type=int, help='input batch size for testing (default: 1000)')
    parser.add_argument('--printfreq', default=100, type=int, help='print frequency (default: 10)')
    parser.add_argument('--learningratescheduler', default='decayscheduler', type=str, help='LR Scheduler. Options:'+str(lrscheduler_choices))

    parser.add_argument('--decayinterval', default=10, type=int, help='decays by a power of decay_var in these epochs')
    parser.add_argument('--decaylevel', default=1.15, type=int, help='decays by a power of decaylevel')
    parser.add_argument('--optimType', default='adam', choices=optim_choices, type=str, help='Optimizers. Options:'+str(optim_choices))

    parser.add_argument('--maxlr', default=2e-4, type=float, help='initial learning rate')
    parser.add_argument('--lr', type=float, help='initial learning rate')
    parser.add_argument('--minlr', default=2e-6, type=float, help='initial learning rate')

    parser.add_argument('--nesterov', action='store_true', help='nesterov momentum')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (Default: 0.9)')
    parser.add_argument('--weightDecay', default=0, type=float, help='weight decay (Default: 1e-4)')

    parser.add_argument('--inpsize', default=224, type=int)
    parser.add_argument('--weight_init', action='store_false', help='Turns off weight inits')
    parser.add_argument('--name', default='test', type=str,help='name of experiment')
    parser.add_argument('--seed', default=123, type=int, help='seed for random functions, and network initialization')


    parser.add_argument('--cachemode', default=True, help='if cachemode')
    parser.add_argument('--cuda',  default=True, help='if cuda is available')
    parser.add_argument('--manualSeed',  default=123, help='fixed seed for experiments')
    parser.add_argument('--ngpus',  default=1, help='no. of gpus')
    parser.add_argument('--logdir',  type=str, default='../logs/', help='log directory')
    parser.add_argument('--tensorboard',help='Log progress to TensorBoard', default=True)
    parser.add_argument('--testOnly', default=False, type=bool, help='run on validation set only')
    parser.add_argument('--start-epoch', default=0, type=int,help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

    parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained DispNet model')
    parser.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH', help='path to pre-trained PoseNet model')

    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--store', default='', type=str,
                        help='path to storing checkpoints (default: none)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')


    return parser
