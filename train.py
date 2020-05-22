from my_packages.SuperResolution import SuperResolutionModule
from my_packages.VOSProjection import VOSProjectionModule
from tensorboardX import SummaryWriter
from utils.frame_utils import *
from utils.flow_utils import *
from utils import tools
from utils.tools import *
import argparse, torch
import colorama, os
import torch.nn as nn
from glob import glob
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--starg_epoch', type=int, default=1)
    parser.add_argument('--total_epoch', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_n_batches', type=int, default=100)

    parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
    parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')
    parser.add_argument('--model_name', '-s', default='SRmodel', type=str)

    parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
    parser.add_argument('--validation_n_batches', type=int, default=-1)
    parser.add_argument('--render_validation', action='store_true', help='run inference (save flows to file) and every validation_frequency epoch')

    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")

    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--skip_validation', action='store_true')

    with tools.TimerBlock("Parsing Arguments") as block:
        args = parser.parse_args()
        if args.number_gpus < 0 : args.number_gpus = torch.cuda.device_count()

        parser.add_argument('--IGNORE',  action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))

        # Print all arguments, color the non-defaults
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.cuda = not args.no_cuda and torch.cuda.is_available()

        if args.inference:
            args.skip_validation = True
            args.skip_training = True
            args.total_epochs = 1
            args.inference_dir = "{}/inference".format(args.save)

    with tools.TimerBlock("Building {} model".format(args.model_name)) as block:
        SRmodel = SuperResolutionModule()
        if args.cuda and args.number_gpus > 1:
            block.log('Parallelizing')
            SRmodel = nn.parallel.DataParallel(SRmodel, device_ids=list(range(args.number_gpus)))
            block.log('Initializing CUDA')
            SRmodel = SRmodel.cuda()

        elif args.cuda and args.number_gpus > 0:
            block.log('Initializing CUDA')
            SRmodel = SRmodel.cuda

        else:
            block.log("CUDA not being used")

        torch.cuda.manual_seed(args.seed)

        if args.resume and os.path.isfile(args.resume):
            block.log("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if not args.inference:
                args.start_epoch = checkpoint['epoch']
            SRmodel.load_state_dict(checkpoint['state_dict'])
            block.log("Loaded checkpoint '{}' (at epoch {})".format(args.resume, checkpoint['epoch']))

        elif args.resume and args.inference:
            block.log("No checkpoint found at '{}'".format(args.resume))
            quit()

        else:
            block.log("Random initialization")

        block.log("Initializing save directory: {}".format(args.save))
        if not os.path.exists(args.save):
            os.makedirs(args.save)

        train_logger = SummaryWriter(log_dir = os.path.join(args.save, 'train'), comment = 'training')
        validation_logger = SummaryWriter(log_dir = os.path.join(args.save, 'validation'), comment = 'validation')

    with tools.TimerBlock("Initializing {} Optimizer".format(args.optimizer)) as block:
        if args.resume and os.path.isfile(args.resume):
            optimizer = checkpoint['optimizer']
        else:
            optimizer = torch.optim.Adam(SRmodel.parameters())

    for argument, value in sorted(vars(args).items()):
        block.log2file(args.log_file, '{}: {}'.format(argument, value))

    def train():
        return

    def inference():
        return

'''need to make dataset util and dataloader.'''
