from network.video_super_resolution import VSR
from utils.frame_utils import *
from utils.video_utils import *
from utils import tools
import argparse, torch
import colorama, os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.nn.functional import interpolate
import warnings
warnings.filterwarnings("ignore")

def ArgmentsParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_n_batches', type=int, default=100)

    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save', default='./work', type=str, help='directory for saving')
    parser.add_argument('--model_name', default='SRmodel', type=str)

    parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
    parser.add_argument('--validation_n_batches', type=int, default=-1)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--skip_validation', action='store_true')

    parser.add_argument('--training_dataset_root', type=str)
    parser.add_argument('--validation_dataset_root', type=str)

    with tools.TimerBlock("Parsing Arguments") as block:
        args = parser.parse_args()
        if args.number_gpus < 0: args.number_gpus = torch.cuda.device_count()

        parser.add_argument('--IGNORE', action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))

        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def InitalizingTrainingAndTestDataset(args):

    def InitalizingTrainingDataset(block):
        if exists(args.training_dataset_root):
            effective_batch_size = args.batch_size * args.number_gpus
            train_dataset = VideoDataset(args.training_dataset_root)
            block.log('Training Dataset: {}'.format(args.training_dataset_root))
            block.log('Training Input: {}'.format(np.array(train_dataset[0][0]).shape))
            block.log('Training Targets: {}'.format(train_dataset[0][0][1].shape))
            train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
            return train_loader

    def InitalizingValidationDataset(block):
        if exists(args.validation_dataset_root):
            effective_batch_size = args.batch_size * args.number_gpus
            validation_dataset = VideoDataset(args.validation_dataset_root)
            block.log('Validataion Dataset: {}'.format(args.validation_dataset_root))
            block.log('Validataion Input: {}'.format(np.array(validation_dataset[0][0]).shape))
            block.log('Validataion Targets: {}'.format(validation_dataset[0][0][1].shape))
            validation_loader = DataLoader(validation_dataset, batch_size=effective_batch_size, shuffle=False)
            return validation_loader

    with tools.TimerBlock("Initializing Datasets") as block:
        train_loader = InitalizingTrainingDataset(block)
        validation_loader = InitalizingValidationDataset(block)

    return train_loader, validation_loader

def BuildMainModelAndOptimizer(args):

    def BuildMainModel(args, block):
        block.log('Building Model')
        SRmodel = VSR()
        if args.cuda and args.number_gpus > 1:
            block.log('Parallelizing')
            SRmodel = nn.parallel.DataParallel(SRmodel, device_ids=list(range(args.number_gpus)))
            block.log('Initializing CUDA')
            SRmodel = SRmodel.cuda()

        elif args.cuda and args.number_gpus > 0:
            block.log('Initializing CUDA')
            SRmodel = SRmodel.cuda()
        else:
            block.log("CUDA not being used")

        return SRmodel

    def InitializingCheckpoint(args):
        if args.resume and os.path.isfile(args.resume):
            block.log("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
        else:
            checkpoint = False
        return checkpoint

    def LoadModelFromCheckpoint(SRmodel, checkpoint, args, block):
        if checkpoint:
            SRmodel.model.load_state_dict(checkpoint['state_dict'])
            block.log("Loaded checkpoint '{}' (at epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            block.log("Random initialization")
        return SRmodel

    def InitializingSaveDirectory(args, block):
        block.log("Initializing save directory: {}".format(args.save))
        if not os.path.exists(args.save):
            os.makedirs(args.save)

    def BuildOptimizer(checkpoint, args, block):
        if checkpoint:
            optimizer = checkpoint['optimizer']
            block.log("Loaded checkpoint '{}'".format(args.resume))
        else:
            optimizer = torch.optim.Adam(SRmodel.parameters())
            block.log("Random initialization")
        return optimizer

    with tools.TimerBlock("Building {} model".format(args.model_name)) as block:
        SRmodel = BuildMainModel(args, block)
        torch.cuda.manual_seed(args.seed)
        checkpoint = InitializingCheckpoint(args)
        SRmodel = LoadModelFromCheckpoint(SRmodel, checkpoint, args, block)
        InitializingSaveDirectory(args, block)

    with tools.TimerBlock("Initializing Optimizer") as block:
        optimizer = BuildOptimizer(checkpoint, args, block)

    return SRmodel, optimizer

def TrainAllProgress(SRmodel, optimizer, train_loader, validation_loader, args):

    def TrainMainModel(args, epoch, data_loader, model, optimizer, is_validate=False, offset=0):
        total_loss = 0
        fakeloss = MSELoss()

        if is_validate:
            model.eval()
            title = 'Validating Epoch {}'.format(epoch)
            args.validation_n_batches = np.inf if args.validation_n_batches < 0 else args.validation_n_batches
            progress = tqdm(tools.IteratorTimer(data_loader), ncols=100,
                            total=np.minimum(len(data_loader), args.validation_n_batches), leave=True, position=offset,
                            desc=title)
        else:
            model.train()
            title = 'Training Epoch {}'.format(epoch)
            args.train_n_batches = np.inf if args.train_n_batches < 0 else args.train_n_batches
            progress = tqdm(tools.IteratorTimer(data_loader), ncols=120,
                            total=np.minimum(len(data_loader), args.train_n_batches), smoothing=.9, miniters=1,
                            leave=True, position=offset, desc=title)

        for batch_idx, datas in enumerate(progress):
            data = torch.stack([torch.stack([interpolate(d.transpose(1, 3).transpose(2, 3).type(torch.float32),
                                            (int(d.shape[1] / 4),int(d.shape[2] / 4))).transpose(1, 3).transpose(1, 2)
                                             for d in dd])
                                for dd in datas]).squeeze()
            target = torch.stack([d[1] for d in datas]).type(torch.float32)
            high_frames = torch.stack([torch.stack(d) for d in datas]).squeeze().type(torch.float32)
            if args.cuda and args.number_gpus > 0:
                data = data.cuda()
                target = target.cuda()
                high_frames = high_frames.cuda()

            estimated_image = None
            for x, y, high_frame in zip(data, target, high_frames):
                optimizer.zero_grad() if not is_validate else None
                output, losses = model(x, y, high_frame, estimated_image)
                estimated_image = output
                loss = fakeloss(output.cpu(), torch.tensor(target, dtype=torch.float32).cpu())
                # loss_val = torch.mean(losses)
                # total_loss += loss_val.item()
                # loss.data = loss_val.data
                total_loss += loss.item()

                if not is_validate:
                    loss.backward()
                    optimizer.step()

            title = '{} Epoch {}'.format('Validating' if is_validate else 'Training', epoch)
            progress.set_description(title)

            if (is_validate and (batch_idx == args.validation_n_batches)) or \
                    ((not is_validate) and (batch_idx == (args.train_n_batches))):
                progress.close()
                break

        return total_loss / float(batch_idx + 1), (batch_idx + 1)

    def SetBestErr(loss, best_err):
        if loss < best_err:
            best_err = loss
        return best_err

    best_err = 1e8
    progress = tqdm(list(range(args.start_epoch, args.total_epochs + 1)), miniters=1, ncols=100,
                    desc='Overall Progress', leave=True, position=True)
    offset = 1
    global_iteration = 0

    for epoch in progress:
        if not args.skip_validation and ((epoch - 1) % args.validation_frequency) == 0:
            validation_loss, _ = TrainMainModel(args=args, epoch=epoch - 1, data_loader=validation_loader, model=SRmodel,
                                       optimizer=optimizer, is_validate=True, offset=offset)
            offset += 1

            best_err = SetBestErr(validation_loss, best_err)

            checkpoint_progress = tqdm(ncols=100, desc='Saving Checkpoint', position=offset)
            tools.save_checkpoint({'arch': args.model_name,
                                   'epoch': epoch,
                                   'state_dict': SRmodel.model.state_dict(),
                                   'best_EPE': best_err,
                                   'optimizer': optimizer},
                                  False, args.save, args.model_name)
            checkpoint_progress.update(1)
            checkpoint_progress.close()
            offset += 1

        if not args.skip_training:
            train_loss, iterations = TrainMainModel(args=args, epoch=epoch, data_loader=train_loader, model=SRmodel,
                                           optimizer=optimizer, offset=offset)
            global_iteration += iterations
            offset += 1

            if ((epoch - 1) % args.validation_frequency) == 0:
                checkpoint_progress = tqdm(ncols=100, desc='Saving Checkpoint', position=offset)
                tools.save_checkpoint({'arch': args.model_name,
                                       'epoch': epoch,
                                       'state_dict': SRmodel.model.state_dict(),
                                       'best_EPE': train_loss},
                                      False, args.save, args.model_name, filename='train-checkpoint.pth.tar')
                checkpoint_progress.update(1)
                checkpoint_progress.close()

def main():
    args = ArgmentsParser()
    train_loader, validation_loader = InitalizingTrainingAndTestDataset(args)
    SRmodel, optimizer = BuildMainModelAndOptimizer(args)
    TrainAllProgress(SRmodel, optimizer, train_loader, validation_loader, args)

if __name__ == '__main__':
    main()