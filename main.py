import os
import warnings
import colorama
import argparse
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.nn.functional import interpolate
from network.video_super_resolution import VSR
from utils import tools
from utils.video_utils import VideoDataset


def train(args, epoch, data_loader, model, optimizer, is_train=True, offset=0):
    """
    Main train function.

    :param args: argments
    :param epoch: Train iteration epoch
    :param data_loader: Data put in model
    :param model: Model to train
    :param optimizer: How to optimize the mdoel
    :param is_train: Evaluation or not
    :param offset: Start iteration
    :return: Model loss
    """
    total_loss = 0
    fakeloss = MSELoss()

    if is_train:
        model.train()
        title = 'Training Epoch {}'.format(epoch)
        args.train_n_batches = np.inf if args.train_n_batches < 0 else args.train_n_batches
        progress = tqdm(tools.IteratorTimer(data_loader), ncols=120,
                        total=np.minimum(len(data_loader), args.train_n_batches), smoothing=.9, miniters=1,
                        leave=True, position=offset, desc=title)
    else:
        model.eval()
        title = 'Validating Epoch {}'.format(epoch)
        args.validation_n_batches = np.inf if args.validation_n_batches < 0 else args.validation_n_batches
        progress = tqdm(tools.IteratorTimer(data_loader), ncols=100,
                        total=np.minimum(len(data_loader), args.validation_n_batches), leave=True, position=offset,
                        desc=title)

    for batch_idx, datas in enumerate(progress):
        data = torch.stack([torch.stack([interpolate(d.transpose(1, 3).transpose(2, 3).type(torch.float32),
                                                     (int(d.shape[1] / 4), int(d.shape[2] / 4))).transpose(1, 3).
                                        transpose(1, 2) for d in dd]) for dd in datas]).squeeze()
        print(data.shape)

        target = torch.stack([d[1] for d in datas]).type(torch.float32)
        high_frames = torch.stack([torch.stack(d) for d in datas]).squeeze().type(torch.float32)

        if args.cuda and args.number_gpus > 0:
            data = data.cuda()
            target = target.cuda()
            high_frames = high_frames.cuda()

        estimated_image = None
        for x, y, high_frame in zip(data, target, high_frames):
            old_state_dict = {}
            for key in model.state_dict():
                old_state_dict[key] = model.state_dict()[key].clone()
            import time
            t = time.time()
            optimizer.zero_grad() if is_train else None
            output, losses = model(x, y, high_frame, estimated_image)
            estimated_image = output
            loss = fakeloss(output.cpu(), torch.tensor(target, dtype=torch.float32).cpu())
            # loss_val = torch.mean(losses)
            # total_loss += loss_val.item()
            # loss.data = loss_val.data
            total_loss += loss.item()

            if is_train:
                loss.backward()
                optimizer.step()

                new_state_dict = {}
                for key in model.state_dict():
                    new_state_dict[key] = model.state_dict()[key].clone()

                different_par = 0
                for key in old_state_dict:
                    if not (old_state_dict[key] == new_state_dict[key]).all():
                        different_par += 1
                        print('Diff in {}'.format(key))

                if different_par == 0:
                    print('All Same')
                print(time.time() - t)

        title = '{} Epoch {}'.format('Training' if is_train else 'Validating', epoch)
        progress.set_description(title)

        if (not is_train and (batch_idx == args.validation_n_batches)) or \
                (is_train and (batch_idx == (args.train_n_batches))):
            progress.close()
            break

    return total_loss / float(batch_idx + 1), (batch_idx + 1)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
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
        if args.number_gpus < 0:
            args.number_gpus = torch.cuda.device_count()

        parser.add_argument('--IGNORE', action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))

        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.cuda = not args.no_cuda and torch.cuda.is_available()

    with tools.TimerBlock("Initializing Datasets") as block:
        args.effective_batch_size = args.batch_size * args.number_gpus

        if pos.path.exists(args.training_dataset_root):
            train_dataset = VideoDataset(args.training_dataset_root)
            block.log('Training Dataset: {}'.format(args.training_dataset_root))
            block.log('Training Input: {}'.format(np.array(train_dataset[0][0]).shape))
            block.log('Training Targets: {}'.format(train_dataset[0][0][1].shape))
            train_loader = DataLoader(train_dataset, batch_size=args.effective_batch_size, shuffle=True)

        if os.path.exists(args.validation_dataset_root):
            validation_dataset = VideoDataset(args.validation_dataset_root)
            block.log('Validataion Dataset: {}'.format(args.validation_dataset_root))
            block.log('Validataion Input: {}'.format(np.array(validation_dataset[0][0]).shape))
            block.log('Validataion Targets: {}'.format(validation_dataset[0][0][1].shape))
            validation_loader = DataLoader(validation_dataset, batch_size=args.effective_batch_size, shuffle=False)

    with tools.TimerBlock("Building {} model".format(args.model_name)) as block:
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

        torch.cuda.manual_seed(args.seed)

        if args.resume and os.path.isfile(args.resume):
            block.log("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            SRmodel.model.load_state_dict(checkpoint['state_dict'])
            block.log("Loaded checkpoint '{}' (at epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            block.log("Random initialization")

        block.log("Initializing save directory: {}".format(args.save))
        if not os.path.exists(args.save):
            os.makedirs(args.save)

    with tools.TimerBlock("Initializing Optimizer") as block:
        if args.resume and os.path.isfile(args.resume):
            optimizer = checkpoint['optimizer']
        else:
            optimizer = torch.optim.Adam(SRmodel.parameters())

    global_iteration = 0
    best_err = 1e8
    offset = 1

    progress = tqdm(list(range(args.start_epoch, args.total_epochs + 1)), miniters=1, ncols=100,
                    desc='Overall Progress', leave=True, position=True)

    for epoch in progress:
        if not args.skip_validation and ((epoch - 1) % args.validation_frequency) == 0:
            validation_loss, _ = train(args=args, epoch=epoch - 1, data_loader=validation_loader, model=SRmodel,
                                       optimizer=optimizer, is_train=False, offset=offset)
            offset += 1

            is_best = False
            if validation_loss < best_err:
                best_err = validation_loss
                is_best = True

            checkpoint_progress = tqdm(ncols=100, desc='Saving Checkpoint', position=offset)
            tools.save_checkpoint({'arch': args.model_name,
                                   'epoch': epoch,
                                   'state_dict': SRmodel.model.state_dict(),
                                   'best_EPE': best_err,
                                   'optimizer': optimizer},
                                  is_best, args.save, args.model_name)
            checkpoint_progress.update(1)
            checkpoint_progress.close()
            offset += 1

        if not args.skip_training:
            train_loss, iterations = train(args=args, epoch=epoch, data_loader=train_loader, model=SRmodel,
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
