import os
import colorama
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import MSELoss
from torch.nn.functional import interpolate
from network.video_super_resolution import VSR
from utils import tools
from utils.tools import MakeCuda, transpose1312, transpose1323
from utils.video_utils import VideoDataset


def ArgmentsParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_n_batches', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--load_optimizer', action='store_true')
    parser.add_argument('--load_lr', action='store_true')

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

    parser.add_argument('--ignore_warning', action='store_true')

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
        if args.cuda and args.number_gpus > 0:
            args.cuda_available = True
        else:
            args.cuda_available = False

    if args.ignore_warning:
        import warnings
        warnings.filterwarnings(action='ignore')

    return args


def InitalizingTrainingAndTestDataset(args):
    def InitalizingTrainingDataset(block):
        if os.path.exists(args.training_dataset_root) and not args.skip_training:
            train_dataset = VideoDataset(args.training_dataset_root)
            block.log('Training Dataset: {}'.format(args.training_dataset_root))
            block.log('Training Input: {}'.format(np.array(train_dataset[0][0]).shape))
            block.log('Training Targets: {}'.format(train_dataset[0][0][1].shape))
            return train_dataset

    def InitalizingValidationDataset(block):
        if os.path.exists(args.validation_dataset_root) and not args.skip_validation:
            validation_dataset = VideoDataset(args.validation_dataset_root)
            block.log('Validataion Dataset: {}'.format(args.validation_dataset_root))
            block.log('Validataion Input: {}'.format(np.array(validation_dataset[0][0]).shape))
            block.log('Validataion Targets: {}'.format(validation_dataset[0][0][1].shape))
            return validation_dataset

    with tools.TimerBlock("Initializing Datasets") as block:
        train_dataset = InitalizingTrainingDataset(block)
        validation_dataset = InitalizingValidationDataset(block)

    return train_dataset, validation_dataset


def BuildMainModelAndOptimizer(args):
    def BuildMainModel(args, block):
        block.log('Building Model')
        SRmodel = VSR()
        if args.cuda_available:
            block.log('Initializing CUDA')
            SRmodel = MakeCuda(SRmodel)
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
        if checkpoint and args.load_optimizer:
            optimizer = checkpoint['optimizer']
            if not args.load_lr:
                optimizer.param_groups[0]['lr'] = args.lr
                block.log("Set learning rate '{}'".format(args.lr))
            block.log("Loaded checkpoint '{}'".format(args.resume))
        else:
            optimizer = torch.optim.Adam(SRmodel.parameters(), lr=args.lr)
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


def TrainAllProgress(SRmodel, optimizer, train_dataset, validation_dataset, args):
    def MakeDataDatasetToTensor(datas):
        data = torch.stack([transpose1312(
            interpolate(transpose1323(d.type(torch.float32)), (int(d.shape[1] / 4), int(d.shape[2] / 4)))) for d in
                            datas])
        return data

    def MakeTargetDatasetToTensor(datas):
        target = datas[:, 1:2].type(torch.float32)
        return target

    def MakeHFDatasetToTensor(datas):
        datas = datas.type(torch.float32)
        return datas

    def TrainMainModel(args, dataset, model, optimizer, is_validate=False):

        fakeloss = MSELoss()

        if is_validate:
            model.eval()
            args.validation_n_batches = np.inf if args.validation_n_batches < 0 else args.validation_n_batches

        else:
            model.train()
            args.train_n_batches = np.inf if args.train_n_batches < 0 else args.train_n_batches

        total_loss = []
        progress = tqdm(list(range(0, len(dataset))), miniters=1, ncols=100,
                        desc='Overall Progress', leave=True, position=True)

        for batch_idx in progress:
            datas = torch.tensor(dataset[batch_idx])
            data = MakeDataDatasetToTensor(datas)
            target = MakeTargetDatasetToTensor(datas)
            high_frames = MakeHFDatasetToTensor(datas)

            if args.cuda_available:
                data = MakeCuda(data)
                target = MakeCuda(target)
                high_frames = MakeCuda(high_frames)

            estimated_image = None
            optimizer.zero_grad() if not is_validate else None

            for x, y, high_frame in zip(data, target, high_frames):
                with torch.no_grad():
                    output, real_loss = model(x, y, high_frame, estimated_image)
                    estimated_image = output
                    total_loss.append(real_loss.data)

            if not is_validate and batch_idx % dataset.splitvideonum == 0 and batch_idx > 0:
                output, real_loss = model(x, y, high_frame, estimated_image)
                loss = fakeloss(output.cpu(), torch.tensor(target, dtype=torch.float32).cpu())
                loss.data = sum(total_loss) / len(total_loss)
                loss.backward()
                optimizer.step()
                print(loss)

                total_loss = []

            if (is_validate and (batch_idx == args.validation_n_batches)) or \
                    ((not is_validate) and (batch_idx == (args.train_n_batches))):
                break

        progress.close()

        return sum(total_loss) / float(batch_idx + 1), (batch_idx + 1)

    progress = tqdm(list(range(args.start_epoch, args.total_epochs + 1)), miniters=1, ncols=100,
                    desc='Overall Progress', leave=True, position=True)
    global_iteration = 0

    for epoch in progress:
        if not args.skip_validation and ((epoch - 1) % args.validation_frequency) == 0:
            validation_loss, _ = TrainMainModel(args=args, dataset=validation_dataset, model=SRmodel,
                                                optimizer=optimizer, is_validate=True)

            checkpoint_progress = tqdm(ncols=100, desc='Saving Checkpoint')
            tools.save_checkpoint({'arch': args.model_name,
                                   'epoch': epoch,
                                   'state_dict': SRmodel.model.state_dict(),
                                   'optimizer': optimizer},
                                  False, args.save, args.model_name)
            checkpoint_progress.update(1)
            checkpoint_progress.close()

        if not args.skip_training:
            train_loss, iterations = TrainMainModel(args=args, dataset=train_dataset, model=SRmodel,
                                                    optimizer=optimizer)
            global_iteration += iterations

            if ((epoch - 1) % args.validation_frequency) == 0:
                checkpoint_progress = tqdm(ncols=100, desc='Saving Checkpoint')
                tools.save_checkpoint({'arch': args.model_name,
                                       'epoch': epoch,
                                       'state_dict': SRmodel.model.state_dict(),
                                       'optimizer': optimizer},
                                      False, args.save, args.model_name, filename='train-checkpoint.pth.tar')
                checkpoint_progress.update(1)
                checkpoint_progress.close()


def main():
    args = ArgmentsParser()
    train_dataset, validation_dataset = InitalizingTrainingAndTestDataset(args)
    SRmodel, optimizer = BuildMainModelAndOptimizer(args)
    TrainAllProgress(SRmodel, optimizer, train_dataset, validation_dataset, args)


if __name__ == '__main__':
    main()