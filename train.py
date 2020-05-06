import os
import shutil
import time
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.optim import lr_scheduler
import tensorboard_logger

from models import build_model
from utils.utils import (train, validate, build_dataflow, get_augmentor,
                         save_checkpoint)
from utils.video_dataset import VideoDataSet
from utils.dataset_config import get_dataset_config
from opts import arg_parser


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = args.cudnn_benchmark
    args.gpu = gpu

    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(
        args.dataset)
    args.num_classes = num_classes

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

    model, arch_name = build_model(args)
    mean = model.mean(args.modality)
    std = model.std(args.modality)

    # overwrite mean and std if they are presented in command
    if args.mean is not None:
        if args.modality == 'rgb':
            if len(args.mean) != 3:
                raise ValueError("When training with rgb, dim of mean must be three.")
        elif args.modality == 'flow':
            if len(args.mean) != 1:
                raise ValueError("When training with flow, dim of mean must be three.")
        mean = args.mean

    if args.std is not None:
        if args.modality == 'rgb':
            if len(args.std) != 3:
                raise ValueError("When training with rgb, dim of std must be three.")
        elif args.modality == 'flow':
            if len(args.std) != 1:
                raise ValueError("When training with flow, dim of std must be three.")
        std = args.std

    model = model.cuda(args.gpu)
    model.eval()

    if args.threed_data:
        dummy_data = (args.input_channels, args.groups, args.input_size, args.input_size)
    else:
        dummy_data = (args.input_channels * args.groups, args.input_size, args.input_size)

    if args.rank == 0:
        torch.cuda.empty_cache()

    if args.show_model and args.rank == 0:
        print(model)
        return 0

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # the batch size should be divided by number of nodes as well
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int(args.workers / ngpus_per_node)

            if args.sync_bn:
                process_group = torch.distributed.new_group(list(range(args.world_size)))
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        # assign rank to 0
        model = torch.nn.DataParallel(model).cuda()
        args.rank = 0

    if args.pretrained is not None:
        if args.rank == 0:
            print("=> using pre-trained model '{}'".format(arch_name))
        if args.gpu is None:
            checkpoint = torch.load(args.pretrained, map_location='cpu')
        else:
            checkpoint = torch.load(args.pretrained, map_location='cuda:{}'.format(args.gpu))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:
        if args.rank == 0:
            print("=> creating model '{}'".format(arch_name))

    # define loss function (criterion) and optimizer
    train_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    val_criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # Data loading code
    val_list = os.path.join(args.datadir, val_list_name)

    val_augmentor = get_augmentor(False, args.input_size, scale_range=args.scale_range, mean=mean,
                                  std=std, disable_scaleup=args.disable_scaleup,
                                  threed_data=args.threed_data,
                                  is_flow=True if args.modality == 'flow' else False,
                                  version=args.augmentor_ver)

    val_dataset = VideoDataSet(args.datadir, val_list, args.groups, args.frames_per_group,
                               num_clips=args.num_clips,
                               modality=args.modality, image_tmpl=image_tmpl,
                               dense_sampling=args.dense_sampling,
                               transform=val_augmentor, is_train=False, test_mode=False,
                               seperator=filename_seperator, filter_video=filter_video)

    val_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                workers=args.workers,
                                is_distributed=args.distributed)

    log_folder = os.path.join(args.logdir, arch_name)
    if args.rank == 0:
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

    if args.evaluate:
        val_top1, val_top5, val_losses, val_speed = validate(val_loader, model, val_criterion,
                                                             gpu_id=args.gpu)
        if args.rank == 0:
            logfile = open(os.path.join(log_folder, 'evaluate_log.log'), 'a')
            print(
                'Val@{}: \tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch'.format(
                    args.input_size, val_losses, val_top1, val_top5, val_speed * 1000.0),
                flush=True)
            print(
                'Val@{}: \tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch'.format(
                    args.input_size, val_losses, val_top1, val_top5, val_speed * 1000.0),
                flush=True,
                file=logfile)
        return

    train_list = os.path.join(args.datadir, train_list_name)

    train_augmentor = get_augmentor(True, args.input_size, scale_range=args.scale_range, mean=mean,
                                    std=std,
                                    disable_scaleup=args.disable_scaleup,
                                    threed_data=args.threed_data,
                                    is_flow=True if args.modality == 'flow' else False,
                                    version=args.augmentor_ver)

    train_dataset = VideoDataSet(args.datadir, train_list, args.groups, args.frames_per_group,
                                 num_clips=args.num_clips,
                                 modality=args.modality, image_tmpl=image_tmpl,
                                 dense_sampling=args.dense_sampling,
                                 transform=train_augmentor, is_train=True, test_mode=False,
                                 seperator=filename_seperator, filter_video=filter_video)

    train_loader = build_dataflow(train_dataset, is_train=True, batch_size=args.batch_size,
                                  workers=args.workers, is_distributed=args.distributed)

    sgd_polices = model.parameters()
    optimizer = torch.optim.SGD(sgd_polices, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    if args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, args.lr_steps[0], gamma=0.1)
    elif args.lr_scheduler == 'multisteps':
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_steps, gamma=0.1)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    elif args.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    best_top1 = 0.0
    # optionally resume from a checkpoint
    if args.resume:
        if args.rank == 0:
            logfile = open(os.path.join(log_folder, 'log.log'), 'a')
        if os.path.isfile(args.resume):
            if args.rank == 0:
                print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_top1 = checkpoint['best_top1']
            if args.gpu is not None:
                if not isinstance(best_top1, float):
                    best_top1 = best_top1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
            except:
                pass
            if args.rank == 0:
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            del checkpoint  # dereference seems crucial
            torch.cuda.empty_cache()
        else:
            raise ValueError("Checkpoint is not found: {}".format(args.resume))
    else:
        if os.path.exists(os.path.join(log_folder, 'log.log')) and args.rank == 0:
            shutil.copyfile(os.path.join(log_folder, 'log.log'), os.path.join(
                log_folder, 'log.log.{}'.format(int(time.time()))))
        if args.rank == 0:
            logfile = open(os.path.join(log_folder, 'log.log'), 'w')

    if args.rank == 0:
        command = " ".join(sys.argv)
        tensorboard_logger.configure(os.path.join(log_folder))
        print(command, flush=True)
        print(args, flush=True)
        print(model, flush=True)
        print(command, file=logfile, flush=True)
        print(args, file=logfile, flush=True)

    if args.resume == '' and args.rank == 0:
        print(model, file=logfile, flush=True)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_top1, train_top5, train_losses, train_speed, speed_data_loader, train_steps = \
            train(train_loader, model, train_criterion, optimizer, epoch + 1,
                  display=args.print_freq, clip_gradient=args.clip_gradient,
                  gpu_id=args.gpu, rank=args.rank)
        if args.distributed:
            dist.barrier()

        # evaluate on validation set
        val_top1, val_top5, val_losses, val_speed = validate(val_loader, model, val_criterion,
                                                             gpu_id=args.gpu)

        # update current learning rate
        if args.lr_scheduler == 'plateau':
            scheduler.step(val_losses)
        else:
            scheduler.step()

        if args.distributed:
            dist.barrier()

        # only logging at rank 0
        if args.rank == 0:
            print('Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\t'
                  'Speed: {:.2f} ms/batch\tData loading: {:.2f} ms/batch'.format(
                epoch + 1, args.epochs, train_losses, train_top1, train_top5, train_speed * 1000.0,
                speed_data_loader * 1000.0), file=logfile, flush=True)
            print('Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\t'
                  'Speed: {:.2f} ms/batch\tData loading: {:.2f} ms/batch'.format(
                epoch + 1, args.epochs, train_losses, train_top1, train_top5, train_speed * 1000.0,
                speed_data_loader * 1000.0), flush=True)
            print('Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\t'
                  'Speed: {:.2f} ms/batch'.format(epoch + 1, args.epochs, val_losses, val_top1,
                                                  val_top5, val_speed * 1000.0), file=logfile,
                  flush=True)
            print('Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\t'
                  'Speed: {:.2f} ms/batch'.format(epoch + 1, args.epochs, val_losses, val_top1,
                                                  val_top5, val_speed * 1000.0), flush=True)

            # remember best prec@1 and save checkpoint
            is_best = val_top1 > best_top1
            best_top1 = max(val_top1, best_top1)

            save_dict = {'epoch': epoch + 1,
                         'arch': arch_name,
                         'state_dict': model.state_dict(),
                         'best_top1': best_top1,
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict()
                         }

            save_checkpoint(save_dict, is_best, filepath=log_folder)
            try:
                # get_lr get all lrs for every layer of current epoch, assume the lr for all layers are identical
                lr = scheduler.optimizer.param_groups[0]['lr']
            except Exception as e:
                lr = None
            if lr is not None:
                tensorboard_logger.log_value('learning-rate', lr, epoch + 1)
            tensorboard_logger.log_value('val-top1', val_top1, epoch + 1)
            tensorboard_logger.log_value('val-loss', val_losses, epoch + 1)
            tensorboard_logger.log_value('train-top1', train_top1, epoch + 1)
            tensorboard_logger.log_value('train-loss', train_losses, epoch + 1)
            tensorboard_logger.log_value('best-val-top1', best_top1, epoch + 1)

        if args.distributed:
            dist.barrier()

    if args.rank == 0:
        logfile.close()


if __name__ == '__main__':
    main()
