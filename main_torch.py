# general libs
from fvcore.nn import FlopCountAnalysis
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from models.segformer import WeTr
from utils.optimizer import PolyWarmupAdamW
import utils.helpers as helpers
from utils import *
from config import *
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import cv2
import os
import sys
import argparse
import re
import math
import random
import time
import warnings

import struct
from multiprocessing import shared_memory
from construct import Struct, Int32un, Long
warnings.filterwarnings('ignore')


# Setup once
sensor_struct = Struct(
    'sig' / Int32un, 'ver' / Int32un, 'rev' / Int32un, 'poll' / Long,
    'sens_off' / Int32un, 'sens_size' / Int32un, 'sens_count' / Int32un,
    'read_off' / Int32un, 'read_size' / Int32un, 'read_count' / Int32un
)

mem = shared_memory.SharedMemory('Global\\HWiNFO_SENS_SM2')
hdr = sensor_struct.parse(mem.buf[:sensor_struct.sizeof()])
struct_fmt = struct.Struct('=III128s128s16sdddd')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Full Pipeline Training')

    # Dataset
    parser.add_argument('-d', '--train-dir', type=str, default=TRAIN_DIR,
                        help='Path to the training set directory.')
    parser.add_argument('--val-dir', type=str, default=VAL_DIR,
                        help='Path to the validation set directory.')
    parser.add_argument('--train-list', type=str, default=TRAIN_LIST,
                        help='Path to the training set list.')
    parser.add_argument('--val-list', type=str, default=VAL_LIST,
                        help='Path to the validation set list.')
    parser.add_argument('--shorter-side', type=int, default=SHORTER_SIDE,
                        help='Shorter side transformation.')
    parser.add_argument('--crop-size', type=int, default=CROP_SIZE,
                        help='Crop size for training,')
    parser.add_argument('--input-size', type=int, default=RESIZE_SIZE,
                        help='Final RGB input size of the model')
    parser.add_argument('--input-size-d', type=int, default=RESIZE_SIZE,
                        help='Final Depth input size of the model')
    parser.add_argument('--normalise-params', type=list, default=NORMALISE_PARAMS,
                        help='Normalisation parameters [scale, mean, std],')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size to train the segmenter model.')
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS,
                        help="Number of workers for pytorch's dataloader.")
    parser.add_argument('--num-classes', type=int, default=NUM_CLASSES,
                        help='Number of output classes for each task.')
    parser.add_argument('--low-scale', type=float, default=LOW_SCALE,
                        help='Lower bound for random scale')
    parser.add_argument('--high-scale', type=float, default=HIGH_SCALE,
                        help='Upper bound for random scale')
    parser.add_argument('--ignore-label', type=int, default=IGNORE_LABEL,
                        help='Label to ignore during training')

    # Encoder
    parser.add_argument('--enc', type=str, default=ENC,
                        help='Encoder net type.')
    parser.add_argument('--enc-pretrained', type=bool, default=ENC_PRETRAINED,
                        help='Whether to init with imagenet weights.')
    # General
    parser.add_argument('--name', default='', type=str,
                        help='model name')
    # parser.add_argument('--gpu', type=int, nargs='+', default=[1],
    #                     help='select gpu.')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='If true, only validate segmentation.')
    parser.add_argument('--freeze-bn', type=bool, nargs='+', default=FREEZE_BN,
                        help='Whether to keep batch norm statistics intact.')
    parser.add_argument('--num-epoch', type=int, nargs='+', default=NUM_SEGM_EPOCHS,
                        help='Number of epochs to train for segmentation network.')
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED,
                        help='Seed to provide (near-)reproducibility.')
    parser.add_argument('-c', '--ckpt', default='model', type=str, metavar='PATH',
                        help='path to save checkpoint (default: model)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--val-every', type=int, default=VAL_EVERY,
                        help='How often to validate current architecture.')
    parser.add_argument('--print-network', action='store_true', default=False,
                        help='Whether print newtork paramemters.')
    parser.add_argument('--print-loss', action='store_true', default=False,
                        help='Whether print losses during training.')
    parser.add_argument('--save-image', type=int, default=100,
                        help='Number to save images during evaluating, -1 to save all.')
    parser.add_argument('-i', '--input', default=['rgb', 'depth'], type=str, nargs='+',
                        help='input type (image, depth)')

    # Optimisers
    parser.add_argument('--lr-enc', type=float, nargs='+', default=LR_ENC,
                        help='Learning rate for encoder.')
    parser.add_argument('--lr-dec', type=float, nargs='+', default=LR_DEC,
                        help='Learning rate for decoder.')
    parser.add_argument('--mom-enc', type=float, default=MOM_ENC,
                        help='Momentum for encoder.')
    parser.add_argument('--mom-dec', type=float, default=MOM_DEC,
                        help='Momentum for decoder.')
    parser.add_argument('--wd-enc', type=float, default=WD_ENC,
                        help='Weight decay for encoder.')
    parser.add_argument('--wd-dec', type=float, default=WD_DEC,
                        help='Weight decay for decoder.')
    parser.add_argument('--optim-dec', type=str, default=OPTIM_DEC,
                        help='Optimiser algorithm for decoder.')
    parser.add_argument('--lamda', type=float, default=LAMDA,
                        help='Lamda for L1 norm.')
    # parser.add_argument('-t', '--bn-threshold', type=float, default=BN_threshold,
    #                     help='Threshold for slimming BNs.')
    parser.add_argument('--backbone', default='mit_b1', type=str)

    # approx args
    parser.add_argument('--err-data', type=str, default='/home/kanchi/a/das169/ECHO/TokenFusion/',
                        help='path to error masks')
    parser.add_argument('-ri', '--refresh-interval', type=int, nargs='+', default=[1, 1],
                        help='refresh interval')
    parser.add_argument('-q', '--quality', type=int, nargs='+', default=[None, None],
                        help='image compression quality')
    parser.add_argument('--cloud', action='store_true',
                        help='cloud or edge')
    parser.add_argument('--inter', type=str, nargs='+', default=['nearest', 'nearest'],
                        help='interpolation method')

    # ---------------------------
    # New flag for realtime inference
    # ---------------------------
    parser.add_argument('--realtime', action='store_true', default=False,
                        help='Run realtime inference using the Intel RealSense L515')

    parser.add_argument("--noise", type=str, default="0",
                        help="Noise level in images")
    parser.add_argument("--depth", action="store_true",
                        help="Enable depth images in the dataset")
    parser.add_argument("--device", type=str, default="CPU",
                        help="Device to use for inference (CPU, GPU, NPU, etc.)")
    parser.add_argument("--experiment", type=str, default="0")
    return parser.parse_args()


def create_segmenter(num_classes, backbone):
    """Create Encoder; for now only ResNet [50,101,152]"""
    segmenter = WeTr(backbone, num_classes)
    param_groups = segmenter.get_param_groups()
    # assert(torch.cuda.is_available())
    # segmenter.to(gpu[0])
    segmenter.to('cpu')
    segmenter = torch.nn.DataParallel(segmenter)
    # segmenter = DistributedDataParallel(wetr, device_ids=[-1], find_unused_parameters=True)
    return segmenter, param_groups


def create_loaders(dataset, inputs, train_dir, val_dir, train_list, val_list,
                   shorter_side, crop_size, input_size, low_scale, high_scale,
                   normalise_params, batch_size, num_workers, ignore_label, input_size_d, inter):
    """
    Args:
      train_dir (str) : path to the root directory of the training set.
      val_dir (str) : path to the root directory of the validation set.
      train_list (str) : path to the training list.
      val_list (str) : path to the validation list.
      shorter_side (int) : parameter of the shorter_side resize transformation.
      crop_size (int) : square crop to apply during the training.
      low_scale (float) : lowest scale ratio for augmentations.
      high_scale (float) : highest scale ratio for augmentations.
      normalise_params (list / tuple) : img_scale, img_mean, img_std.
      batch_size (int) : training batch size.
      num_workers (int) : number of workers to parallelise data loading operations.
      ignore_label (int) : label to pad segmentation masks with

    Returns:
      train_loader, val loader

    """
    # Torch libraries
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split
    # Custom libraries
    from utils.datasets import SegDataset as Dataset
    from utils.transforms import Normalise, Pad, RandomCrop, RandomMirror, ResizeAndScale, \
        CropAlignToMask, ResizeAlignToMask, ToTensor, ResizeInputs, ResizeandUpscaleInputs
    from utils.approx_transforms import MemoryApprox, CommunicationApprox

    input_names, input_mask_idxs = ['rgb', 'depth'], [0, 2, 1]

    AlignToMask = CropAlignToMask if dataset == 'nyudv2' else ResizeAlignToMask
    composed_trn = transforms.Compose([
        AlignToMask(),
        ResizeAndScale(shorter_side, low_scale, high_scale),
        Pad(crop_size, [123.675, 116.28, 103.53], ignore_label),
        RandomMirror(),
        RandomCrop(crop_size),
        ResizeInputs(input_size),
        Normalise(*normalise_params),
        ToTensor()
    ])
    commapprox = CommunicationApprox(quality=args.quality, cloud=args.cloud)
    composed_val = transforms.Compose([
        AlignToMask(),
        # ResizeInputs(input_size),
        ResizeandUpscaleInputs(input_size, input_size_d, inter),
        MemoryApprox(refresh_interval=args.refresh_interval,
                     mask_dir=args.err_data, img_size=args.input_size),
        commapprox,
        Normalise(*normalise_params),
        ToTensor()
    ])
    # Training and validation sets
    trainset = Dataset(dataset=dataset, data_file=train_list, data_dir=train_dir,
                       input_names=input_names, input_mask_idxs=input_mask_idxs,
                       transform_trn=composed_trn, transform_val=composed_val,
                       stage='train', ignore_label=ignore_label)

    validset = Dataset(dataset=dataset, data_file=val_list, data_dir=val_dir,
                       input_names=input_names, input_mask_idxs=input_mask_idxs,
                       transform_trn=None, transform_val=composed_val, stage='val',
                       ignore_label=ignore_label)
    # DBG:
    # print(validset[0]['depth'].dtype)
    print_log('Created train set {} examples, val set {} examples'.format(
        len(trainset), len(validset)))
    # Training and validation loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(validset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, commapprox


def create_optimisers(lr_enc, lr_dec, mom_enc, mom_dec, wd_enc, wd_dec, param_enc, param_dec, optim_dec):
    """Create optimisers for encoder, decoder and controller"""
    optim_enc = torch.optim.SGD(
        param_enc, lr=lr_enc, momentum=mom_enc, weight_decay=wd_enc)
    if optim_dec == 'sgd':
        optim_dec = torch.optim.SGD(
            param_dec, lr=lr_dec, momentum=mom_dec, weight_decay=wd_dec)
    elif optim_dec == 'adam':
        optim_dec = torch.optim.Adam(
            param_dec, lr=lr_dec, weight_decay=wd_dec, eps=1e-3)

    return optim_enc, optim_dec


def load_ckpt(ckpt_path, ckpt_dict):
    # ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    for (k, v) in ckpt_dict.items():
        if k in ckpt:
            v.load_state_dict(ckpt[k], strict=False)
    best_val = ckpt.get('best_val', 0)
    epoch_start = ckpt.get('epoch_start', 0)
    print_log('Found checkpoint at {} with best_val {:.4f} at epoch {}'.
              format(ckpt_path, best_val, epoch_start))
    return best_val, epoch_start


def train(segmenter, input_types, train_loader, optimizer, epoch,
          segm_crit, freeze_bn, lamda, batch_size, print_loss=False):
    """Training segmenter

    Args:
      segmenter (nn.Module) : segmentation network
      train_loader (DataLoader) : training data iterator
      optim_enc (optim) : optimiser for encoder
      optim_dec (optim) : optimiser for decoder
      epoch (int) : current epoch
      segm_crit (nn.Loss) : segmentation criterion
      freeze_bn (bool) : whether to keep BN params intact

    """
    train_loader.dataset.set_stage('train')
    segmenter.train()
    if freeze_bn:
        for module in segmenter.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    example_ct = 0
    # TODO: REPLACE WITH BATCH SIZE
    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / batch_size)
    for i, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
        # print('train input:', sample['rgb'].shape, sample['depth'].shape, sample['mask'].shape)
        start = time.time()
        inputs = [sample[key].float() for key in input_types]
        target = sample['mask'].long()
        example_ct += len(sample)
        # Compute outputs
        outputs, masks = segmenter(inputs)
        loss = 0
        for output in outputs:
            output = nn.functional.interpolate(output, size=target.size()[1:],
                                               mode='bilinear', align_corners=False)
            soft_output = nn.LogSoftmax()(output)
            # Compute loss and backpropagate
            loss += segm_crit(soft_output, target)

        if lamda > 0:
            L1_loss = 0
            for mask in masks:
                L1_loss += sum([torch.abs(m).sum() for m in mask])
            loss += lamda * L1_loss

        optimizer.zero_grad()
        loss.backward()
        if print_loss:
            print('step: %-3d: loss=%.2f' % (i, loss), flush=True)
        optimizer.step()
        losses.update(loss.item())
        batch_time.update(time.time() - start)

    portion_rgbs, portion_depths = [], []
    for idx, mask in enumerate(masks):
        portion_rgb = (mask[0] < 0.02).sum() / mask[0].flatten().shape[0]
        portion_depth = (mask[1] < 0.02).sum() / mask[1].flatten().shape[0]
        portion_rgbs.append(portion_rgb)
        portion_depths.append(portion_depth)
    portion_rgbs = sum(portion_rgbs) / len(portion_rgbs)
    portion_depths = sum(portion_depths) / len(portion_depths)
    print('Epoch %d, portion of scores<0.02 (rgb depth): %.2f%% %.2f%%' %
          (epoch, portion_rgbs * 100, portion_depths * 100), flush=True)

    metrics = {"train/train_loss": loss,
               "train/epoch": epoch,
               "train/example_ct": example_ct,
               "train/portion_rgbs": portion_rgbs * 100,
               "train/portion_depths": portion_depths * 100,
               }


def validate(segmenter, input_types, val_loader, epoch, num_classes=-1, save_image=0, commclass=None):
    """Validate segmenter

    Args:
      segmenter (nn.Module) : segmentation network
      val_loader (DataLoader) : training data iterator
      epoch (int) : current epoch
      num_classes (int) : number of classes to consider

    Returns:
      Mean IoU (float)
    """
    global best_iou
    val_loader.dataset.set_stage('val')
    segmenter.eval()
    conf_mat = []
    for _ in range(len(input_types) + 1):
        conf_mat.append(np.zeros((num_classes, num_classes), dtype=int))
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            # ## Arghadip: DBG for only a few inputs
            # if i > 20:
            #     break
            # print('valid input:', sample['rgb'].shape, sample['depth'].shape, sample['mask'].shape)
            start = time.time()
            # ## DBG: Arghadip
            # print("RGB datatype from dataloader: ", sample['rgb'].dtype)      # --> float32
            # print("Depth datatype from dataloader: ", sample['rgb'].dtype)    # --> float32
            inputs = [sample[key].float() for key in input_types]
            # print("RGB datatype before segmenter: ", inputs[0].dtype)         # --> float32
            # print("Depth datatype before segmenter: ", inputs[1].dtype)       # --> float32

            target = sample['mask']
            gt = target[0].data.cpu().numpy().astype(np.uint8)
            gt_idx = gt < num_classes  # Ignore every class index larger than the number of classes

            start_time = time.time()  # Record the start time

            # Compute outputs
            # outputs, alpha_soft = segmenter(inputs)
            # outputs, _ = segmenter(inputs)
            input1, input2 = inputs
            outputs = segmenter(input1, input2)
            # outputs = segmenter(inputs)

            end_time = time.time()  # Record the end time
            # Calculate the time taken for the iteration
            iteration_time = end_time - start_time
            # Print the iteration number and the time taken
            print(
                f"Iteration {i+1}/{len(val_loader.dataset)}: {iteration_time:.4f} seconds")

            # Print the number of parameters for the first iteration
            if i == 0:
                print(sum(p.numel() for p in segmenter.parameters()))

            # DBG:
            # compress_metrics = {'bytes_rgb': commclass.compress_size['rgb'],
            #                     'bytes_depth': commclass.compress_size['depth'],
            #                     'bytes_total': commclass.compress_size['total']
            #                     #   'bytes_avg_rgb': commclass.compress_size['rgb']//654,
            #                     #   'bytes_avg_depth': commclass.compress_size['depth']//654,
            #                     #   'bytes_avg_total': commclass.compress_size['total']//654,
            #                       }
            # print(compress_metrics)

            # if i == len(val_loader) - 1:
            # ## DBG:
            # # if i == 20:
            #     compress_metrics = {'bytes_rgb': commclass.compress_size['rgb'],
            #                       'bytes_depth': commclass.compress_size['depth'],
            #                       'bytes_total': commclass.compress_size['total'],
            #                       'bytes_avg_rgb': commclass.compress_size['rgb']//654,
            #                       'bytes_avg_depth': commclass.compress_size['depth']//654,
            #                       'bytes_avg_total': commclass.compress_size['total']//654,
            #                       }
            #     print(compress_metrics)
            #     ## DBG:
            #     # print(commclass.compress_size)
            #     # print(commclass)
            #     # wandb.log({**compress_metrics, **compress_metrics})
            # if i == 0:
            #     # measure flops
            #     flops = FlopCountAnalysis(segmenter, inputs)
            #     print(flops)
            #     val_metrics = {
            #                     'flops': flops.total(),
            #                     'input_height': sample['rgb'].shape[2],
            #                     'input_width': sample['rgb'].shape[3]
            #                     }
            #     wandb.log({**val_metrics, **val_metrics})

            for idx, output in enumerate(outputs):
                output = cv2.resize(output[0, :num_classes].data.cpu().numpy().transpose(1, 2, 0),
                                    target.size()[1:][::-1],
                                    interpolation=cv2.INTER_CUBIC).argmax(axis=2).astype(np.uint8)
                # Compute IoU
                iou = confusion_matrix(gt[gt_idx], output[gt_idx], num_classes)
                conf_mat[idx] += iou
                # conf_mat[idx] += confusion_matrix(gt[gt_idx], output[gt_idx], num_classes)
                if i < save_image or save_image == -1:
                    # DBG
                    # if i == 0:
                    # Create a wandb Table to log images, labels and predictions to
                    # table = wandb.Table(columns=["image", "depth", "mask", "IoU"])#+[f"score_{i}" for i in range(num_classes)])

                    # table.add_data(wandb.Image(inputs[0].data.cpu().numpy()), inputs[1].data.cpu().numpy(),
                    #                   sample['mask'].data.cpu().numpy(), iou)

                    output_dir = "result"

                    img = make_validation_img(inputs[0].data.cpu().numpy(),
                                              inputs[1].data.cpu().numpy(),
                                              sample['mask'].data.cpu().numpy(),
                                              output[np.newaxis, :])
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = os.path.join(output_dir, f"validate_{i}.png")
                    cv2.imwrite(output_file, img[:, :, ::-1])
                    print(f"imwrite at {output_file}")

        # if save_image == -1:
        #     print("Logging model predictions to W&B")
        #     # wandb.log({"val_table/predictions_table":table}, commit=False)

    # for idx, input_type in enumerate(input_types + ['ens']):
    #     glob, mean, iou = getScores(conf_mat[idx])
    #     best_iou_note = ''
    #     if iou > best_iou:
    #         best_iou = iou
    #         best_iou_note = '    (best)'
    #     alpha = '        '
    #     # if idx < len(alpha_soft):
    #     #     alpha = '    %.2f' % alpha_soft[idx]
    #     input_type_str = '(%s)' % input_type
    #     print_log('Epoch %-4d %-7s   glob_acc=%-5.2f    mean_acc=%-5.2f    IoU=%-5.2f%s%s' %
    #         (epoch, input_type_str, glob, mean, iou, alpha, best_iou_note))
    #     # val_metrics = {
    #     #                 # f'val/{input_type}/val_epoch': epoch,
    #     #                 f'{input_type}/global_accuracy': glob,
    #     #                 f'{input_type}/mean_accuracy': mean,
    #     #                 f'{input_type}/iou': iou,
    #     #               }
    #     # wandb.log({**val_metrics, **val_metrics})
    # print_log('')
    # print_log("Model Backbone{}, Image Size {}, Total flops {}, Flops by ops {}".
    #       format(args.backbone, args.input_size, flops.total(), flops.by_operator()))
    # # iou = None
    # return iou


def get_fixed_palette():
    """
    Generates a fixed palette mapping each class index to a unique color.
    """
    np.random.seed(42)  # Fixed seed for consistency
    return {label: np.random.randint(0, 255, (3,), dtype=np.uint8) for label in range(40)}


def colorize_segmentation(segmentation, palette):
    """
    Converts a segmentation map (HxW) into a color image using a fixed palette.

    Args:
      segmentation (np.array): Segmentation map of shape (H, W) with class indices.
      palette (dict): Dictionary mapping class labels to RGB colors.

    Returns:
      color_seg (np.array): Color image representing the segmentation.
    """
    h, w = segmentation.shape
    color_seg = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in palette.items():
        color_seg[segmentation == label] = color
    return color_seg


def create_loaders_realtime(color_image, depth_image, input_size,
                            normalise_params, input_size_d, inter):
    """
    Args:
      train_dir (str) : path to the root directory of the training set.
      val_dir (str) : path to the root directory of the validation set.
      train_list (str) : path to the training list.
      val_list (str) : path to the validation list.
      shorter_side (int) : parameter of the shorter_side resize transformation.
      crop_size (int) : square crop to apply during the training.
      low_scale (float) : lowest scale ratio for augmentations.
      high_scale (float) : highest scale ratio for augmentations.
      normalise_params (list / tuple) : img_scale, img_mean, img_std.
      batch_size (int) : training batch size.
      num_workers (int) : number of workers to parallelise data loading operations.
      ignore_label (int) : label to pad segmentation masks with

    Returns:
      train_loader, val loader

    """
    # Torch libraries
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split
    # Custom libraries
    from utils.datasets import SegDataset as Dataset
    from utils.transforms import Normalise, Pad, RandomCrop, RandomMirror, ResizeAndScale, \
        CropAlignToMask, ResizeAlignToMask, ToTensor, ResizeInputs, ResizeandUpscaleInputs
    from utils.approx_transforms import MemoryApprox, CommunicationApprox

    input_names, input_mask_idxs = ['rgb', 'depth'], [0, 2, 1]

    AlignToMask = CropAlignToMask  # if dataset == 'nyudv2' else ResizeAlignToMask
    commapprox = CommunicationApprox(quality=args.quality, cloud=args.cloud)
    composed_val = transforms.Compose([
        AlignToMask(),
        # ResizeInputs(input_size),
        ResizeandUpscaleInputs(input_size, input_size_d, inter),
        MemoryApprox(refresh_interval=args.refresh_interval,
                     mask_dir=args.err_data, img_size=args.input_size),
        commapprox,
        Normalise(*normalise_params),
        ToTensor()
    ])

    sample = {}
    for key in input_names:
        if key == 'rgb':
            sample[key] = color_image
        elif key == 'depth':
            sample[key] = depth_image

    # mask = torch.zeros((468, 625), dtype=torch.uint8)
    mask = np.zeros((468, 625), dtype=np.uint8)
    # assert len(mask.shape) == 2, 'Masks must be encoded without colourmap'
    sample['mask'] = mask
    sample['inputs'] = ['rgb', 'depth']

    sample = composed_val(sample)
    del sample['inputs']

    return sample


def create_windows():

    # Set window positions
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 480
    OFFSET_X = 50  # Distance from left
    OFFSET_Y = 50  # Distance from top

    # Initialize and position OpenCV windows once
    cv2.namedWindow('RGB Input', cv2.WINDOW_NORMAL)
    cv2.moveWindow('RGB Input', OFFSET_X, OFFSET_Y)

    cv2.namedWindow('LiDAR Input', cv2.WINDOW_NORMAL)
    cv2.moveWindow('LiDAR Input', OFFSET_X + WINDOW_WIDTH, OFFSET_Y)

    cv2.namedWindow('Segmentation Mask', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Segmentation Mask', OFFSET_X, OFFSET_Y + WINDOW_HEIGHT)

    cv2.namedWindow('System Status', cv2.WINDOW_NORMAL)
    cv2.moveWindow('System Status', OFFSET_X +
                   WINDOW_WIDTH, OFFSET_Y + WINDOW_HEIGHT)


def get_cpu_power():
    off, size, count = hdr.read_off, hdr.read_size, hdr.read_count
    for i in range(count):
        start = off + i * size
        end = start + struct_fmt.size
        r = struct_fmt.unpack(mem.buf[start:end])
        label = r[3].split(b'\x00')[0].decode(errors='ignore')
        unit = r[5].split(b'\x00')[0].decode('mbcs', errors='ignore')
        if label == "CPU Package Power" and unit == "W":
            return r[6]
    return None  # Not found


def display_windows(fps, processed_rgb, depth_image, lidar_img, color_seg, hello_img, args):

    # First, clear the canvas to black
    hello_img[:] = 0

    # Define colors in BGR
    GREEN = (30, 255, 50)
    BLUE = (255, 100, 20)
    WHITE = (255, 255, 255)

    # get current power
    power = get_cpu_power()
    line8 = f"Power: {power:.2f} W" if power else "Power reading not found."

    # Compute FPS display (special handling for experiment "3")
    fps_display = f"{fps * 2:.2f}" if args.experiment in ["3", "4"] else f"{fps:.2f}"
    line7 = f"FPS: {fps_display}"

    if power:  # Avoid division by None or zero
        fps_value = fps * 2 if args.experiment in ["3", "4"] else fps
        ppw = fps_value / power
        line9 = f"Performance/Watt: {ppw:.2f} FPS/W"
    else:
        line9 = "Performance/Watt: Not available"

    # Common lines
    line2 = f"Compute: Torch on {args.device}"
    line3 = "Modalities:"
    line4 = f"    RGB: 640x480 (Noise {'ON' if args.noise != '0' else 'OFF'})"
    line5 = f"    LiDAR: {'640x480' if args.depth else 'OFF'}"
    line6 = f"Backbone Model: {args.backbone}"

    # Experiment-specific configurations
    experiment_config = {
        "0": {
            "line1": "Running Custom Configuration",
            "line10": ""
        },
        "1": {
            "line1": "Running Experiment 1",
            "line10": "Takeaway: Baseline Accuracy, Low FPS"
        },
        "2": {
            "line1": "Running Experiment 2",
            "line10": "Takeaway: Baseline Accuracy, Moderate FPS"
        },
        "3": {
            "line1": "Running Experiment 3",
            "line10": "Takeaway: Low Accuracy, High FPS"
        },
        "4": {
            "line1": "Running Experiment 4",
            "line10": "Takeaway: Baseline Accuracy, High FPS"
        },
        "5": {
            "line1": "Running Experiment 5",
            "line10": "Takeaway: Low Accuracy, moderate FPS"
        },
        "6": {
            "line1": "Running Experiment 6",
            "line10": "Takeaway: Moderate Accuracy, moderate FPS"
        }
    }

    config = experiment_config.get(args.experiment, {
        "line1": "Running Unknown Experiment",
        "line9": "",
        "line10": ""
    })

    lines = [
        config["line1"], line2, line3, line4, line5, line6,
        line7, line8, line9, config["line10"]
    ]

    # Font and spacing
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    y0 = 50
    dy = 40

    # Draw each line with color coding
    for i, line in enumerate(lines):
        y = y0 + i * dy
        x = 10

        if i == 0:
            # First line: green
            cv2.putText(hello_img, line, (x, y), font,
                        font_scale, GREEN, thickness)
        elif ':' in line:
            label, value = line.split(':', 1)
            label += ':'  # add colon back

            # Draw label in blue
            cv2.putText(hello_img, label, (x, y), font,
                        font_scale, BLUE, thickness)

            # Get width of label to offset value
            (label_width, _), _ = cv2.getTextSize(
                label, font, font_scale, thickness)
            cv2.putText(hello_img, value.strip(), (x + label_width +
                        10, y), font, font_scale, WHITE, thickness)
        else:
            # Just draw in white (like 'Modalities:')
            cv2.putText(hello_img, line, (x, y), font,
                        font_scale, WHITE, thickness)

    cv2.imshow('System Status', hello_img)

    # Display RGB image
    cv2.imshow('RGB Input', processed_rgb)

    # Display LiDAR image
    lidar_img[:] = 0
    if args.depth:
        cv2.imshow('LiDAR Input', depth_image)
    else:
        fps_text = f"Lidar Disabled!"
        cv2.putText(
            lidar_img,
            fps_text,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        cv2.imshow('LiDAR Input', lidar_img)

    # Display segmentation mask
    cv2.imshow('Segmentation Mask', color_seg)


def run_realtime_inference(segmenter, input_types, epoch, num_classes=-1, save_image=0):
    """
    Realtime inference function using the Intel RealSense L515.
    Optimized for CPU-only execution.
    """
    import pyrealsense2 as rs
    from torchvision import transforms
    import time

    noise_level = float(args.noise)

    def add_speckle_noise(image):
        noise = np.random.randn(*image.shape)
        noisy = image + image * noise * noise_level
        return np.clip(noisy, 0, 255).astype(np.uint8)

    # Configure the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    # Model in eval mode
    segmenter.eval()

    # Initialize OpenCV windows
    create_windows()

    # Pre-create a black canvas for displaying text messages
    status_image = np.zeros((480, 680, 3), dtype=np.uint8)
    lidar_img = np.zeros((480, 640, 3), dtype=np.uint8)

    fixed_palette = get_fixed_palette()

    print("Starting realtime inference. Press ESC to exit.")
    prev_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            if args.depth:
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue
                depth_image = np.asanyarray(depth_frame.get_data())
            else:
                # Create a dummy blank depth image with the same height and width as the color image.
                dummy_depth = np.zeros(
                    (color_image.shape[0], color_image.shape[1]), dtype=np.uint8)
                depth_image = dummy_depth

            # Expand depth to 3 channels
            depth_image = np.repeat(depth_image[..., np.newaxis], 3, axis=2)

            # Apply noise if argument is used
            processed_rgb = add_speckle_noise(color_image)

            sample = create_loaders_realtime(
                processed_rgb, depth_image, args.input_size,
                args.normalise_params, args.input_size_d, args.inter)

            # inputs = [sample[key].float() for key in input_types]
            inputs = [sample[key].float().unsqueeze(0) for key in input_types]

            with torch.no_grad():
                input1, input2 = inputs
                outputs = segmenter(input1, input2)

            pred = outputs[2]
            pred = nn.functional.interpolate(
                pred, size=color_image.shape[:2], mode='bilinear', align_corners=False)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

            color_seg = colorize_segmentation(pred, fixed_palette)

            # --- Calculate FPS ---
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            # --- Display windows ---
            display_windows(fps, processed_rgb, depth_image,
                            lidar_img, color_seg, status_image, args)

            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main():
    global args, best_iou
    best_iou = 0
    args = get_arguments()
    # print("args.cloud= ", args.cloud)     ## Arghadip: DBG
    args.num_stages = len(args.lr_enc)

    ckpt_dir = os.path.join('ckpt', args.ckpt)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.system('cp -r *py models utils data %s' % ckpt_dir)
    helpers.logger = open(os.path.join(ckpt_dir, 'log.txt'), 'w+')
    print_log(' '.join(sys.argv))

    # Set random seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    # Generate Segmenter
    # torch.cuda.set_device(args.gpu[0])
    # segmenter, param_groups = create_segmenter(args.num_classes, args.gpu, args.backbone)
    segmenter, param_groups = create_segmenter(args.num_classes, args.backbone)

    if args.print_network:
        print_log('')
    # segmenter = model_init(segmenter, args.enc, len(args.input), imagenet=args.enc_pretrained)
    print_log('Loaded Segmenter {}, ImageNet-Pre-Trained={}, #PARAMS={:3.2f}M'
              .format(args.backbone, args.enc_pretrained, compute_params(segmenter) / 1e6))

    # Restore if any
    best_val, epoch_start = 0, 0
    if args.resume:
        if os.path.isfile(args.resume):
            best_val, epoch_start = load_ckpt(
                args.resume, {'segmenter': segmenter})
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume))
            return
    epoch_current = epoch_start
    # Criterion
    segm_crit = nn.NLLLoss(ignore_index=args.ignore_label)
    # Saver
    saver = Saver(args=vars(args), ckpt_dir=ckpt_dir, best_val=best_val,
                  condition=lambda x, y: x > y)  # keep checkpoint with the best validation score

    lrs = [6e-5, 3e-5, 1.5e-5]

    for task_idx in range(args.num_stages):
        optimizer = PolyWarmupAdamW(
            params=[
                {
                    "params": param_groups[0],
                    "lr": lrs[task_idx],
                    "weight_decay": 0.01,
                },
                {
                    "params": param_groups[1],
                    "lr": lrs[task_idx],
                    "weight_decay": 0.0,
                },
                {
                    "params": param_groups[2],
                    "lr": lrs[task_idx] * 10,
                    "weight_decay": 0.01,
                },
            ],
            lr=lrs[task_idx],
            weight_decay=0.01,
            betas=[0.9, 0.999],
            warmup_iter=1500,
            max_iter=40000,
            warmup_ratio=1e-6,
            power=1.0
        )
        total_epoch = sum([args.num_epoch[idx] for idx in range(task_idx + 1)])
        if epoch_start >= total_epoch:
            continue
        start = time.time()
        torch.cuda.empty_cache()
        # Create dataloaders
        train_loader, val_loader, commapprox = create_loaders(
            DATASET, args.input, args.train_dir, args.val_dir, args.train_list, args.val_list,
            args.shorter_side, args.crop_size, args.input_size, args.low_scale, args.high_scale,
            args.normalise_params, args.batch_size, args.num_workers, args.ignore_label, args.input_size_d, args.inter)

        if args.realtime:
            return run_realtime_inference(segmenter, args.input, 0, num_classes=args.num_classes,
                                          save_image=args.save_image)

        if args.evaluate:
            # wandb.init(project=args.ckpt, name=f'{args.backbone}_{args.input_size},{args.input_size_d}_{args.refresh_interval}_{args.quality}',
            #            config={'backbone': args.backbone,
            #                    'params': sum(p.numel() for p in segmenter.parameters()),
            #                    'batch_size': args.batch_size,
            #                    'input size_rgb':args.input_size,
            #                    'input size_depth':args.input_size_d,
            #                    'inter_rgb': args.inter[0],
            #                    'inter_depth': args.inter[1],
            #                    'refresh_interval_rgb': args.refresh_interval[0],
            #                    'refresh_interval_depth': args.refresh_interval[1],
            #                    'quality_rgb': args.quality[0],
            #                    'quality_depth': args.quality[1],
            #                    }
            #            )
            # if args.refresh_interval[0] == args.refresh_interval[1]:
            #     category = 'Both'
            # elif args.refresh_interval[0] != 1:
            #     category = 'Only RGB'
            # elif args.refresh_interval[1] != 1:
            #     category = 'Only Depth'

            # args_metrics = {
            #                 # 'memory optim': category,
            #                 'val/refresh_interval_rgb': args.refresh_interval[0],
            #                 'val/refresh_interval_depth': args.refresh_interval[1],
            #                 'val/quality_rgb': args.quality[0],
            #                 'val/quality_depth': args.quality[1]
            #                 }

            # wandb.log({**args_metrics, **args_metrics})
            # DBG:
            # print(commapprox.compress_size)
            return validate(segmenter, args.input, val_loader, 0, num_classes=args.num_classes,
                            save_image=args.save_image, commclass=commapprox)

        # Optimisers
        print_log('Training Stage {}'.format(str(task_idx)))
        # optim_enc, optim_dec = create_optimisers(
        #     args.lr_enc[task_idx], args.lr_dec[task_idx],
        #     args.mom_enc, args.mom_dec,
        #     args.wd_enc, args.wd_dec,
        #     enc_params, dec_params, args.optim_dec)

        # wandb.init(project=args.ckpt, config={'stage': task_idx, 'epochs': total_epoch, 'batch_size': args.batch_size, 'lr': lrs})
        for epoch in range(min(args.num_epoch[task_idx], total_epoch - epoch_start)):
            train(segmenter, args.input, train_loader, optimizer, epoch_current,
                  segm_crit, args.freeze_bn, args.lamda, args.batch_size, args.print_loss)
            if (epoch + 1) % (args.val_every) == 0:
                miou = validate(segmenter, args.input, val_loader,
                                epoch_current, args.num_classes)
                saver.save(
                    miou, {'segmenter': segmenter.state_dict(), 'epoch_start': epoch_current})
            epoch_current += 1

        print_log('Stage {} finished, time spent {:.3f}min\n'.format(
            task_idx, (time.time() - start) / 60.))

    print_log(
        'All stages are now finished. Best Val is {:.3f}'.format(saver.best_val))
    helpers.logger.close()


if __name__ == '__main__':
    main()
