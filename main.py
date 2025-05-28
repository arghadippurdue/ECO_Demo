# general libs
from fvcore.nn import FlopCountAnalysis
from pathlib import Path
import openvino as ov
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
    # New flags
    # ---------------------------
    parser.add_argument('--dataset', action='store_true', default=False,
                        help='If true, only validate segmentation.')
    parser.add_argument('--realtime', action='store_true', default=False,
                        help='Run realtime inference using the Intel RealSense L515')
    parser.add_argument("--noise", type=str, default="0",
                        help="Noise level in images")
    parser.add_argument("--depth", action="store_true",
                        help="Enable depth images in the dataset")
    parser.add_argument("--device", type=str, default="CPU",
                        help="Device to use for inference (CPU, GPU, NPU, etc.)")
    parser.add_argument("--experiment", type=str, default="0")
    parser.add_argument('--backbone', default='mit_b2', type=str)
    parser.add_argument('--reportpower', action='store_true', default=False,
                        help='Report Power Using HWinfo')
    parser.add_argument('--framework', default='ov', type=str)
    return parser.parse_args()


def get_model():
    model_map = {
        "mit_b2": "ov_model/enc_dec_b2_torch_v1.xml",
        "mit_b3": "ov_model/enc_dec_b3_torch_v1.xml"
    }

    backbone_key = args.backbone
    # quantization = args.quantization

    try:
        model_path = model_map[backbone_key]
        print(f"Compiling the model from: {model_path}")
        return model_path
    except KeyError:
        raise ValueError(
            f"Invalid combination: backbone='{args.backbone}', quantization='{args.quantization}'")


def setup_hwinfo():
    global hdr
    global struct_fmt
    global mem

    sensor_struct = Struct(
        'sig' / Int32un, 'ver' / Int32un, 'rev' / Int32un, 'poll' / Long,
        'sens_off' / Int32un, 'sens_size' / Int32un, 'sens_count' / Int32un,
        'read_off' / Int32un, 'read_size' / Int32un, 'read_count' / Int32un
    )

    mem = shared_memory.SharedMemory('Global\\HWiNFO_SENS_SM2')
    hdr = sensor_struct.parse(mem.buf[:sensor_struct.sizeof()])
    struct_fmt = struct.Struct('=III128s128s16sdddd')


def get_cpu_power():
    """
    Returns (CPU Package Power, System Agent Power) in watts.
    Either element may be None if the sensor isn’t found.
    """
    cpu_pkg = None
    sys_agent = None

    off, size, count = hdr.read_off, hdr.read_size, hdr.read_count
    for i in range(count):
        start = off + i * size
        end = start + struct_fmt.size
        r = struct_fmt.unpack(mem.buf[start:end])

        label = r[3].split(b'\x00')[0].decode(errors='ignore')
        unit = r[5].split(b'\x00')[0].decode('mbcs', errors='ignore')
        if unit != "W":
            continue

        if label == "CPU Package Power":
            cpu_pkg = r[6]
        elif label == "System Agent Power":
            sys_agent = r[6]

        # Break early once we have both values
        if cpu_pkg is not None and sys_agent is not None:
            break

    return cpu_pkg, sys_agent

# ----------------------------------------------Dataset------------------------------------------------------

def create_segmenter(num_classes, backbone):
    """Create Encoder; for now only ResNet [50,101,152]"""
    segmenter = WeTr(backbone, num_classes)
    param_groups = segmenter.get_param_groups()
    # assert(torch.cuda.is_available())
    # segmenter.to(gpu[0])
    segmenter.to('cpu')
    if args.framework == "torch":
        # AD: Found the issue. If commented, wrong output
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
    # print_log('Created train set {} examples, val set {} examples'.format(
        # len(trainset), len(validset)))
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

    if args.framework == "torch":
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        for (k, v) in ckpt_dict.items():
            if k in ckpt:
                v.load_state_dict(ckpt[k], strict=False)
        best_val = ckpt.get('best_val', 0)
        epoch_start = ckpt.get('epoch_start', 0)
        return best_val, epoch_start

    elif args.framework == "ov":
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        for (k, v) in ckpt_dict.items():
            if k in ckpt:
                state_dict = ckpt[k]
                # print(state_dict.keys())
                # Remove "module." prefix if model was trained with DataParallel
                new_state_dict = {k.removeprefix(
                    'module.'): v for k, v in state_dict.items()}
                # print(new_state_dict.keys())
                v.load_state_dict(new_state_dict, strict=True)
                # print(v.state_dict().keys())
                # v.load_state_dict(ckpt[k], strict=True)
        best_val = ckpt.get('best_val', 0)
        epoch_start = ckpt.get('epoch_start', 0)
        return best_val, epoch_start


def validate_torch(segmenter, input_types, val_loader, epoch, num_classes=-1, save_image=0, commclass=None):
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
            # if i > 3:
            # break
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

            end_time = time.time()

            if args.reportpower:
                pkg_w, sa_w = get_cpu_power()
                print(f"CPU Package: {pkg_w:.3f} W,  NPU: {sa_w:.3f} W")

            iteration_time = end_time - start_time
            print(
                f"Iteration {i+1}/{len(val_loader.dataset)}: {iteration_time:.4f} seconds")

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

                if idx == 2 and (i < save_image or save_image == -1):
                    output_dir = "results_new"

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

    for idx, input_type in enumerate(input_types + ['ens']):
        glob, mean, iou = getScores(conf_mat[idx])
        best_iou_note = ''
        if iou > best_iou:
            best_iou = iou
            best_iou_note = '    (best)'
        alpha = '        '
        # if idx < len(alpha_soft):
        #     alpha = '    %.2f' % alpha_soft[idx]
        input_type_str = '(%s)' % input_type
        print_log('Epoch %-4d %-7s   glob_acc=%-5.2f    mean_acc=%-5.2f    IoU=%-5.2f%s%s' %
                  (epoch, input_type_str, glob, mean, iou, alpha, best_iou_note))
        # val_metrics = {
        #                 # f'val/{input_type}/val_epoch': epoch,
        #                 f'{input_type}/global_accuracy': glob,
        #                 f'{input_type}/mean_accuracy': mean,
        #                 f'{input_type}/iou': iou,
        #               }
        # wandb.log({**val_metrics, **val_metrics})
    # print_log('')
    # print_log("Model Backbone{}, Image Size {}, Total flops {}, Flops by ops {}".
    #       format(args.backbone, args.input_size, flops.total(), flops.by_operator()))
    # iou = None
    return iou


def validate_ov(input_types, val_loader, epoch, num_classes=-1, save_image=0):
    """Validate segmenter

    Args:
      segmenter (nn.Module) : segmentation network
      val_loader (DataLoader) : training data iterator
      epoch (int) : current epoch
      num_classes (int) : number of classes to consider

    Returns:
      Mean IoU (float)
    """

    core = ov.Core()

    # compiled_model = core.compile_model(model=str(model_path), device_name="NPU")
    # compiled_model = core.compile_model(model=str(model_path), device_name="NPU", {ov::intel_npu::turbo(true), ov::intel_npu::tiles(6)})

    if args.device == "CPU":
        device_name = "CPU"
        # config = {}
    elif args.device == "GPU":
        device_name = "GPU"
        # config = {}
    else:
        device_name = "NPU"
        # config = {"NPU_MAX_TILES": 6, "NPU_TILES": 6}

    compiled_model = core.compile_model(
        model=get_model(),
        device_name=device_name
        # config={"NPU_MAX_TILES": 6, "NPU_TILES": 6}
    )

    global best_iou
    conf_mat = []
    conf_mat = np.zeros((3, num_classes, num_classes), dtype=np.int64)

    def add_speckle_noise(tensor: torch.Tensor) -> torch.Tensor:
        noise_level = float(args.noise)
        noise = torch.randn_like(tensor)
        noisy = tensor + tensor * noise * noise_level
        return noisy

    for i, sample in enumerate(val_loader):

        if i > 10:
            break

        target = sample['mask']
        inputs = [sample[key].float() for key in input_types]
        input1, _ = inputs
        B, C, H, W = input1.shape          # C should be 3 for RGB

        if not args.depth:
            inputs[1] = torch.zeros(
                (B, 3, H, W), dtype=input1.dtype, device=input1.device)

        inputs[0] = add_speckle_noise(input1)

        # IOU
        gt = target[0].data.cpu().numpy().astype(np.uint8)
        gt_idx = gt < num_classes  # Ignore every class index larger than the number of classes

        # ## Save for debug
        # file_path = "inputs.pth"
        # torch.save({"input1": input1, "input2": input1}, file_path)
        # print(f"Inputs saved to {file_path}")

        # DBG: with torch.no_grad():
        #     torch_output = segmenter(input1, input2)[output_idx]#.cpu().numpy()

        start_time = time.time()

        infer_res = compiled_model(inputs={
            "input1": inputs[0].numpy(),
            "input2": inputs[1].numpy()
        })

        end_time = time.time()

        if args.reportpower:
            pkg_w, sa_w = get_cpu_power()
            print(f"CPU Package: {pkg_w:.3f} W,  NPU: {sa_w:.3f} W")

        iteration_time = end_time - start_time
        print(
            f"Iteration {i+1}/{len(val_loader.dataset)}: {iteration_time:.4f} seconds")

        for idx in range(3):

            ov_output = np.asarray(infer_res[idx].data)
            output = torch.tensor(ov_output)
            output = cv2.resize(output[0, :num_classes].data.cpu().numpy().transpose(1, 2, 0),
                                target.size()[1:][::-1], interpolation=cv2.INTER_CUBIC).argmax(axis=2).astype(np.uint8)

            iou = confusion_matrix(gt[gt_idx], output[gt_idx], num_classes)
            conf_mat[idx] += iou

            if idx == 2 and (i < save_image or save_image == -1):
                output_dir = "results_new"
                img = make_validation_img(inputs[0].data.cpu().numpy(),
                                          inputs[1].data.cpu().numpy(),
                                          sample['mask'].data.cpu().numpy(),
                                          output[np.newaxis, :])
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"validate_{i}.png")
                cv2.imwrite(output_file, img[:, :, ::-1])
                print(f"imwrite at {output_file}")

    for idx, input_type in enumerate(input_types + ['ens']):
        glob, mean, iou = getScores(conf_mat[idx])
        best_iou_note = ''
        if iou > best_iou:
            best_iou = iou
            best_iou_note = '    (best)'
        alpha = '        '
        # if idx < len(alpha_soft):
        #     alpha = '    %.2f' % alpha_soft[idx]
        input_type_str = '(%s)' % input_type
        print_log('Epoch %-4d %-7s   glob_acc=%-5.2f    mean_acc=%-5.2f    IoU=%-5.2f%s%s' %
                  (epoch, input_type_str, glob, mean, iou, alpha, best_iou_note))

        # # 3. Retrieve original color image
        # #    shape [3, H, W] → [H, W, 3]
        # color_image = sample['rgb'][0].cpu().numpy().transpose(1, 2, 0)
        # orig_h, orig_w, _ = color_image.shape

        # # 4. Resize each channel back to the original size
        # pred_resized = np.zeros((num_classes, orig_h, orig_w), dtype=pred.dtype)
        # for c in range(num_classes):
        #     pred_resized[c] = cv2.resize(pred[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # # 5. Argmax to get final segmentation map
        # seg_mask = np.argmax(pred_resized, axis=0).astype(np.uint8)

        # # 6. Optionally save the 4-in-1 image
        # if (i < save_image) or (save_image == -1):
        #     # Convert the float RGB image into uint8 for visualization
        #     rgb_3ch = (color_image * 255).astype(np.uint8) if color_image.max() <= 1.0 else color_image.astype(np.uint8)

        #     # (A) LiDAR / Depth
        #     #    Assuming sample['depth'] has shape [1, H, W]
        #     if 'depth' in sample:
        #         lidar_data = sample['depth'][0].cpu().numpy()
        #         # Normalize to 0-255 for display
        #         lidar_data = (lidar_data / lidar_data.max() * 255).astype(np.uint8)
        #         lidar_3ch = cv2.cvtColor(lidar_data, cv2.COLOR_GRAY2BGR)
        #     else:
        #         # Fallback if there's no 'depth' in your dataset
        #         lidar_3ch = np.zeros_like(rgb_3ch)

        #     # (B) Reference mask
        #     #    shape [H, W]
        #     if 'mask' in sample:
        #         ref_mask = sample['mask'][0].cpu().numpy().astype(np.uint8)
        #         ref_mask_3ch = colorize_segmentation(ref_mask)
        #     else:
        #         # Fallback if there's no 'mask'
        #         ref_mask_3ch = np.zeros_like(rgb_3ch)

        #     # (C) Computed (predicted) mask
        #     comp_mask_3ch = colorize_segmentation(seg_mask)

        #     # Ensure all images are same size (H, W)
        #     # (We know they're all roughly the same from your code, but just to be safe:)
        #     lidar_3ch    = cv2.resize(lidar_3ch, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        #     ref_mask_3ch = cv2.resize(ref_mask_3ch, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        #     comp_mask_3ch= cv2.resize(comp_mask_3ch, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        #     # (D) Concatenate horizontally: [RGB | LiDAR | Ref | Pred]
        #     combined = np.hstack((rgb_3ch, lidar_3ch, ref_mask_3ch, comp_mask_3ch))

        #     # 7. Write to disk
        #     out_dir = "segmentation_results"
        #     os.makedirs(out_dir, exist_ok=True)
        #     out_file = os.path.join(out_dir, f"validate_{i}.png")

        #     # OpenCV expects BGR, so flip the channel order
        #     cv2.imwrite(out_file, combined[:, :, ::-1])
        #     print(f"[INFO] Segmentation result saved to {out_file}")

    return iou

# -------------------Realtime----------------------------------------------

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


def display_windows(fps, processed_rgb, depth_image, lidar_img, color_seg, hello_img, args):

    # First, clear the canvas to black
    hello_img[:] = 0

    # Define colors in BGR
    GREEN = (30, 255, 50)
    BLUE = (255, 100, 20)
    WHITE = (255, 255, 255)

    # get current power
    # Get CPU and System Agent power
    cpu_pkg_power, sys_agent_power = get_cpu_power()

    # Line for Power display
    cpu_display_power_str = "Not available"
    npu_display_power_str = "Not available"

    if sys_agent_power is not None:
        npu_display_power_str = f"{sys_agent_power:.2f} W"

    if cpu_pkg_power is not None and sys_agent_power is not None:
        cpu_calc_power = cpu_pkg_power - sys_agent_power
        cpu_display_power_str = f"{cpu_calc_power:.2f} W"
    # else if cpu_pkg_power is not None (and sys_agent_power is None):
        # In this case, cpu_calc_power (cpu_pkg - sys_agent) cannot be calculated.
        # So cpu_display_power_str remains "Not available"

    line8 = f"CPU Power: {cpu_display_power_str} NPU Power: {npu_display_power_str}"

    # Line for FPS display
    # FPS will be doubled if --depth arg is false
    effective_fps = fps * 2 if not args.depth else fps
    line7 = f"FPS: {effective_fps:.2f}"

    # Line for Performance/Watt
    # This will use the 'cpu_pkg_power' for the denominator, similar to original commented logic
    if cpu_pkg_power is not None and cpu_pkg_power > 0:
        ppw = effective_fps / cpu_pkg_power
        line9 = f"Performance/Watt: {ppw:.2f} FPS/W"
    else:
        line9 = "Performance/Watt: Not available"
        
        
    # Line for Compute framework
    framework_name = ""
    if args.framework == 'ov':
        framework_name = "OpenVINO"
    elif args.framework == 'torch':
        framework_name = "PyTorch" # Assuming 'torch' refers to PyTorch

    line2 = f"Compute: {framework_name} on {args.device}"
    
    line3 = "Modalities:"
    if args.noise == '100':
        line4 = "    RGB: OFF"
    elif args.noise == '99':
        line4 = "    RGB: 640x480 (Sensor Failure)"
    else:
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
            "line10": "Takeaway: Moderate Accuracy & FPS"
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


def run_realtime_inference_ov(segmenter, input_types, epoch, num_classes=-1, save_image=0):
    """
    Realtime inference function using the Intel RealSense L515.
    Optimized for CPU-only execution.
    """
    import pyrealsense2 as rs
    import time

    noise_level = float(args.noise)

    def add_speckle_noise(image):
        noise = np.random.randn(*image.shape)
        noisy = image + image * noise * noise_level
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def add_black_noise(image):
        return np.zeros_like(image, dtype=np.uint8)
    # Configure the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    if args.depth:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    core = ov.Core()
    # print(f"Compiling the model from: {model_map[args.backbone]}")

    if args.device == "CPU":
        device_name = "CPU"
        config = {}
    elif args.device == "GPU":
        device_name = "GPU"
        config = {}
    else:
        device_name = "NPU"
        config = {"NPU_MAX_TILES": 6, "NPU_TILES": 6}

    compiled_model = core.compile_model(
        model=get_model(),
        device_name=device_name,
        config=config
    )
    output_idx = 2

    # Initialize OpenCV windows
    create_windows()

    # Pre-create a black canvas for displaying text messages
    status_image = np.zeros((480, 680, 3), dtype=np.uint8)
    lidar_img = np.zeros((480, 640, 3), dtype=np.uint8)

    fixed_palette = get_fixed_palette()

    print("\033[92m Starting ECO realtime demo! Close window to exit.")
    prev_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            if not color_frame:
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
            if args.noise == "100":
                processed_rgb = add_black_noise(color_image)
            else:
                processed_rgb = add_speckle_noise(color_image)

            sample = create_loaders_realtime(
                processed_rgb, depth_image, args.input_size,
                args.normalise_params, args.input_size_d, args.inter)

            # Convert each torch tensor to a NumPy array and build a dictionary
            inputs = {}
            for i, key in enumerate(input_types):
                tensor = sample[key].float().unsqueeze(
                    0)  # shape: [1, C, H, W]
                # Map the i-th model input name to the numpy array
                input_name = compiled_model.inputs[i].get_any_name()
                inputs[input_name] = tensor.cpu().numpy()

            # Run inference using the OpenVINO compiled model
            # input1, input2 = inputs
            # ov_output = compiled_model(inputs={
            #     "input1": input1.numpy(),
            #     "input2": input2.numpy()
            # })[output_idx].data
            ov_output = compiled_model(inputs=inputs)[output_idx].data

            ov_output = np.asarray(ov_output)

            # Assume the output shape is [1, num_classes, H_out, W_out]
            pred = ov_output[0]
            if num_classes == -1:
                num_classes = pred.shape[0]
            else:
                pred = pred[:num_classes]

            # Resize each channel to match the original image resolution
            pred_resized = np.zeros(
                (num_classes, color_image.shape[0], color_image.shape[1]), dtype=pred.dtype)
            for c in range(num_classes):
                pred_resized[c] = cv2.resize(pred[c], (color_image.shape[1], color_image.shape[0]),
                                             interpolation=cv2.INTER_LINEAR)

            # Take the argmax along the channel dimension to get the segmentation mask
            seg_mask = np.argmax(pred_resized, axis=0).astype(np.uint8)

            # Colorize the segmentation mask (assuming this function is defined)
            color_seg = colorize_segmentation(seg_mask, fixed_palette)

            # --- Calculate FPS ---
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            # --- Display windows ---
            display_windows(fps, processed_rgb, depth_image,
                            lidar_img, color_seg, status_image, args)

            # print("Evaulation complete")

            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def run_realtime_inference_torch(segmenter, input_types, epoch, num_classes=-1, save_image=0):
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

    print("\033[92m Starting ECO realtime demo! Close window to exit.")
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
    # print_log('Loaded Segmenter {}, ImageNet-Pre-Trained={}, #PARAMS={:3.2f}M'
            #   .format(args.backbone, args.enc_pretrained, compute_params(segmenter) / 1e6))

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


# ---------------------------------inference-----------------------------------------------

        if args.reportpower:
            setup_hwinfo()

        if args.realtime:
            if args.framework == "torch":
                return run_realtime_inference_torch(segmenter, args.input, 0, num_classes=args.num_classes,
                                                    save_image=args.save_image)
            elif args.framework == "ov":
                return run_realtime_inference_ov(segmenter, args.input, 0, num_classes=args.num_classes,
                                                 save_image=args.save_image)
        if args.dataset:
            if args.framework == "torch":
                return validate_torch(segmenter, args.input, val_loader, 0, num_classes=args.num_classes,
                                      save_image=args.save_image, commclass=commapprox)
            elif args.framework == "ov":
                return validate_ov(args.input, val_loader, 0, num_classes=args.num_classes,
                                   save_image=args.save_image)

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
