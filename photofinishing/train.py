"""
Copyright (c) 2025 Samsung Electronics Co., Ltd.

Author(s):
Mahmoud Afifi (m.afifi1@samsung.com, m.3afifi@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

This file contains the training script for the photofinishing module.
"""

import argparse
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import numpy as np
import tensorboard.summary
from tabulate import tabulate
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, List
from utils.file_utils import write_json_file, read_json_file

from dataset import Data
from photofinishing_model import PhotofinishingModule
from loss_utils import PhotofinishingLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.constants import *

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def show_ltm_gain_maps(writer: SummaryWriter, ltm_map: torch.Tensor, global_step: int, tag_prefix: str):
  """
  Adds LTM coefficient maps (1 or 3 channels) as grayscale heatmaps to TensorBoard.

  Args:
    writer: TensorBoard SummaryWriter.
    ltm_map: Tensor of shape (B, C, H, W), C can be 1, 2, or 3.
    global_step: TensorBoard global step.
    tag_prefix: Base tag name for logging.
  """
  b, c, h, w = ltm_map.shape
  ltm_map = ltm_map.detach().cpu()

  maps_per_row = 8
  for i in range(c):
    channel_maps = ltm_map[:, i]
    rows = (b + maps_per_row - 1) // maps_per_row
    padded = torch.zeros((rows * maps_per_row, h, w))
    padded[:b] = channel_maps
    grid = padded.view(rows, maps_per_row, h, w)
    grid = grid.permute(0, 2, 1, 3).reshape(rows * h, maps_per_row * w)


    fig, ax = plt.subplots(figsize=(maps_per_row * 2, rows * 2))
    im = ax.imshow(grid.numpy(), cmap='summer', aspect='auto')
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    ax.set_title(f'LTM Coeff Map - Channel {i}')

    writer.add_figure(f'{tag_prefix}/LTM_Batch_Channel_{i}', fig, global_step=global_step)
    plt.close(fig)


def show_3d_lut(writer: SummaryWriter, lut3d: torch.Tensor, global_step: int, tag_prefix: str):
  """Visualizes a 3D LUT as a 2D grid of RG slices (slicing over B) for TensorBoard.

  Args:
    lut3d: torch.Tensor of shape (1, 3, D, D, D), values in [0, 1]
    writer: TensorBoard SummaryWriter
    global_step: int
    tag_prefix: str
  """
  lut = lut3d.squeeze(0).detach().cpu().permute(1, 2, 3, 0).numpy()
  c = lut.shape[0]

  r_vals = np.linspace(0, 1, c)
  g_vals = np.linspace(0, 1, c)
  b_vals = np.linspace(0, 1, c)
  rr, gg, bb = np.meshgrid(r_vals, g_vals, b_vals, indexing='ij')

  r_in = rr.flatten()
  g_in = gg.flatten()
  b_in = bb.flatten()
  rgb_in = np.stack([r_in, g_in, b_in], axis=-1)

  rgb_out = lut.reshape(-1, 3)

  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111, projection='3d')

  ax.scatter(rgb_in[:, 0], rgb_in[:, 1], rgb_in[:, 2],
             c=np.clip(rgb_out, 0.0, 1.0), marker='o', s=10)

  ax.set_xlabel('Input R')
  ax.set_ylabel('Input G')
  ax.set_zlabel('Input B')
  ax.set_title('3D LUT')
  plt.tight_layout()

  writer.add_figure(tag_prefix, fig, global_step=global_step)
  plt.close(fig)


def show_cbcr_lut_heatmaps(writer: SummaryWriter, cbcr_lut: torch.Tensor, global_step: int, tag_prefix: str):
  """
  Visualizes CbCr LUT as a perceptual chroma map (Y fixed, convert YCbCr -> RGB).

  Args:
      writer: TensorBoard SummaryWriter.
      cbcr_lut: Tensor of shape (B, 2, H, W), values in [-0.5, 0.5].
      global_step: Global step.
      tag_prefix: Prefix for TensorBoard tags.
  """

  def cbcr_to_rgb(cb_: torch.Tensor, cr_: torch.Tensor, y_val: Optional[float] = 0.5) -> np.ndarray:
    y = torch.full_like(cb, y_val)
    ycbcr = torch.stack([y, cb_, cr_], dim=0).permute(1, 2, 0).numpy()
    matrix = np.array(YCBCR_TO_RGB)
    rgb = ycbcr @ matrix.T
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb

  b = cbcr_lut.shape[0]
  for idx in range(b):
    cb = cbcr_lut[idx, 0].detach().cpu()
    cr = cbcr_lut[idx, 1].detach().cpu()
    rgb_img = cbcr_to_rgb(cb, cr)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(np.rot90(rgb_img, k=1), interpolation='nearest')
    ax.set_title('CbCr Chroma Map')
    ax.axis('off')
    writer.add_figure(f'{tag_prefix}/CbCrGrid_{idx}', fig, global_step=global_step)
    plt.close(fig)

def print_line(end: Optional[bool]=False, length: Optional[int]=100):
  """Prints a separator line."""
  line = '-' * length
  if end:
    print(f'\n{line}\n')
  else:
    print(f'\n{line}')


def training(model: PhotofinishingModule, epochs: int, lr: float, l2_reg: float, tr_device: torch.device,
             train_loader: DataLoader, val_loader: DataLoader, train: Data, global_step: List[int],
             validation_frequency: int, exp_name: str, batch_size: int, compute_loss: PhotofinishingLoss,
             writer: tensorboard.summary.Writer, log: Dict):
  """Performs training on the given dataloaders."""

  optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=l2_reg)
  scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)

  for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    with tqdm(total=len(train) * batch_size, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
      for batch_idx, batch in enumerate(train_loader):
        images = batch['in_images'].squeeze(0)
        gt_images = batch['gt_images'].squeeze(0)
        in_images = images.to(device=tr_device, non_blocking=True)
        gt_images = gt_images.to(device=tr_device, non_blocking=True)
        out_images = model(in_images, training_mode=True)

        gt_images_lin = model.de_gamma(gt_images, out_images['gamma_factor'])
        gt_lin_ycbcr_images = model.rgb_to_ycbcr(gt_images_lin)[:, 1:, ...]
        gt_lin_y_images = model.rgb_to_ycbcr(gt_images_lin)[:, 0, ...].unsqueeze(1)

        loss, detailed_loss = compute_loss(
          out_img=out_images['output'], gt_img=gt_images, lsrgb_out_img=out_images['processed_lsrgb'],
          lsrgb_gt_img=gt_images_lin, rgb_lut_out_img=out_images['lsrgb_3d_lut'],
          cbcr_out_img=out_images['processed_cbcr'], cbcr_lut=out_images['cbcr_lut'], pre_tm_y = out_images['y_gain'],
          ltm_y = out_images['ltm_y'], gtm_y = out_images['gtm_y'], ltm_map = out_images['ltm_params'],
          cbcr_gt_img=gt_lin_ycbcr_images, y_gt_img=gt_lin_y_images)

        epoch_loss += loss.item()

        if writer:
          writer.add_scalar(f'Loss/train', loss.item(), global_step[0])
          writer.add_scalar(f'L1/train', detailed_loss['l1'], global_step[0])
          writer.add_scalar(f'VGG/train', detailed_loss['vgg'], global_step[0])
          writer.add_scalar(f'PSNR/train', detailed_loss['psnr'], global_step[0])
          writer.add_scalar(f'SSIM/train', detailed_loss['ssim'], global_step[0])
          writer.add_scalar(f'DeltaE/train', detailed_loss['delta-e'], global_step[0])
          writer.add_scalar(f'CbCr/train', detailed_loss['cbcr'], global_step[0])
          writer.add_scalar(f'LuT smoothness/train', detailed_loss['lut-smoothness'], global_step[0])
          writer.add_scalar(f'TM/train', detailed_loss['tm'], global_step[0])
          writer.add_scalar(f'LTM smoothness/train', detailed_loss['ltm-smoothness'], global_step[0])
          writer.add_scalar(f'Luma energy consistency/train', detailed_loss['luma-energy-consistency'], global_step[0])
        pbar.set_postfix({
          f'Batch-loss': f'{detailed_loss["total"]:.4f} - L1={detailed_loss["l1"]:.4f}, '
                         f'PSNR={detailed_loss["psnr"]:.4f}, SSIM={detailed_loss["ssim"]:.4f}, '
                         f'D-E={detailed_loss["delta-e"]:.4f}, '
                         f'VGG={detailed_loss["vgg"]:.4f}, '
                         f'CbCr={detailed_loss["cbcr"]:.4f}, '
                         f'TM={detailed_loss["tm"]:.4f}, '
                         f'LuT TV={detailed_loss["lut-smoothness"]:.4f}, '
                         f'LTM TV={detailed_loss["ltm-smoothness"]:.4f}, '
                         f'Luma energy={detailed_loss["luma-energy-consistency"]:.4f}'})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.update(np.ceil(images.shape[0]))
        global_step[0] += 1

    if (epoch + 1) % validation_frequency == 0:
      val_loss = validate(model=model, loader=val_loader, val_device=tr_device,
                          compute_loss=compute_loss, writer=writer, global_step=global_step[0])
      print_line()

      logging.info(
        f'Validation loss: {val_loss["total"]:.4f} - L1={val_loss["l1"]:.4f}, '
        f'PSNR={val_loss["psnr"]:.4f}, SSIM={val_loss["ssim"]:.4f}, '
        f'D-E={val_loss["delta-e"]:.4f}, VGG={val_loss["vgg"]:.4f}, CbCr={val_loss["cbcr"]:.4f}, '
        f'TM={val_loss["tm"]:.4f}, LuT TV={val_loss["lut-smoothness"]:.4f}, '
        f'LTM TV={val_loss["ltm-smoothness"]:.4f}, '
        f'Luma energy={val_loss["luma-energy-consistency"]:.4f}\n'
        )

      checkpoint_model_name = os.path.join(BASE_DIR, 'checkpoints', f'{exp_name}_{epoch + 1}.pth')
      torch.save(model.state_dict(), checkpoint_model_name)
      logging.info(f'Checkpoint {epoch + 1} saved!')
      print_line(end=True)

      log['checkpoint_model_name'].append(checkpoint_model_name)
      log['val_psnr'].append(
        val_loss['psnr'].item() if isinstance(val_loss['psnr'], torch.Tensor) else val_loss['psnr'])
      write_json_file(log, os.path.join(BASE_DIR, 'logs', f'{exp_name}'))

      if writer:
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step[0])
        writer.add_scalar(f'Loss/val', val_loss["total"], global_step[0])
        writer.add_scalar(f'L1/val', val_loss['l1'], global_step[0])
        writer.add_scalar(f'SSIM/val', val_loss['ssim'], global_step[0])
        writer.add_scalar(f'VGG/val', val_loss['vgg'], global_step[0])
        writer.add_scalar(f'PSNR/val', val_loss['psnr'], global_step[0])
        writer.add_scalar(f'DeltaE/val', val_loss['delta-e'], global_step[0])
        writer.add_scalar(f'LuT smoothness/val', val_loss['lut-smoothness'], global_step[0])
        writer.add_scalar(f'CbCr/val', val_loss['cbcr'], global_step[0])
        writer.add_scalar(f'TM/val', val_loss['tm'], global_step[0])
        writer.add_scalar(f'LTM smoothness/val', val_loss['ltm-smoothness'], global_step[0])
        writer.add_scalar(f'Luma energy consistency/val', val_loss['luma-energy-consistency'], global_step[0])
        writer.add_images('Input images/train', images, global_step[0])
        writer.add_images('Output images/train', out_images['output'], global_step[0])
        writer.add_images('GT images/train', gt_images, global_step[0])

    scheduler.step()

  torch.save(model.state_dict(), os.path.join(BASE_DIR, 'models', f'{exp_name}.pth'))
  logging.info('Saved trained model!')
  best_model_idx = log['val_psnr'].index(max(log['val_psnr']))
  best_model_name = log['checkpoint_model_name'][best_model_idx]
  shutil.copy(best_model_name, os.path.join(BASE_DIR, 'models', f'{exp_name}-best.pth'))

def train_net(model: PhotofinishingModule, tr_device: torch.device, in_tr_dir: str, gt_tr_dir: str,
              data_tr_dir: str, in_val_dir: str, gt_val_dir: str, data_val_dir: str, epochs: int, batch_size: int,
              lr: float, l2_reg: float, in_sz: int, validation_frequency: int, exp_name: str,
              temp_folder: str, overwrite_temp_folder: bool, delete_temp_folder: bool,
              l1_loss_weight: float, ssim_loss_weight: float, delta_e_loss_weight: float, perceptual_loss_weight: float,
              luma_energy_consistency_loss_weight: float, ltm_smoothness_loss_weight: float, tm_loss_weight: float,
              cbcr_loss_weight: float, lut_smoothness_loss_weight: float, no_tensorboard: bool, extract_patches: bool):
  """Trains photofinishing networks."""

  print_line()
  print(f'Training on {in_sz}x{in_sz} images ...')
  print_line(end=True)

  compute_loss = PhotofinishingLoss(l1_weight=l1_loss_weight, ssim_weight=ssim_loss_weight,
                                    delta_e_weight=delta_e_loss_weight, vgg_weight=perceptual_loss_weight,
                                    cbcr_weight=cbcr_loss_weight, lut_smooth_weight=lut_smoothness_loss_weight,
                                    luma_energy_weight=luma_energy_consistency_loss_weight,
                                    ltm_smooth_weight=ltm_smoothness_loss_weight, tm_weight=tm_loss_weight,
                                    device=device)

  writer = SummaryWriter(comment=f'TB-{exp_name}') if SummaryWriter and not no_tensorboard else None
  global_step = [0]

  train = Data(in_img_dir=in_tr_dir, gt_img_dir=gt_tr_dir, data_dir=data_tr_dir if data_tr_dir is None else data_tr_dir,
               temp_folder=temp_folder, overwrite_temp_folder=overwrite_temp_folder, batch_size=batch_size,
               image_size=in_sz, shuffle=True, geometric_aug=True, extract_patches=extract_patches)
  val = Data(in_img_dir=in_val_dir, gt_img_dir=gt_val_dir,
             data_dir=data_val_dir if data_val_dir is None else data_val_dir, image_size=in_sz,
             temp_folder=temp_folder, overwrite_temp_folder=overwrite_temp_folder, geometric_aug=False,
             batch_size=batch_size, shuffle=True, extract_patches=extract_patches)
  train_loader = DataLoader(train, batch_size=1, num_workers=12, pin_memory=True, persistent_workers=False,
                            shuffle=True)
  val_loader = DataLoader(val, batch_size=1, num_workers=12, pin_memory=True, drop_last=True,
                          persistent_workers=False, shuffle=True)

  log = {'checkpoint_model_name': [], 'val_psnr': []}

  training(model=model, epochs=epochs, lr=lr, l2_reg=l2_reg, tr_device=tr_device,
           train_loader=train_loader, val_loader=val_loader, train=train, global_step=global_step,
           validation_frequency=validation_frequency, exp_name=exp_name, batch_size=batch_size,
           compute_loss=compute_loss, writer=writer, log=log)

  if writer:
    writer.close()
  logging.info(f'End of training.')

  if delete_temp_folder:
    logging.info('Deleting temp folders')
    if extract_patches:
      postfix = '_patches'
    else:
      postfix = ''
    tr_temp_dir = os.path.join(os.path.dirname(gt_tr_dir),
                               f'{temp_folder}_{os.path.basename(gt_tr_dir)}_bs_{batch_size}_sz_{in_sz}{postfix}')

    val_temp_dir = os.path.join(os.path.dirname(gt_val_dir),
                                f'{temp_folder}_{os.path.basename(gt_val_dir)}_bs_{batch_size}_sz_{in_sz}{postfix}')
    shutil.rmtree(tr_temp_dir)
    if tr_temp_dir != val_temp_dir:
      shutil.rmtree(val_temp_dir)
    logging.info('Done!')


def validate(model: PhotofinishingModule, loader: DataLoader, val_device: torch.device,
             compute_loss: PhotofinishingLoss, writer: SummaryWriter, global_step: int,
             ) -> Dict[str, float]:
  """Network validation."""
  model.eval()

  val_loss = {'total': 0.0, 'l1': 0.0, 'ssim': 0.0, 'delta-e': 0.0, 'psnr': 0.0, 'cbcr': 0.0,
              'lut-smoothness': 0.0, 'luma-energy-consistency': 0.0, 'ltm-smoothness': 0.0,
              'tm': 0.0, 'vgg': 0.0}

  with torch.no_grad():
    for idx, batch in enumerate(loader):
      in_images = batch['in_images'].squeeze(0)
      gt_images = batch['gt_images'].squeeze(0)
      in_images = in_images.to(device=val_device, non_blocking=True)
      gt_images = gt_images.to(device=val_device, non_blocking=True)
      out_images = model(in_images, training_mode=True)

      gt_images_lin = model.de_gamma(gt_images, out_images['gamma_factor'])
      gt_lin_ycbcr_images = model.rgb_to_ycbcr(gt_images_lin)[:, 1:, ...]
      gt_lin_y_images = model.rgb_to_ycbcr(gt_images_lin)[:, 0, ...].unsqueeze(1)

      _, detailed_b_loss= compute_loss(
        out_img=out_images['output'], gt_img=gt_images, lsrgb_out_img=out_images['processed_lsrgb'],
        lsrgb_gt_img=gt_images_lin, rgb_lut_out_img=out_images['lsrgb_3d_lut'],
        cbcr_out_img=out_images['processed_cbcr'], cbcr_gt_img=gt_lin_ycbcr_images, y_gt_img=gt_lin_y_images,
        cbcr_lut=out_images['cbcr_lut'], pre_tm_y=out_images['y_gain'], ltm_y=out_images['ltm_y'],
        gtm_y=out_images['gtm_y'], ltm_map=out_images['ltm_params'])

      for key in val_loss:
        val_loss[key] += detailed_b_loss[key]

  if writer:
    writer.add_images('Input images/val', in_images, global_step)
    if out_images['lsrgb_gtm'] is not None:
      writer.add_images('GTM images/val', out_images['lsrgb_gtm'], global_step)
    writer.add_images('LTM images/val', out_images['lsrgb_ltm'], global_step)
    show_cbcr_lut_heatmaps(writer, out_images['cbcr_lut'], global_step, tag_prefix='CbCr LuT/val')
    if 'rgb_lut' in out_images and out_images['rgb_lut'] is not None:
      show_3d_lut(writer, out_images['rgb_lut'], global_step, tag_prefix='RGB LuT/val')
    show_ltm_gain_maps(writer, out_images['ltm_params'], global_step, tag_prefix='LTM param map/val')
    writer.add_images('Output LsRGB images/val', out_images['processed_lsrgb'], global_step)
    writer.add_images('Output images/val', out_images['output'], global_step)
    writer.add_images('GT images/val', gt_images, global_step)

  for key in val_loss:
    val_loss[key] /= idx
  model.train()
  return val_loss


def get_args():
  parser = argparse.ArgumentParser(description='Photofinishing module.')
  parser.add_argument(
    '--in-training-dir', dest='in_tr_dir', type=str, required=True, help='Training input image directory.')
  parser.add_argument(
    '--gt-training-dir', dest='gt_tr_dir', type=str, required=True,
    help='Training ground-truth image directory.')
  parser.add_argument(
    '--data-training-dir', dest='data_tr_dir', type=str, default=None,
    help='Training input data directory.')
  parser.add_argument(
    '--in-validation-dir', dest='in_vl_dir', type=str, required=True,
    help='Validation input image directory.')
  parser.add_argument(
    '--gt-validation-dir', dest='gt_vl_dir', type=str, required=True,
    help='Validation ground-truth image directory.')
  parser.add_argument('--data-validation-dir', dest='data_vl_dir', default=None, type=str,
                      help='Validation input data directory.')
  parser.add_argument('--use-3d-lut', dest='use_3d_lut', action='store_true',
                      help='To use a global learnable 3D LuT.')
  parser.add_argument('--epochs', type=int, default=600, dest='epochs')
  parser.add_argument('--batch-size', type=int, default=8, dest='batch_size')
  parser.add_argument('--learning-rate', type=float, default=0.0001, dest='lr')
  parser.add_argument('--l2reg', type=float, default=0.0000001, help='L2 Regularization factor',
                      dest='l2_r')
  parser.add_argument( '--load', dest='load', type=str, default=None, help='Load model from a .pth file')
  parser.add_argument('--validation-frequency', dest='val_frq', type=int, default=4)
  parser.add_argument('--in-size', dest='in_sz', type=int,
                      default=PHOTOFINISHING_TRAINING_INPUT_SIZE, help='Size of training images.')
  parser.add_argument('--temp-folder', dest='temp_folder', type=str, default='ps_temp_h5',
                      help='Name of temporary folder to save training images.')
  parser.add_argument('--overwrite-temp-folder', dest='overwrite_temp_folder', action='store_true',
                      help='Overwrite temp_folder if it exists.')
  parser.add_argument('--extract-patches', dest='extract_patches', action='store_true',
                      help='To extract patches of --in-size.')
  parser.add_argument('--no-tensorboard', dest='no_tensorboard', action='store_true',
                      help='To skip TensorBoard logging.')
  parser.add_argument('--delete-temp-folder', dest='delete_temp_folder', action='store_true',
                      help='To delete temp_folder after training.')
  parser.add_argument('--l1-loss-weight', type=float, default=2.5,
                      help='Weight for L1 loss (set to 0 to disable).')
  parser.add_argument('--ssim-loss-weight', type=float, default=0.5,
                      help='Weight for SSIM loss (set to 0 to disable).')
  parser.add_argument('--delta-e-loss-weight', type=float, default=0.02,
                      help='Weight for delta E loss (set to 0 to disable).')
  parser.add_argument('--perceptual-loss-weight', type=float, default=0.01,
                      help='Weight for perceptual loss (set to 0 to disable).')
  parser.add_argument('--cbcr-loss-weight', type=float, default=1.0,
                      help='Weight for CbCr loss (set to 0 to disable).')
  parser.add_argument('--lut-smoothness-loss-weight', type=float, default=0.06,
                      help='Weight for CbCr/RGB LuT TV loss (set to 0 to disable).')
  parser.add_argument('--tm-loss-weight', type=float, default=0.5,
                      help='Weight for LTM/GTM gain map L1 loss (set to 0 to disable).')
  parser.add_argument('--ltm-smoothness-loss-weight', type=float, default=0.6,
                      help='Weight for LTM gain map smoothness loss (set to 0 to disable).')
  parser.add_argument('--luma-energy-consistency-loss-weight', type=float, default=0.2,
                      help='Weight for LTM/GTM luma energy consistency loss (set to 0 to disable).')
  parser.add_argument('--exp-name', type=str, default=None)
  return parser.parse_args()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  args = get_args()
  assert args.in_sz >= 256, 'Expected input size >= 256.'
  print(tabulate([(key, value) for key, value in vars(args).items()], headers=['Argument', 'Value'], tablefmt='grid'))
  if args.l1_loss_weight + args.ssim_loss_weight == 0:
    raise ValueError(f'At least the weight(s) of one of the following loses should be > 0: [l1, ssim].')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.info(f'Using device {device}')

  model_name = 'photofinishing'
  if args.exp_name is not None and args.exp_name != '':
    model_name += f'_{args.exp_name}'

  logging.info(f'Training of photofinishing module -- model name: {model_name} ...')

  os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
  os.makedirs(os.path.join(BASE_DIR, 'config'), exist_ok=True)
  os.makedirs(os.path.join(BASE_DIR, 'checkpoints'), exist_ok=True)
  os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

  if args.load is not None:
    config_base = os.path.splitext(os.path.basename(args.load))[0]
    config = read_json_file(os.path.join(BASE_DIR, 'config', f'{config_base}.json'))
    if config['use_3d_lut'] != args.use_3d_lut:
      net = PhotofinishingModule(device=device, use_3d_lut=config['use_3d_lut'])
      net.load_state_dict(torch.load(args.load, map_location=device, weights_only=True))
      if args.use_3d_lut and not config['use_3d_lut']:
        net.add_3d_lut()
      elif not args.use_3d_lut and config['use_3d_lut']:
        net.remove_3d_lut()
    else:
      net = PhotofinishingModule(device=device, use_3d_lut=args.use_3d_lut)
      net.load_state_dict(torch.load(args.load, map_location=device, weights_only=True))
    logging.info(f'Model loaded from {args.load}')
  else:
    net = PhotofinishingModule(device=device, use_3d_lut=args.use_3d_lut)
  config = {'use_3d_lut': args.use_3d_lut}

  write_json_file(config, os.path.join(BASE_DIR, 'config', f'{model_name}.json'))
  write_json_file(config, os.path.join(BASE_DIR, 'config', f'{model_name}-best.json'))

  net.print_num_of_params()

  net.to(device=device)
  try:
    train_net(
      model=net,
      tr_device=device,
      in_tr_dir=args.in_tr_dir,
      gt_tr_dir=args.gt_tr_dir,
      data_tr_dir=args.data_tr_dir,
      in_val_dir=args.in_vl_dir,
      gt_val_dir=args.gt_vl_dir,
      data_val_dir=args.data_vl_dir,
      epochs=args.epochs,
      batch_size=args.batch_size,
      lr=args.lr,
      l2_reg=args.l2_r,
      exp_name=model_name,
      validation_frequency=args.val_frq,
      in_sz=args.in_sz,
      temp_folder=args.temp_folder,
      l1_loss_weight=args.l1_loss_weight,
      ssim_loss_weight=args.ssim_loss_weight,
      delta_e_loss_weight=args.delta_e_loss_weight,
      perceptual_loss_weight=args.perceptual_loss_weight,
      cbcr_loss_weight=args.cbcr_loss_weight,
      lut_smoothness_loss_weight=args.lut_smoothness_loss_weight,
      luma_energy_consistency_loss_weight=args.luma_energy_consistency_loss_weight,
      tm_loss_weight=args.tm_loss_weight,
      ltm_smoothness_loss_weight=args.ltm_smoothness_loss_weight,
      overwrite_temp_folder=args.overwrite_temp_folder,
      delete_temp_folder=args.delete_temp_folder,
      no_tensorboard=args.no_tensorboard,
      extract_patches=args.extract_patches,
    )
  except KeyboardInterrupt:
    torch.save(net.state_dict(), 'interrupted_checkpoint.pth')
    logging.info('Saved interrupt checkpoint backup')
    try:
      sys.exit(0)
    except SystemExit:
      os._exit(0)
