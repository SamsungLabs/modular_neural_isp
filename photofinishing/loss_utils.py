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

This file contains loss functions used to train the photofinishing module.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from typing import Optional, Dict, Tuple, Union, List
import torch.nn.functional as F
import torch
import torch.nn as nn
from utils.constants import *
from torchvision.models import vgg19, VGG19_Weights


def ssim_loss(pred_img: torch.Tensor, gt_img: torch.Tensor, window_size: Optional[int]=11,
              c1: Optional[float]=0.01 ** 2, c2: Optional[float]=0.03 ** 2) -> torch.Tensor:
  """Differentiable SSIM loss."""

  def create_window(window_sz: int, chs: int):
    coords = torch.arange(window_sz).float() - window_sz // 2
    gauss = torch.exp(-(coords ** 2) / 2.0)
    gauss = gauss / gauss.sum()
    window_2d = gauss[:, None] @ gauss[None, :]
    window = window_2d.expand(chs, 1, window_sz, window_sz).contiguous()
    return window

  channel = pred_img.shape[1]
  window = create_window(window_size, channel).to(pred_img.device)

  mu1 = F.conv2d(pred_img, window, padding=window_size // 2, groups=channel)
  mu2 = F.conv2d(gt_img, window, padding=window_size // 2, groups=channel)

  mu1_sq = mu1.pow(2)
  mu2_sq = mu2.pow(2)
  mu1_mu2 = mu1 * mu2

  sigma1_sq = F.conv2d(pred_img * pred_img, window, padding=window_size // 2, groups=channel) - mu1_sq
  sigma2_sq = F.conv2d(gt_img * gt_img, window, padding=window_size // 2, groups=channel) - mu2_sq
  sigma12 = F.conv2d(pred_img * gt_img, window, padding=window_size // 2, groups=channel) - mu1_mu2

  ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
             ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

  return 1 - ssim_map.mean()

def lsrgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
  """Converts linear sRGB to CIE XYZ"""
  rgb = rgb.clamp(0, 1)
  rgb_to_xyz_matrix = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
  ], device=rgb.device, dtype=rgb.dtype)
  rgb = rgb.permute(0, 2, 3, 1)
  xyz = torch.matmul(rgb, rgb_to_xyz_matrix.T)
  return xyz.permute(0, 3, 1, 2)


def xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor:
  """Converts XYZ to LAB using differentiable approximation."""
  # D65 reference white
  x_n, y_n, z_n = 0.95047, 1.00000, 1.08883
  x = xyz[:, 0, :, :] / x_n
  y = xyz[:, 1, :, :] / y_n
  z = xyz[:, 2, :, :] / z_n

  # Smooth approximation of f(t)
  def f_soft(t: torch.Tensor) -> torch.Tensor:
    delta = 6 / 29
    threshold = delta ** 3
    linear_part = t / (3 * delta ** 2) + 4 / 29
    nonlinear_part = t.clamp(min=1e-6).pow(1 / 3)
    blend = torch.sigmoid(150 * (t - threshold))
    return blend * nonlinear_part + (1 - blend) * linear_part

  fx, fy, fz = f_soft(x), f_soft(y), f_soft(z)
  l = 116 * fy - 16
  a = 500 * (fx - fy)
  b = 200 * (fy - fz)
  return torch.stack([l, a, b], dim=1)

def lsrgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
  """Converts linear sRGB to LAB"""
  return xyz_to_lab(lsrgb_to_xyz(rgb))

def deltae_loss(pred_img: torch.Tensor, gt_img: torch.Tensor) -> torch.Tensor:
  """Delta E76 loss."""
  pred_img = pred_img.clamp(1e-4, 1.0)
  gt_img = gt_img.clamp(1e-4, 1.0)
  lab_pred = lsrgb_to_lab(pred_img)
  lab_gt = lsrgb_to_lab(gt_img)
  delta_e = torch.norm(lab_pred - lab_gt, dim=1)
  delta_e = torch.nan_to_num(delta_e, nan=0.0, posinf=1.0, neginf=0.0)
  return delta_e.mean()


def tv_loss(x: torch.Tensor) -> torch.Tensor:
  """Computes Total Variation (TV) loss to encourage smoothness."""
  diff_h = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
  diff_w = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
  return (diff_h.mean() + diff_w.mean()) / 2


def luma_energy_loss(pre_tm_y: torch.Tensor, gtm_y: torch.Tensor) -> torch.Tensor:
  """Encourages TM to preserve brightness of input luma."""
  gtm_luma_energy = (gtm_y.mean(dim=(1, 2, 3)) - pre_tm_y.mean(dim=(1, 2, 3))).abs().mean()
  return gtm_luma_energy

def tm_loss(gt_y: torch.Tensor, ltm_y: torch.Tensor, gtm_y: Optional[torch.Tensor]=None) -> torch.Tensor:
  """Tone mapping loss.

  Encourages the GTM to roughly preserve the overall tone curve,
  while LTM approximates the final output more closely.
  """
  _, _, h, w = gt_y.shape
  gt_y_ds = F.interpolate(gt_y, size=(h // 8, w // 8), mode='bilinear', align_corners=False)
  gtm_y_ds = F.interpolate(gtm_y, size=(h // 8, w // 8), mode='bilinear', align_corners=False)
  global_loss = F.l1_loss(gtm_y_ds, gt_y_ds)
  local_loss = F.l1_loss(ltm_y, gt_y)
  return 0.6 * global_loss + local_loss


class VGGPerceptualLoss(nn.Module):
  def __init__(self, layers: List[str], resize: Optional[bool] = True):
    super().__init__()
    self._vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    self.layers = layers
    self.resize = resize
    self.layer_map = {
      'relu1_1': 1,
      'relu1_2': 3,
      'relu2_1': 6,
      'relu2_2': 8,
      'relu3_1': 11,
      'relu3_2': 13,
      'relu3_3': 15,
      'relu3_4': 17,
      'relu4_1': 20,
      'relu4_2': 22,
      'relu4_3': 24,
      'relu4_4': 26,
      'relu5_1': 29,
    }
    for param in self._vgg.parameters():
      param.requires_grad = False

    self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

  def forward(self, out_img: torch.Tensor, gt_img: torch.Tensor) -> torch.Tensor:
    if out_img.shape[1] != 3:
      raise ValueError('Input to perceptual loss must have 3 channels (RGB)')

    out_img = self._normalize(out_img)
    gt_img = self._normalize(gt_img)

    loss = 0.0
    x = out_img
    y = gt_img
    for name, layer in self._vgg._modules.items():
      x = layer(x)
      y = layer(y)
      if int(name) in [self.layer_map[l] for l in self.layers]:
        loss += nn.functional.l1_loss(x, y)
    return loss

  def _normalize(self, img: torch.Tensor) -> torch.Tensor:
    if self.resize:
      img = nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
    return (img - self.mean.to(img.device)) / self.std.to(img.device)

class PhotofinishingLoss(nn.Module):
  def __init__(self, l1_weight: Optional[float]=0.0, ssim_weight: Optional[float]=0.0,
               delta_e_weight: Optional[float]=0.0, vgg_weight: Optional[float]=0.0, cbcr_weight: Optional[float]=0.0,
               lut_smooth_weight: Optional[float]=0.0, luma_energy_weight: Optional[float]=0.0,
               ltm_smooth_weight: Optional[float]=0.0, tm_weight: Optional[float]=0.0, device=torch.device('cuda')):
    super().__init__()
    self._weights = {
      'l1': l1_weight,
      'ssim': ssim_weight,
      'delta-e': delta_e_weight,
      'vgg': vgg_weight,
      'cbcr': cbcr_weight,
      'lut-smoothness': lut_smooth_weight,
      'luma-energy-consistency': luma_energy_weight,
      'ltm-smoothness': ltm_smooth_weight,
      'tm': tm_weight,
    }
    self._device = device
    self._vgg_loss = VGGPerceptualLoss(layers=['relu3_3', 'relu4_2']).to(self._device)

  def forward(self, out_img, gt_img, lsrgb_out_img=None, lsrgb_gt_img=None, rgb_lut_out_img=None, cbcr_out_img=None,
              cbcr_gt_img=None, y_gt_img=None, cbcr_lut=None, pre_tm_y=None, ltm_y=None, gtm_y=None, ltm_map=None
              ) -> Tuple[Union[torch.Tensor, float], Dict[str, float]]:

    detailed_loss = {k: 0.0 for k in self._weights}
    detailed_loss['total'] = 0.0
    detailed_loss['psnr'] = 0.0
    total_loss = 0.0

    if self._weights['l1'] > 0:
      l1 = self._weights['l1'] * F.l1_loss(out_img, gt_img)
      total_loss += l1
      detailed_loss['l1'] = l1.item()

    if self._weights['vgg'] > 0:
      perceptual = self._weights['vgg'] * self._vgg_loss(out_img, gt_img)
      total_loss += perceptual
      detailed_loss['vgg'] = perceptual.item()

    if self._weights['luma-energy-consistency'] > 0:
      if any(x is None for x in [pre_tm_y, gtm_y]):
        raise ValueError('Missing y tensors for luma-energy loss')
      energy = self._weights['luma-energy-consistency'] * luma_energy_loss(pre_tm_y, gtm_y)
      total_loss += energy
      detailed_loss['luma-energy-consistency'] = energy.item()

    if self._weights['ltm-smoothness'] > 0:
      if ltm_map is None:
        raise ValueError('Missing ltm_map for smoothness loss')
      smooth = self._weights['ltm-smoothness'] * tv_loss(ltm_map)
      total_loss += smooth
      detailed_loss['ltm-smoothness'] = smooth.item()

    if self._weights['tm'] > 0:
      if ltm_y is None or y_gt_img is None or gtm_y is None:
        raise ValueError('Missing ltm_y, y_gt_img, or gtm_y for TM loss')
      tm = self._weights['tm'] * tm_loss(y_gt_img, ltm_y, gtm_y)
      total_loss += tm
      detailed_loss['tm'] = tm.item()


    if self._weights['lut-smoothness'] > 0:
      if cbcr_lut is None:
        raise ValueError('Missing LUT for smoothness loss')
      lut_smooth = self._weights['lut-smoothness'] * tv_loss(cbcr_lut)
      total_loss += lut_smooth
      detailed_loss['lut-smoothness'] = lut_smooth.item()

    if self._weights['cbcr'] > 0:
      if cbcr_out_img is None or cbcr_gt_img is None:
        raise ValueError('Missing cbcr tensors for loss')
      cbcr = self._weights['cbcr'] * F.l1_loss(cbcr_out_img, cbcr_gt_img)
      total_loss += cbcr
      detailed_loss['cbcr'] = cbcr.item()

    if self._weights['delta-e'] > 0:
      if lsrgb_out_img is None or lsrgb_gt_img is None:
        raise ValueError('Missing linear sRGB images for delta-E loss')
      if rgb_lut_out_img is not None:
        rgb_lut_delta_e = 0.5 * deltae_loss(rgb_lut_out_img, lsrgb_gt_img)
      else:
        rgb_lut_delta_e = 0.0
      delta = self._weights['delta-e'] * (deltae_loss(lsrgb_out_img, lsrgb_gt_img) + rgb_lut_delta_e)
      total_loss += delta
      detailed_loss['delta-e'] = delta.item()

    if self._weights['ssim'] > 0:
      ssim_val = self._weights['ssim'] * ssim_loss(out_img, gt_img)
      total_loss += ssim_val
      detailed_loss['ssim'] = ssim_val.item()

    detailed_loss['total'] = total_loss.item()

    # PSNR (for reporting only)
    mse = torch.mean((out_img - gt_img) ** 2)
    if mse == 0:
      mse += EPS
    detailed_loss['psnr'] = -10 * torch.log10(mse).item()

    return total_loss, detailed_loss