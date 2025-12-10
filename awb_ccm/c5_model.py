"""
Author(s):
Mahmoud Afifi (m.afifi1@samsung.com, m.3afifi@gmail.com)

Implementation of a modified version of:
"Cross-Camera Convolutional Color Constancy", ICCV 2021 (https://arxiv.org/abs/2011.11890).
"""

import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
from utils.vect_utils import vect_norm
from utils.img_utils import rgb_to_uv
from utils.constants import *
from typing import Tuple, Union, Optional

class IllumEstimator(nn.Module):
  def __init__(self, device: torch.device=torch.device('cuda')):
    super().__init__()
    self._boundary_values = [-2.85, 2.85]
    self._hist_size = 48
    coords = np.arange(-(self._hist_size - 1) / 2, (self._hist_size - 1) / 2 + 1)
    u, v = np.meshgrid(coords, coords[::-1])
    self._u_coord = torch.from_numpy(u).float().unsqueeze(0).to(device)
    self._v_coord = torch.from_numpy(v).float().unsqueeze(0).to(device)
    self._u_coord.requires_grad = False
    self._v_coord.requires_grad = False

    initial_conv = 8
    max_conv = 32
    depth = 4

    self._encoder = Encoder(4, initial_conv, max_conv, depth).to(device)
    self._bias_decoder = Decoder(1, initial_conv, max_conv, depth).to(device)
    self._filter_decoder = Decoder(2, initial_conv, max_conv, depth).to(device)
    self._bottleneck = DoubleConvBlock(
      in_depth=min(initial_conv * 2 ** (depth - 1), max_conv), mid_depth=min(initial_conv * 2 ** depth, max_conv),
      out_depth=min(initial_conv * 2 ** (depth - 1), max_conv), pooling=False, norm_type='in').to(device)

    self._softmax = nn.Softmax(dim=-1)

  def forward(self, hist: torch.Tensor, inference: Optional[bool]=False) -> Union[torch.Tensor, Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    latent, encoder_out = self._encoder(hist)
    latent = self._bottleneck(latent)
    bias = torch.squeeze(self._bias_decoder(latent, encoder_out))
    filt = self._filter_decoder(latent, encoder_out)
    hist_fft = fft.rfft2(hist[:, :2])
    filt_fft = fft.rfft2(filt)
    proc_hist = fft.irfft2(hist_fft * filt_fft)
    heat_map = torch.reshape(
      self._softmax(torch.reshape(torch.clamp(proc_hist.sum(dim=1) + bias, -100, 100),
                                  (proc_hist.shape[0], -1))), proc_hist[:, 0, ...].shape
    )

    u = (heat_map * self._u_coord).sum(dim=[-1, -2])
    v = (heat_map * self._v_coord).sum(dim=[-1, -2])
    scale = (self._boundary_values[1] - self._boundary_values[0]) / self._hist_size
    u *= scale
    v *= scale
    uv = torch.stack([u, v], dim=1)
    rb = torch.exp(-uv)
    illum_rgb = torch.stack([rb[:, 0], torch.ones_like(rb[:, 0]), rb[:, 1]], dim=-1)
    illum_rgb = illum_rgb / vect_norm(illum_rgb)
    if inference:
      return illum_rgb
    else:
      return illum_rgb, heat_map, filt, bias

  def print_num_of_params(self, show_message: Optional[bool]=True) -> str:
    """Prints number of parameters in the model."""
    total_num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    message = f'Total number of network params: {total_num_params}'
    if show_message:
      print(message)
    return message + '\n'

  @staticmethod
  def get_hist_colors(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gets valid chroma and color values for histogram computation."""
    img = np.reshape(img, (-1, 3))
    uv_chroma = rgb_to_uv(img)
    valid_pixels = np.sum(img, axis=1) > EPS
    valid_chroma = uv_chroma[valid_pixels, :]
    valid_colors = img[valid_pixels, :]
    return valid_chroma, valid_colors

  @staticmethod
  def compute_histogram(chroma: np.ndarray, rgb: Optional[np.ndarray]=None, bins: Optional[int]=48) -> np.ndarray:
    """Computes log-chroma histogram of a given log-chroma values."""
    hist_boundary = [-2.85, 2.85]
    eps = np.sum(np.abs(hist_boundary)) / (bins - 1)
    hist_boundary = np.sort(hist_boundary)
    bins_u = np.arange(hist_boundary[0], hist_boundary[1] + eps / 2, eps)
    bins_v = np.flip(bins_u)
    if rgb is None:
      intensity = np.ones(chroma.shape[0])
    else:
      intensity = np.sqrt(np.sum(rgb ** 2, axis=1))
    # differences in log_u space
    diff_u = np.abs(np.tile(chroma[:, 0], (len(bins_u), 1)).transpose() -
                    np.tile(bins_u, (len(chroma[:, 0]), 1)))

    # differences in log_v space
    diff_v = np.abs(np.tile(chroma[:, 1], (len(bins_v), 1)).transpose() -
                    np.tile(bins_v, (len(chroma[:, 1]), 1)))

    # counts only U values that is higher than the threshold value
    diff_u[diff_u > eps] = 0
    diff_u[diff_u != 0] = 1

    # counts only V values that is higher than the threshold value
    diff_v[diff_v > eps] = 0
    diff_v[diff_v != 0] = 1

    weighted_diff_v = np.tile(intensity, (len(bins_v), 1)) * diff_v.transpose()
    hist = np.matmul(weighted_diff_v, diff_u)
    norm_factor = np.sum(hist) + EPS
    hist = np.sqrt(hist / norm_factor)
    return hist

  @staticmethod
  def get_uv_coords(bins: Optional[int]=48) -> Tuple[np.ndarray, np.ndarray]:
    u_coord, v_coord = np.meshgrid(np.arange(-(bins - 1) / 2, ((bins - 1) / 2) + 1),
                                   np.arange((bins - 1) / 2, (-(bins - 1) / 2) - 1, -1))
    u_coord = (u_coord + ((bins - 1) / 2)) / (bins - 1)
    v_coord = (v_coord + ((bins - 1) / 2)) / (bins - 1)
    return u_coord, v_coord


class ConvBlock(nn.Module):
  def __init__(self, kernel: int, in_depth: int, conv_depth: int, stride: Optional[int]=1,
               padding: Optional[int]=1, normalization: Optional[bool]=False, norm_type: Optional[str]='bn',
               pooling: Optional[bool]=False, activation: Optional[bool]=True, dilation: Optional[int]=1,
               return_before_pooling: Optional[bool]=False):
    super().__init__()
    self._conv = nn.Conv2d(
      in_channels=in_depth, out_channels=conv_depth, kernel_size=kernel, stride=stride, padding=padding,
      dilation=dilation, padding_mode='replicate')
    nn.init.kaiming_normal_(self._conv.weight)
    nn.init.zeros_(self._conv.bias)

    self._activation_layer = nn.LeakyReLU(inplace=False) if activation else None

    if normalization:
      self._norm_layer = nn.BatchNorm2d(conv_depth) if norm_type == 'bn' else nn.InstanceNorm2d(
        conv_depth, affine=False)
    else:
      self._norm_layer = None

    self._pooling_layer = nn.MaxPool2d(2, stride=2) if pooling else None
    self._return_before_pooling = return_before_pooling

  def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    x = self._conv(x)
    if self._norm_layer is not None:
      x = self._norm_layer(x)
    if self._activation_layer is not None:
      x = self._activation_layer(x)
    if self._pooling_layer is not None:
      pooled = self._pooling_layer(x)
      return (pooled, x) if self._return_before_pooling else pooled
    else:
      return (x, x) if self._return_before_pooling else x


class DoubleConvBlock(nn.Module):
  def __init__(self, in_depth: int, out_depth: int, mid_depth: Optional[int]=None, kernel: Optional[int]=3,
               stride: Optional[int]=1, normalization: Optional[bool]=False, norm_type: Optional[str]='bn',
               pooling: Optional[bool]=True, return_before_pooling: Optional[bool]=False):
    super().__init__()
    mid_depth = mid_depth or out_depth
    norm_flag = True if normalization else False
    self._conv1 = ConvBlock(kernel, in_depth, mid_depth, stride=stride, normalization=False, norm_type=norm_type)
    self._conv2 = ConvBlock(kernel, mid_depth, out_depth, stride=stride, normalization=norm_flag, norm_type=norm_type,
                            pooling=pooling, return_before_pooling=return_before_pooling)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._conv2(self._conv1(x))


class Encoder(nn.Module):
  def __init__(self, in_channels: int, first_conv_depth: int, max_conv_depth: int, depth: Optional[int]=4):
    super().__init__()
    self._blocks = nn.ModuleList()
    for i in range(depth):
      in_depth = in_channels if i == 0 else min(first_conv_depth * 2 ** (i - 1), max_conv_depth)
      out_depth = min(first_conv_depth * 2 ** i, max_conv_depth)
      norm = (i % 2 == 0)
      block = DoubleConvBlock(
        in_depth, out_depth, normalization=norm, norm_type='bn', return_before_pooling=True)
      self._blocks.append(block)

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    skips = []
    for block in self._blocks:
      x, before_pooling = block(x)
      skips.append(before_pooling)
    skips.reverse()
    return x, skips


class Decoder(nn.Module):
  def __init__(self, output_channels: int, encoder_first_conv_depth: int, encoder_max_conv_depth: int, depth: int):
    super().__init__()

    self._upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self._blocks = nn.ModuleList()

    for i in range(depth):
      in_depth = int(min(encoder_first_conv_depth * 2 ** (depth - i), encoder_max_conv_depth * 2))
      mid_depth = int(min(encoder_first_conv_depth * 2 ** (depth - 1 - i), encoder_max_conv_depth))
      out_depth = int(min(encoder_first_conv_depth * 2 ** (depth - 2 - i), encoder_max_conv_depth))
      block = DoubleConvBlock(
        in_depth=in_depth, out_depth=out_depth, mid_depth=mid_depth, normalization=True, norm_type='in',
        pooling=False)
      self._blocks.append(block)
    self._final_conv = ConvBlock(3, out_depth, output_channels, activation=False)

  def forward(self, x: torch.Tensor, skips: torch.Tensor) -> torch.Tensor:
    for block, skip in zip(self._blocks, skips):
      x = self._upsample(x)
      x = torch.cat([skip, x], dim=1)
      x = block(x)
    return self._final_conv(x)
