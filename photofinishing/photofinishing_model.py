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

This file contains model architecture for photofinishing module.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict

import time
from utils.constants import *


def run_on_cpu_if_mps(*tensors):
  """If any tensor is on MPS, move all tensors to CPU and return (run_on_cpu, cpu_tensors)."""
  run_on_cpu = any(
    torch.is_tensor(t) and t.device.type == 'mps'
    for t in tensors
  )
  if not run_on_cpu:
    return False, tensors
  cpu_tensors = tuple(
    t.detach().cpu() if torch.is_tensor(t) else t
    for t in tensors
  )
  return True, cpu_tensors


class MultiBranchConvBlock(nn.Module):
  """Multi-branch conv block with dilation and depthwise convs."""

  def __init__(self, channels: int, act_func: nn.Module):
    super().__init__()

    def depthwise_branch(kernel_size: int, dilation: int) -> nn.Sequential:
      pad = dilation * (kernel_size // 2)
      return nn.Sequential(
        nn.ReflectionPad2d(pad),
        nn.Conv2d(
          channels, channels,
          kernel_size=kernel_size,
          padding=0,
          dilation=dilation,
          groups=channels,
          bias=True
        ),
        act_func.__class__(*act_func.parameters())
      )

    self._branch1 = depthwise_branch(kernel_size=3, dilation=1)
    self._branch2 = depthwise_branch(kernel_size=3, dilation=2)
    self._branch3 = depthwise_branch(kernel_size=5, dilation=1)
    self._fuse = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)

  def forward(self, x):
    out = self._branch1(x) + self._branch2(x) + self._branch3(x)
    out = self._fuse(out)
    return out


class CoordinateAttention(nn.Module):
  """Coordinate Attention for Efficient Mobile Network Design."""

  def __init__(self, in_channels: int, act_func: nn.Module, reduction: Optional[int] = 4):
    super().__init__()
    reduced_channels = max(8, in_channels // reduction)
    self._pool_h = nn.AdaptiveAvgPool2d((None, 1))
    self._pool_w = nn.AdaptiveAvgPool2d((1, None))
    self._conv1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
    self._n1 = nn.GroupNorm(num_groups=4, num_channels=reduced_channels)
    self._act = act_func
    self._conv_h = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
    self._conv_w = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)

  def forward(self, x):
    b, c, h, w = x.size()
    x_h = self._pool_h(x)
    x_w = self._pool_w(x).permute(0, 1, 3, 2)
    y = torch.cat([x_h, x_w], dim=2)
    y = self._conv1(y)
    y = self._n1(y)
    y = self._act(y)
    x_h, x_w = torch.split(y, [h, w], dim=2)
    x_w = x_w.permute(0, 1, 3, 2)
    a_h = self._conv_h(x_h).sigmoid()
    a_w = self._conv_w(x_w).sigmoid()
    out = x * a_h * a_w
    return out


class LuTNet(nn.Module):
  """Neural network for predicting the CbCr chroma lookup table (LuT)."""

  def __init__(self, act_func: nn.Module, lut_size: Optional[int] = CBCR_LUT_SIZE,
               input_size: Optional[int] = LUT_NET_INPUT_SIZE,
               bottleneck_dim: Optional[int] = LUT_NET_BOTTLENECK_CHANNELS):
    super().__init__()
    self._lut_size = lut_size
    self._input_size = input_size
    self.register_buffer('identity_grid', LuTNet._identity_cbcr_meshgrid(self._lut_size))
    self._base_lut = nn.Parameter(LuTNet._identity_cbcr_meshgrid(self._lut_size))
    self._bottleneck_dim = bottleneck_dim
    self._act_func = act_func
    self._hist_net = nn.Sequential(
      nn.Conv2d(1, 8, kernel_size=3, padding='same', padding_mode='reflect'),
      nn.GroupNorm(num_groups=2, num_channels=8),
      self._act_func.__class__(*self._act_func.parameters()),
      nn.Conv2d(8, 8, kernel_size=3, padding='same', padding_mode='reflect'),
      self._act_func.__class__(*self._act_func.parameters()),
      nn.Conv2d(8, 4, kernel_size=1),
      nn.Tanh()
    )
    self._enc1 = nn.Sequential(
      nn.Conv2d(6, self._bottleneck_dim, kernel_size=3, padding='same', padding_mode='reflect'),
      nn.GroupNorm(num_groups=4, num_channels=self._bottleneck_dim),
      self._act_func.__class__(*self._act_func.parameters()),
    )
    self._enc2 = nn.Sequential(
      nn.Conv2d(self._bottleneck_dim, self._bottleneck_dim, kernel_size=3, padding='same', padding_mode='reflect'),
      self._act_func.__class__(*self._act_func.parameters()),
      MultiBranchConvBlock(self._bottleneck_dim, act_func=self._act_func),
    )
    self._enc3 = nn.Sequential(
      nn.Conv2d(self._bottleneck_dim, self._bottleneck_dim, kernel_size=3, padding='same', padding_mode='reflect'),
      self._act_func.__class__(*self._act_func.parameters()),
    )

    self._coord_attn = CoordinateAttention(self._bottleneck_dim, act_func=self._act_func)

    self._dec1 = nn.Sequential(
      nn.Conv2d(self._bottleneck_dim, self._bottleneck_dim, kernel_size=3, padding='same', padding_mode='reflect'),
      nn.GroupNorm(num_groups=4, num_channels=self._bottleneck_dim),
      self._act_func.__class__(*self._act_func.parameters()),
    )

    self._dec2 = nn.Sequential(
      nn.Conv2d(self._bottleneck_dim, self._bottleneck_dim, kernel_size=3, padding='same', padding_mode='reflect'),
      self._act_func.__class__(*self._act_func.parameters()),
      MultiBranchConvBlock(self._bottleneck_dim, act_func=self._act_func),
    )
    self._dec3 = nn.Sequential(
      nn.Conv2d(self._bottleneck_dim, self._bottleneck_dim, kernel_size=3, padding='same', padding_mode='reflect'),
      self._act_func.__class__(*self._act_func.parameters()),
    )
    self._out = nn.Sequential(
      nn.Conv2d(self._bottleneck_dim, 2, kernel_size=1),
      nn.Tanh()
    )

    self._y_net = nn.Sequential(
      nn.Conv2d(1, 8, kernel_size=3, padding='same', padding_mode='reflect'),
      nn.GroupNorm(num_groups=2, num_channels=8),
      self._act_func.__class__(*self._act_func.parameters()),
      CoordinateAttention(8, reduction=2, act_func=self._act_func),
      MultiBranchConvBlock(channels=8, act_func=self._act_func),
      nn.AdaptiveAvgPool2d((8, 8)),
      nn.Conv2d(8, 8, kernel_size=3, padding=1),
      self._act_func.__class__(*self._act_func.parameters()),
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
      nn.Linear(8, self._bottleneck_dim),
      nn.Sigmoid()
    )

  def get_cbcr_lut_size(self) -> int:
    """Returns CbCr LuT number of bins."""
    return self._lut_size

  def get_num_of_params(self) -> int:
    """Returns total number of parameters."""
    total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    return total_params

  def forward(self, ycbcr: torch.Tensor) -> torch.Tensor:
    ycbcr = F.interpolate(ycbcr, size=(self._input_size, self._input_size), mode='bilinear', align_corners=True)
    b = ycbcr.size(0)
    y, cbcr = ycbcr[:, :1, ...], ycbcr[:, 1:, ...]

    hist = self._differentiable_cbcr_histogram(cbcr, bins=self._lut_size)

    identity_grid = self.identity_grid.expand(b, -1, -1, -1)
    x = torch.cat([self._hist_net(hist), identity_grid], dim=1)

    # Encoder
    x1 = self._enc1(x)
    x2 = self._enc2(x1)
    x3 = self._enc3(x2)
    x3 = self._coord_attn(x3)

    # Apply luminance-guided attention
    attn = self._y_net(y)
    x3 = x3 * attn.unsqueeze(-1).unsqueeze(-1)

    # Decoder
    d1 = self._dec1(x3)
    d2 = self._dec2(d1)
    d3 = self._dec3(d2)
    lut = self._out(d3)
    return self._base_lut.expand(b, -1, -1, -1) + lut

  @staticmethod
  def _differentiable_cbcr_histogram(cbcr, bins=CBCR_LUT_SIZE, min_val=-0.5, max_val=0.5, sigma=0.075):
    """Computes differentiable CbCr histogram."""

    b = cbcr.shape[0]
    cb = cbcr[:, 0, :, :].reshape(b, -1, 1, 1)
    cr = cbcr[:, 1, :, :].reshape(b, -1, 1, 1)

    edges = torch.linspace(min_val, max_val, bins, device=cbcr.device)
    cb_centers = edges.view(1, 1, bins, 1)
    cr_centers = edges.view(1, 1, 1, bins)

    cb_diff = (cb - cb_centers) ** 2
    cr_diff = (cr - cr_centers) ** 2
    weight = torch.exp(-(cb_diff + cr_diff) / (2 * sigma ** 2))

    hist = weight.sum(dim=1, keepdim=True)
    hist = hist / (hist.sum(dim=(2, 3), keepdim=True) + EPS)
    hist = torch.sqrt(hist)
    return hist

  @staticmethod
  def _identity_cbcr_meshgrid(bins=CBCR_LUT_SIZE, min_val=-0.5, max_val=0.5):
    cb = torch.linspace(min_val, max_val, bins)
    cr = torch.linspace(min_val, max_val, bins)
    mesh_cb, mesh_cr = torch.meshgrid(cb, cr, indexing='ij')
    grid = torch.stack([mesh_cb, mesh_cr], dim=0).unsqueeze(0)
    return grid


class MultiScaleGuidanceNet(nn.Module):
  """Multi-scale guidance network."""

  def __init__(self, base_channels, act_func):
    super().__init__()
    self._base_channels = base_channels
    self._act_func = act_func
    self._guide_conv_low = self._guide_conv()
    self._guide_conv_mid = self._guide_conv()
    self._guide_conv_high = self._guide_conv()
    self._fusion = nn.Sequential(
      nn.Conv2d(self._base_channels * 3, self._base_channels, kernel_size=3, padding=1, padding_mode='reflect'),
      self._act_func.__class__(*self._act_func.parameters()),
      nn.Conv2d(self._base_channels, 1, kernel_size=1),
      nn.Tanh()
    )

  def _guide_conv(self):
    return nn.Sequential(
      nn.Conv2d(1, self._base_channels // 2, kernel_size=3, padding=1, padding_mode='reflect'),
      nn.GroupNorm(num_groups=2, num_channels=self._base_channels // 2),
      self._act_func.__class__(*self._act_func.parameters()),
      MultiBranchConvBlock(self._base_channels // 2, act_func=self._act_func),
      CoordinateAttention(self._base_channels // 2, act_func=self._act_func),
      nn.Conv2d(self._base_channels // 2, self._base_channels, kernel_size=3, padding=1, padding_mode='reflect'),
      self._act_func.__class__(*self._act_func.parameters()),
      nn.Conv2d(self._base_channels, self._base_channels, kernel_size=3, padding=1, padding_mode='reflect'),
      self._act_func.__class__(*self._act_func.parameters()),
      nn.Conv2d(self._base_channels, self._base_channels, kernel_size=3, padding=1, padding_mode='reflect'),
      self._act_func.__class__(*self._act_func.parameters()),
      nn.Conv2d(self._base_channels, self._base_channels, kernel_size=1)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x_low = F.avg_pool2d(x, 4)
    x_mid = F.avg_pool2d(x, 2)
    g_low = F.interpolate(self._guide_conv_low(x_low), size=x.shape[-2:], mode='bilinear', align_corners=True)
    g_mid = F.interpolate(self._guide_conv_mid(x_mid), size=x.shape[-2:], mode='bilinear', align_corners=True)
    g_high = self._guide_conv_high(x)
    return self._fusion(torch.cat([g_low, g_mid, g_high], dim=1))


class LocalToneMappingNet(nn.Module):
  """Neural network for predicting local tone-mapping coefficients."""

  def __init__(self, act_func: nn.Module, base_channels: Optional[int] = LTM_NET_BASE_CHANNELS,
               num_coeffs: Optional[int] = 5,
               grid_depth: Optional[int] = LTM_GRID_DEPTH, grid_size: Optional[int] = LTM_GRID_SIZE,
               input_size: Optional[int] = LTM_NET_INPUT_SIZE):
    super().__init__()
    self._grid_size = grid_size
    self._grid_depth = grid_depth
    self._input_size = input_size
    self._num_coeffs = num_coeffs
    self._base_channels = base_channels
    self._act_func = act_func
    self._guide_conv = MultiScaleGuidanceNet(base_channels=self._base_channels // 2, act_func=self._act_func)
    self._grid_net = nn.Sequential(
      nn.Conv2d(6, self._base_channels // 2, kernel_size=3, padding=1, padding_mode='reflect'),
      nn.GroupNorm(num_groups=2, num_channels=self._base_channels // 2),
      self._act_func.__class__(*self._act_func.parameters()),

      MultiBranchConvBlock(self._base_channels // 2, act_func=self._act_func),
      CoordinateAttention(self._base_channels // 2, act_func=self._act_func),

      nn.Conv2d(self._base_channels // 2, self._base_channels, kernel_size=3, padding=1, padding_mode='reflect'),
      self._act_func.__class__(*self._act_func.parameters()),
      nn.Conv2d(self._base_channels, self._base_channels, kernel_size=3, padding=1, padding_mode='reflect'),
      nn.GroupNorm(num_groups=4, num_channels=self._base_channels),
      self._act_func.__class__(*self._act_func.parameters()),
      nn.AdaptiveAvgPool2d((self._grid_size, self._grid_size)),
      nn.Conv2d(self._base_channels, self._base_channels, kernel_size=3, padding=1, padding_mode='reflect'),
      nn.GroupNorm(num_groups=4, num_channels=self._base_channels),
      self._act_func.__class__(*self._act_func.parameters()),
      nn.Conv2d(self._base_channels, self._num_coeffs * self._grid_depth, kernel_size=1),
      nn.Softplus()
    )

  def get_num_of_params(self) -> int:
    """Returns total number of parameters."""
    total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    return total_params

  @staticmethod
  def _bilateral_solver(guide: torch.Tensor, coeff_map: torch.Tensor,
                        k: Optional[int] = 7, sigma_spatial: Optional[float] = 3.0, sigma_luma: Optional[float] = 0.01,
                        lam: Optional[float] = 1e-3, n_iter: Optional[int] = BILATERAL_SOLVER_ITERS,
                        omega: Optional[float] = 1.6):
    """GPU-accelerated bilateral solver (runs fully on CPU if MPS is detected)."""

    orig_device = coeff_map.device
    run_on_cpu = (orig_device.type == 'mps')

    if run_on_cpu:
      guide = guide.detach().cpu()
      coeff_map = coeff_map.detach().cpu()

    if guide.shape[1] == 3:
      guide = 0.2989 * guide[:, 0:1] + 0.5870 * guide[:, 1:2] + 0.1140 * guide[:, 2:3]

    b, c, h, w = coeff_map.shape
    pad = k // 2
    device = coeff_map.device
    dtype = coeff_map.dtype

    # pre-computed bilateral weights (fixed for all iterations).
    guide_p = F.pad(guide, (pad, pad, pad, pad), mode='reflect')
    neigh_guide = F.unfold(guide_p, kernel_size=k, padding=0).view(b, 1, k * k, h, w)
    center_guide = guide.unsqueeze(2)
    diff2 = (neigh_guide - center_guide).pow(2)
    range_w = torch.exp(-diff2 / (2 * (sigma_luma ** 2)))
    yy, xx = torch.meshgrid(
      torch.arange(-pad, pad + 1, device=device, dtype=dtype),
      torch.arange(-pad, pad + 1, device=device, dtype=dtype),
      indexing="ij"
    )
    spatial = torch.exp(-(xx ** 2 + yy ** 2) / (2 * (sigma_spatial ** 2)))
    spatial = spatial.reshape(1, 1, k * k, 1, 1)
    w_b = range_w * spatial
    w_b = w_b / w_b.sum(dim=2, keepdim=True).clamp_min(EPS)
    out = coeff_map.clone()
    inv_alpha = 1.0 / (lam + 1.0)

    for _ in range(n_iter):
      out_p = F.pad(out, (pad, pad, pad, pad), mode='reflect')
      neigh_coeff = F.unfold(out_p, kernel_size=k, padding=0).view(b, c, k * k, h, w)
      smooth = (neigh_coeff * w_b).sum(dim=2)
      target = (lam * coeff_map + smooth) * inv_alpha
      out = out + omega * (target - out)

    if run_on_cpu:
      out = out.to(orig_device)

    return out

  def _forward_training_mode(self, x: torch.Tensor, x_gtm: torch.Tensor) -> torch.Tensor:
    y = x[:, 0:1, ...]
    guide = self._guide_conv(y)
    x_in = torch.concatenate([x, x_gtm], dim=1)
    _, _, h, w = x_in.shape
    if h != self._input_size or w != self._input_size:
      x_ds = F.interpolate(x_in, size=(self._input_size, self._input_size), mode='bilinear', align_corners=True)
    else:
      x_ds = x_in
    grid = self._grid_net(x_ds)
    grid = grid.view(-1, self._num_coeffs, self._grid_depth, self._grid_size, self._grid_size)
    a_map = self._bilateral_slice(grid, guide)
    return a_map

  def forward(self, x: torch.Tensor, x_gtm: torch.Tensor, post_process_ltm: Optional[bool] = False,
              solver_iter: Optional[int] = BILATERAL_SOLVER_ITERS,
              training_mode: Optional[bool] = False) -> torch.Tensor:
    if training_mode:
      return self._forward_training_mode(x=x, x_gtm=x_gtm)
    if post_process_ltm:
      # Multi-scale and refinement of the LTM coeffs to mitigate potential halo artifacts
      scales = [1.0, 0.5, 0.25, 0.125, 0.0625]
      coeffs = []
      for s in scales:
        if s != 1.0:
          x_s = F.interpolate(x, scale_factor=s, mode='bilinear', align_corners=True)
          x_gtm_s = F.interpolate(x_gtm, scale_factor=s, mode='bilinear', align_corners=True)
        else:
          x_s, x_gtm_s = x, x_gtm
        guide_s = self._blur_tensor(self._guide_conv(x_s[:, 0:1, ...]))
        x_in_s = torch.cat([x_s, x_gtm_s], dim=1)
        x_ds_s = F.interpolate(x_in_s, size=(self._input_size, self._input_size), mode='bilinear', align_corners=True)
        grid_s = self._grid_net(self._blur_tensor(x_ds_s))
        grid_s = grid_s.view(-1, self._num_coeffs, self._grid_depth, self._grid_size, self._grid_size)
        a_map_s = self._bilateral_slice(grid_s, guide_s)
        if s != 1.0:
          a_map_s = F.interpolate(a_map_s, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        coeffs.append(a_map_s)
      a_map = sum(coeffs) / len(coeffs)
      with torch.no_grad():
        a_map = self._bilateral_solver(guide=x, coeff_map=a_map, k=7, sigma_spatial=3.0, sigma_luma=0.008,
                                       lam=5e-4, n_iter=solver_iter, omega=1.6)
    else:
      y = x[:, 0:1, ...]
      guide = self._guide_conv(y)
      x_in = torch.concatenate([x, x_gtm], dim=1)
      _, _, h, w = x_in.shape
      if h != self._input_size or w != self._input_size:
        x_ds = F.interpolate(x_in, size=(self._input_size, self._input_size), mode='bilinear', align_corners=True)
      else:
        x_ds = x_in
      grid = self._grid_net(x_ds)
      grid = grid.view(-1, self._num_coeffs, self._grid_depth, self._grid_size, self._grid_size)
      a_map = self._bilateral_slice(grid, guide)
    return a_map

  @staticmethod
  def _blur_tensor(t, ksize=5):
    pad = (ksize // 2, ksize // 2, ksize // 2, ksize // 2)
    return F.avg_pool2d(F.pad(t, pad, mode='reflect'), kernel_size=ksize, stride=1)

  @staticmethod
  def _bilateral_slice(grid: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
    run_on_cpu, (grid, guide) = run_on_cpu_if_mps(grid, guide)
    b = grid.shape[0]
    device = guide.device
    h = torch.linspace(-1, 1, guide.shape[2], device=device)
    w = torch.linspace(-1, 1, guide.shape[3], device=device)
    grid_h, grid_w = torch.meshgrid(h, w, indexing='ij')
    grid_h = grid_h.expand(b, -1, -1).unsqueeze(1)
    grid_w = grid_w.expand(b, -1, -1).unsqueeze(1)
    coords = torch.cat([grid_w, grid_h, guide], dim=1).permute(0, 2, 3, 1).unsqueeze(1)
    out = F.grid_sample(grid, coords, align_corners=True, padding_mode='border').squeeze(2)
    if run_on_cpu:
      out = out.to('mps')
    return out


class GlobalToneMappingNet(nn.Module):
  """Neural network for predicting global tone-mapping coefficients."""

  def __init__(self, act_func: nn.Module, input_size: Optional[int] = GTM_NET_INPUT_SIZE,
               base_channels: Optional[int] = GTM_NET_BASE_CHANNELS):
    super().__init__()
    self._input_size = input_size
    self._base_channels = base_channels
    self._act_func = act_func
    self._gtm_net = nn.Sequential(
      nn.Conv2d(3, self._base_channels // 2, kernel_size=3, padding=1, padding_mode='reflect'),
      nn.GroupNorm(num_groups=2, num_channels=self._base_channels // 2),
      self._act_func.__class__(*self._act_func.parameters()),
      MultiBranchConvBlock(self._base_channels // 2, act_func=self._act_func),
      CoordinateAttention(self._base_channels // 2, act_func=self._act_func),
      nn.Conv2d(self._base_channels // 2, self._base_channels, kernel_size=3, padding=1, padding_mode='reflect'),
      self._act_func.__class__(*self._act_func.parameters()),
      nn.Conv2d(self._base_channels, self._base_channels, kernel_size=3, padding=1, padding_mode='reflect'),
      self._act_func.__class__(*self._act_func.parameters()),
      nn.AdaptiveAvgPool2d((self._input_size // 8, self._input_size // 8)),
      nn.Conv2d(self._base_channels, self._base_channels * 2, kernel_size=3, padding=1, padding_mode='reflect'),
      self._act_func.__class__(*self._act_func.parameters()),
      nn.AdaptiveAvgPool2d((4, 4)),
      nn.Conv2d(self._base_channels * 2, self._base_channels * 2, kernel_size=3, padding=1, padding_mode='reflect'),
      self._act_func.__class__(*self._act_func.parameters()),
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
      nn.Linear(self._base_channels * 2, 3),
      nn.Softplus()
    )

  def get_num_of_params(self) -> int:
    """Returns total number of parameters."""
    total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    return total_params

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Return global tone mapping parameters (B, 3)."""
    x_thumb = F.interpolate(x, size=(self._input_size, self._input_size), mode='bilinear', align_corners=True)
    return self._gtm_net(x_thumb)


class BaseNet(nn.Module):
  """Base network for estimating digital gain and gamma coefficients."""

  def __init__(self, act_func: nn.Module, input_size: int, base_channels: int):
    super().__init__()
    self._net = nn.Sequential(
      nn.Conv2d(3, base_channels // 2, kernel_size=3, padding=1, padding_mode='reflect'),
      nn.GroupNorm(num_groups=2, num_channels=base_channels // 2),
      act_func.__class__(*act_func.parameters()),
      MultiBranchConvBlock(base_channels // 2, act_func=act_func),
      CoordinateAttention(base_channels // 2, act_func=act_func),
      nn.Conv2d(base_channels // 2, base_channels, kernel_size=3, padding=1, padding_mode='reflect'),
      act_func.__class__(*act_func.parameters()),
      nn.AdaptiveAvgPool2d((input_size // 4, input_size // 4)),
      nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, padding_mode='reflect'),
      act_func.__class__(*act_func.parameters()),
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
      nn.Linear(base_channels, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    """Forward function."""
    return self._net(x)


class GainNet(nn.Module):
  """Neural network for predicting digital gain scale."""

  def __init__(self, act_func: nn.Module, input_size: Optional[int] = GAIN_NET_INPUT_SIZE,
               base_channels: Optional[int] = GAIN_NET_BASE_CHANNELS,
               gain_min: Optional[float] = GAIN_MIN, gain_max: Optional[float] = GAIN_MAX):
    super().__init__()
    self._input_size = input_size
    self._gain_min = gain_min
    self._gain_max = gain_max
    self._base_channels = base_channels
    self._act_func = act_func
    self._gain_net = BaseNet(act_func=self._act_func, input_size=self._input_size, base_channels=self._base_channels)

  def get_num_of_params(self) -> int:
    """Returns total number of parameters."""
    total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    return total_params

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(x, size=(self._input_size, self._input_size), mode='bilinear', align_corners=True)
    gain_scale = self._gain_net(x)
    gain_factor = self._gain_min + (self._gain_max - self._gain_min) * gain_scale
    return gain_factor.view(-1, 1, 1, 1)


class GammaNet(nn.Module):
  """Neural network for predicting the gamma correction coefficient."""

  def __init__(self, act_func: nn.Module, input_size: Optional[int] = GAMMA_NET_INPUT_SIZE,
               base_channels: Optional[int] = GAMMA_NET_BASE_CHANNELS,
               gamma_min: Optional[float] = GAMMA_MIN, gamma_max: Optional[float] = GAMMA_MAX):
    super().__init__()
    self._input_size = input_size
    self._gamma_min = gamma_min
    self._gamma_max = gamma_max
    self._base_channels = base_channels
    self._act_func = act_func
    self._gamma_net = BaseNet(act_func=self._act_func, input_size=self._input_size, base_channels=self._base_channels)

  def get_num_of_params(self) -> int:
    """Returns total number of parameters."""
    total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    return total_params

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(x, size=(self._input_size, self._input_size), mode='bilinear', align_corners=True)
    gamma_scale = self._gamma_net(x)
    gamma_factor = self._gamma_min + (self._gamma_max - self._gamma_min) * gamma_scale
    return 1 / gamma_factor.view(-1, 1, 1, 1)


class RGB3DLUT(nn.Module):
  """RGB 3D Lookup table."""

  def __init__(self, lut_size: Optional[int] = RGB_LUT_SIZE):
    super().__init__()
    self._lut_size = lut_size

    # identity LUT grid in [0, 1]
    coords = torch.linspace(0, 1, lut_size)
    r, g, b = torch.meshgrid(coords, coords, coords, indexing='ij')
    identity = torch.stack([r, g, b], dim=0)
    # inverse softplus
    init = torch.log(torch.expm1(identity.clamp(min=1e-4)))
    self._lut = nn.Parameter(init.unsqueeze(0))

  def forward(self):
    return F.softplus(self._lut)

  def get_num_of_params(self):
    return self._lut.numel()


class PhotofinishingModule(nn.Module):
  """Photofinishing module."""

  def __init__(self, device: Optional[torch.device] = torch.device('cuda'), use_3d_lut: Optional[bool] = False):
    super().__init__()
    self._act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    self._lut_size = CBCR_LUT_SIZE
    self._device = device
    self._rgb_to_ycbcr_matrix = torch.tensor(RGB_TO_YCBCR, device=self._device, dtype=torch.float32)
    self._ycbcr_to_rgb_matrix = torch.tensor(YCBCR_TO_RGB, device=self._device, dtype=torch.float32)

    self._gain_net = GainNet(act_func=self._act).to(device=self._device)
    self._gtm_net = GlobalToneMappingNet(act_func=self._act).to(device=self._device)
    self._ltm_net = LocalToneMappingNet(act_func=self._act).to(device=self._device)
    self._gamma_net = GammaNet(act_func=self._act).to(device=self._device)
    self._lut_net = LuTNet(act_func=self._act, lut_size=self._lut_size).to(device=self._device)
    self._3d_lut = None
    if use_3d_lut:
      self.add_3d_lut()

  def get_cbcr_lut_size(self) -> int:
    """Returns CbCr LuT number of bins."""
    return self._lut_net.get_cbcr_lut_size()

  @staticmethod
  def _smooth_step(edge_0, edge_1, x):
    t = ((x - edge_0) / (edge_1 - edge_0))
    return t * t

  @staticmethod
  def _adjust_shadows(ycbcr: torch.Tensor, amount: float, thresh: Optional[float] = 0.3, eps: Optional[float] = 0.1
                      ) -> torch.Tensor:
    """Lift or deepen shadows.

    Args:
      ycbcr: (1,3,H,W) YCbCr tensor
      amount: positive -> lift shadows, negative -> deepen
      thresh: threshold for shadow region
      eps: smooth transition width
    """
    if amount == 0:
      return ycbcr
    amount = amount / 20
    y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
    # mask for dark regions
    shadow_mask = 1 - PhotofinishingModule._smooth_step(thresh, thresh + eps, y)
    margin = 0.1
    valid_mask = (y > margin) & (y < 1 - margin)
    shadow_mask = shadow_mask * valid_mask.float()
    # adjust shadows
    y_adj = (y + amount * (1 - y) * shadow_mask).clamp(0, 1)
    return torch.cat([y_adj, cb, cr], dim=1)

  @staticmethod
  def _adjust_highlights(ycbcr: torch.Tensor, amount, thresh: Optional[float] = 0.7, eps: Optional[float] = 0.1
                         ) -> torch.Tensor:
    """Compress or boost highlights.

    Args:
      ycbcr: (1,3,H,W) YCbCr tensor
      amount: positive -> compress highlights, negative -> boost them
      thresh: threshold for highlight region
      eps: smooth transition width
    """
    if amount == 0:
      return ycbcr
    amount = amount / 20
    y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
    # mask for bright regions
    highlight_mask = PhotofinishingModule._smooth_step(thresh, thresh + eps, y)
    margin = 0.1
    valid_mask = (y > margin) & (y < 1 - margin)
    highlight_mask = highlight_mask * valid_mask.float()
    # adjust highlights
    y_adj = (y + amount * y * highlight_mask).clamp(0, 1)

    return torch.cat([y_adj, cb, cr], dim=1)

  @staticmethod
  def _adjust_contrast(ycbcr: torch.Tensor, amount: float) -> torch.Tensor:
    """Adjusts contrast around mid-gray. amount: positive=increase contrast, negative=decrease"""
    if amount == 0:
      return ycbcr
    y = ycbcr[:, 0:1, ...]
    strength = 0.5
    y = ((y - 0.5) * (1 + strength * amount) + 0.5).clamp(0, 1)
    return torch.concatenate([y, ycbcr[:, 1:, ...]], dim=1)

  @staticmethod
  def _adjust_vibrance(img: torch.Tensor, amount: float) -> torch.Tensor:
    """Adjust vibrance: boost muted colors more than saturated ones (positive=boost vibrance, negative=reduce)."""
    if amount == 0:
        return img
    img = img.clamp(0, 1)
    if img.ndim == 3:
        img = img.unsqueeze(0)
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    maxc = img.max(dim=1).values
    minc = img.min(dim=1).values
    deltac = maxc - minc
    v = maxc
    s = deltac / (v + EPS)
    rc = (v - r) / (deltac + EPS)
    gc = (v - g) / (deltac + EPS)
    bc = (v - b) / (deltac + EPS)
    h = torch.zeros_like(v)
    mask_r = (maxc == r)
    mask_g = (maxc == g)
    mask_b = (maxc == b)
    h[mask_r] = (bc - gc)[mask_r]
    h[mask_g] = 2.0 + (rc - bc)[mask_g]
    h[mask_b] = 4.0 + (gc - rc)[mask_b]
    h = (h / 6.0) % 1.0
    s = (s * (1 + amount * (1 - s))).clamp(0, 1)
    h6 = h * 6.0
    i = torch.floor(h6).to(torch.int64) % 6
    f = h6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r_c = torch.stack([v, q, p, p, t, v], dim=1)
    g_c = torch.stack([t, v, v, q, p, p], dim=1)
    b_c = torch.stack([p, p, t, v, v, q], dim=1)
    idx = i.unsqueeze(1)
    r_out = torch.gather(r_c, 1, idx).squeeze(1)
    g_out = torch.gather(g_c, 1, idx).squeeze(1)
    b_out = torch.gather(b_c, 1, idx).squeeze(1)
    out = torch.stack([r_out, g_out, b_out], dim=1)
    return out.clamp(0, 1)

  @staticmethod
  def _adjust_saturation(img: torch.Tensor, amount: float) -> torch.Tensor:
    """Adjust saturation in HSV (positive=boost saturation, negative=reduce)."""
    if amount == 0:
        return img

    img = img.clamp(0, 1)
    if img.ndim == 3:
        img = img.unsqueeze(0)

    r, g, b = img[:, 0], img[:, 1], img[:, 2]

    maxc = img.max(dim=1).values
    minc = img.min(dim=1).values
    deltac = maxc - minc

    v = maxc
    s = deltac / (v + EPS)

    rc = (v - r) / (deltac + EPS)
    gc = (v - g) / (deltac + EPS)
    bc = (v - b) / (deltac + EPS)

    h = torch.zeros_like(v)

    mask_r = (maxc == r)
    mask_g = (maxc == g)
    mask_b = (maxc == b)

    h[mask_r] = (bc - gc)[mask_r]
    h[mask_g] = 2.0 + (rc - bc)[mask_g]
    h[mask_b] = 4.0 + (gc - rc)[mask_b]
    h = (h / 6.0) % 1.0
    s = (s * (1.0 + amount)).clamp(0, 1)
    h6 = h * 6.0
    i = torch.floor(h6).to(torch.int64) % 6
    f = h6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r_c = torch.stack([v, q, p, p, t, v], dim=1)
    g_c = torch.stack([t, v, v, q, p, p], dim=1)
    b_c = torch.stack([p, p, t, v, v, q], dim=1)
    idx = i.unsqueeze(1)
    r_out = torch.gather(r_c, 1, idx).squeeze(1)
    g_out = torch.gather(g_c, 1, idx).squeeze(1)
    b_out = torch.gather(b_c, 1, idx).squeeze(1)
    out = torch.stack([r_out, g_out, b_out], dim=1)
    return out.clamp_(0, 1)

  def _forward_training_mode(self, x_lsrgb: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Forward function of photofinishing module (training mode)."""
    output = {}
    gain = self._gain_net(x_lsrgb)
    x_lsrgb_gain = self._apply_gain(x_lsrgb, gain)
    output['y_gain'] = self.rgb_to_ycbcr(x_lsrgb_gain)[:, 0:1, ...]
    output['lsrgb_gain'] = x_lsrgb_gain
    gtm_params = self._gtm_net(x_lsrgb_gain)
    x_lsrgb_gtm = self._apply_gtm(x_lsrgb_gain, gtm_params)
    output['gtm_y'] = self.rgb_to_ycbcr(x_lsrgb_gtm)[:, 0:1, ...]
    output['lsrgb_gtm'] = x_lsrgb_gtm
    ltm_params = self._ltm_net(x_lsrgb_gain, x_lsrgb_gtm, training_mode=True)
    output.update({'ltm_params': ltm_params})
    x_lsrgb_ltm = self._apply_ltm(x_lsrgb_gain, x_lsrgb_gtm, ltm_params)
    x_ycbcr_ltm = self.rgb_to_ycbcr(x_lsrgb_ltm.clamp(0.0, 1.0))
    output['ltm_y'] = x_ycbcr_ltm[:, 0: 1, ...]
    output['lsrgb_ltm'] = x_lsrgb_ltm
    if self._3d_lut:
      rgb_lut = self._3d_lut()
      x_lsrgb_lut = self._apply_3d_lut_on_rgb(x_lsrgb_ltm, rgb_lut)
      x_ycbcr_lut = self.rgb_to_ycbcr(x_lsrgb_lut.clamp(0.0, 1.0))
      output['rgb_lut'] = rgb_lut
      output['lsrgb_3d_lut'] = x_lsrgb_lut
    else:
      x_ycbcr_lut = x_ycbcr_ltm
      output['rgb_lut'] = None
      output['lsrgb_3d_lut'] = None
    lut = self._lut_net(x_ycbcr_lut)

    output['cbcr_lut'] = lut
    processed_cbcr = self._apply_2d_lut_on_cbcr(x_ycbcr_lut[:, 1:, ...], lut)
    x_ycbcr = torch.concatenate([x_ycbcr_lut[:, 0, ...].unsqueeze(1), processed_cbcr], dim=1)
    x_lsrgb_pre_final = self.ycbcr_to_rgb(x_ycbcr)
    output['processed_cbcr'] = processed_cbcr
    output['processed_lsrgb'] = x_lsrgb_pre_final
    gamma = self._gamma_net(x_lsrgb_pre_final)
    output['gamma_factor'] = gamma
    x_srgb_out = self._apply_gamma(x_lsrgb_pre_final, gamma).clamp(min=0.0, max=1.0)
    output['output'] = x_srgb_out
    return output

  def forward(self, x_lsrgb: torch.Tensor, return_intermediate: Optional[bool] = False,
              post_process_ltm: Optional[bool] = False, report_time: Optional[bool] = False,
              training_mode: Optional[bool] = False, solver_iter: Optional[int] = BILATERAL_SOLVER_ITERS,
              contrast_amount: Optional[float] = 0.0, vibrance_amount: Optional[float] = 0.0,
              saturation_amount: Optional[float] = 0.0, highlight_amount: Optional[float] = 0.0,
              shadow_amount: Optional[float] = 0.0, input_gain_factor: Optional[torch.Tensor] = None,
              input_gtm_params: Optional[torch.Tensor] = None, input_ltm_params: Optional[torch.Tensor] = None,
              input_chroma_lut: Optional[torch.Tensor] = None, input_gamma_factor: Optional[torch.Tensor] = None,
              return_params: Optional[bool] = True, gain_blending_weight: Optional[float] = None,
              gtm_blending_weight: Optional[float] = None, ltm_blending_weight: Optional[float] = None,
              chroma_blending_weight: Optional[float] = None, gamma_blending_weight: Optional[float] = None
              ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Forward function of the photofinishing module.

    Args:
        x_lsrgb: Input linear sRGB image(s) as a torch tensor.
        return_intermediate: If True, returns intermediate stage outputs.
        post_process_ltm: If True, enables multi-scale and refinement of the LTM coefficients to mitigate halo artifacts
           (Sec. B.1, supplementary).
        report_time: If True, reports processing time for each stage.
        training_mode: If True, skips several conditional operations for efficiency during training.
        solver_iter: Number of bilateral-solver iterations used for LTM post-processing (effective only if
           `post_process_ltm` is True).
        contrast_amount: Contrast adjustment amount in [-1, 1].
        vibrance_amount: Vibrance adjustment amount in [-1, 1].
        saturation_amount: Saturation adjustment amount in [-1, 1].
        highlight_amount: Highlight adjustment amount in [-1, 1].
        shadow_amount: Shadow adjustment amount in [-1, 1].
        input_gain_factor: Pre-computed digital gain factor. If provided, the digital-gain network will not be executed.
        input_gtm_params: Pre-computed global tone mapping (GTM) coefficients. If provided, the GTM network will not be
           executed.
        input_ltm_params: Pre-computed local tone mapping (LTM) coefficients. If provided, the LTM network will not be
           executed.
        input_chroma_lut: Pre-computed chroma mapping LuT. If provided, the chroma mapping network will not be executed.
        input_gamma_factor: Pre-computed gamma correction factor. If provided, the gamma network will not be executed.
        return_params: If True (default), returns the predicted parameters of each stage in the photofinishing module.
        gain_blending_weight: Blending weight for digital gain in [0, 1]. A value of 0 disables digital gain; 1 applies
           it fully.
        gtm_blending_weight: Blending weight for GTM in [0, 1].
        ltm_blending_weight: Blending weight for LTM in [0, 1].
        chroma_blending_weight: Blending weight for chroma mapping in [0, 1].
        gamma_blending_weight: Blending weight for gamma correction in [0, 1].

    Returns:
        A dictionary containing:
          - final output image,
          - intermediate outputs (optional),
          - processing time information (optional),
          - predicted coefficients/parameters for each stage (optional).
    """

    if training_mode:
      return self._forward_training_mode(x_lsrgb)

    output = {}

    # digital gain
    if report_time:
      gain_start_time = time.perf_counter()
    else:
      gain_start_time = 0
    if input_gain_factor is None:
      gain = self._gain_net(x_lsrgb)
    else:
      gain = input_gain_factor
    if isinstance(gain, torch.Tensor):
      x_lsrgb_gain = self._apply_gain(x_lsrgb, gain)
    else:
      x_lsrgb_gain = x_lsrgb

    if gain_blending_weight is not None:
      x_lsrgb_gain = x_lsrgb_gain * gain_blending_weight + (1 - gain_blending_weight) * x_lsrgb

    if report_time:
      gain_time = time.perf_counter() - gain_start_time
      output['gain_time'] = gain_time

    if return_intermediate:
      output['lsrgb_gain'] = x_lsrgb_gain

    if return_params:
      output['pred_gain'] = gain

    # global tone mapping
    if report_time:
      gtm_start_time = time.perf_counter()
    else:
      gtm_start_time = 0
    if input_gtm_params is None:
      gtm_params = self._gtm_net(x_lsrgb_gain)
    else:
      gtm_params = input_gtm_params

    if isinstance(gtm_params, torch.Tensor):
      x_lsrgb_gtm = self._apply_gtm(x_lsrgb_gain, gtm_params)
    else:
      x_lsrgb_gtm = x_lsrgb_gain

    if gtm_blending_weight is not None:
      x_lsrgb_gtm = x_lsrgb_gtm * gtm_blending_weight + (1 - gtm_blending_weight) * x_lsrgb_gain

    if report_time:
      gtm_time = time.perf_counter() - gtm_start_time
      output['gtm_time'] = gtm_time

    if return_params:
      output['pred_gtm'] = gtm_params

    if return_intermediate:
      output['lsrgb_gtm'] = x_lsrgb_gtm

    # local tone mapping
    if report_time:
      ltm_start_time = time.perf_counter()
    else:
      ltm_start_time = 0
    if input_ltm_params is None:
      ltm_params = self._ltm_net(x_lsrgb_gain, x_lsrgb_gtm, post_process_ltm=post_process_ltm, solver_iter=solver_iter)
    else:
      ltm_params = input_ltm_params
    if report_time:
      ltm_time = time.perf_counter() - ltm_start_time
      output['ltm_time'] = ltm_time

    if isinstance(ltm_params, torch.Tensor):
      x_lsrgb_ltm = self._apply_ltm(x_lsrgb_gain, x_lsrgb_gtm, ltm_params)
    else:
      x_lsrgb_ltm = x_lsrgb_gtm

    if ltm_blending_weight is not None:
      x_lsrgb_ltm = x_lsrgb_ltm * ltm_blending_weight + (1 - ltm_blending_weight) * x_lsrgb_gtm

    if return_params:
      output['pred_ltm'] = ltm_params

    x_ycbcr_ltm = self.rgb_to_ycbcr(x_lsrgb_ltm.clamp(0.0, 1.0))

    if return_intermediate:
      output['lsrgb_ltm'] = x_lsrgb_ltm

    if contrast_amount:
      x_ycbcr_ltm = self._adjust_contrast(x_ycbcr_ltm, contrast_amount)
      if self._3d_lut:
        x_lsrgb_ltm = self.ycbcr_to_rgb(x_ycbcr_ltm)

    if highlight_amount or shadow_amount:
      x_ycbcr_ltm = self._adjust_highlights(x_ycbcr_ltm, amount=highlight_amount)
      x_ycbcr_ltm = self._adjust_shadows(x_ycbcr_ltm, amount=shadow_amount)
      if self._3d_lut:
        x_lsrgb_ltm = self.ycbcr_to_rgb(x_ycbcr_ltm)

    # global 3D RGB LuT (optional)
    if report_time:
      chroma_map_start_time = time.perf_counter()
    else:
      chroma_map_start_time = 0
    if self._3d_lut:
      rgb_lut = self._3d_lut()
      x_lsrgb_lut = self._apply_3d_lut_on_rgb(x_lsrgb_ltm, rgb_lut)
      x_ycbcr_lut = self.rgb_to_ycbcr(x_lsrgb_lut.clamp(0.0, 1.0))
    else:
      x_ycbcr_lut = x_ycbcr_ltm

    # chroma mapping
    if input_chroma_lut is None:
      lut = self._lut_net(x_ycbcr_lut)
    else:
      lut = input_chroma_lut

    if isinstance(lut, torch.Tensor):
      processed_cbcr = self._apply_2d_lut_on_cbcr(x_ycbcr_lut[:, 1:, ...], lut)
    else:
      processed_cbcr = x_ycbcr_lut[:, 1:, ...]

    x_ycbcr = torch.concatenate([x_ycbcr_lut[:, 0, ...].unsqueeze(1), processed_cbcr], dim=1)
    x_lsrgb_pre_final = self.ycbcr_to_rgb(x_ycbcr)

    if chroma_blending_weight is not None:
      x_lsrgb_pre_final = x_lsrgb_pre_final * chroma_blending_weight + (1 - chroma_blending_weight) * x_lsrgb_ltm
    if report_time:
      chroma_map_time = time.perf_counter() - chroma_map_start_time
      output['chroma_mapping_time'] = chroma_map_time

    if return_intermediate:
      output['processed_lsrgb'] = x_lsrgb_pre_final

    if return_params:
      output['pred_lut'] = lut

    # gamma correction
    if report_time:
      gamma_start_time = time.perf_counter()
    else:
      gamma_start_time = 0
    if input_gamma_factor is None:
      gamma = self._gamma_net(x_lsrgb_pre_final)
    else:
      gamma = input_gamma_factor

    if isinstance(gamma, torch.Tensor):
      x_srgb_out = self._apply_gamma(x_lsrgb_pre_final, gamma).clamp(min=0.0, max=1.0)
    else:
      x_srgb_out = x_lsrgb_pre_final

    if gamma_blending_weight is not None:
      x_srgb_out = x_srgb_out * gamma_blending_weight + (1 - gamma_blending_weight) * x_lsrgb_pre_final

    if vibrance_amount:
      x_srgb_out = self._adjust_vibrance(x_srgb_out, vibrance_amount)

    if saturation_amount:
      x_srgb_out = self._adjust_saturation(x_srgb_out, saturation_amount)

    if report_time:
      gamma_time = time.perf_counter() - gamma_start_time
      output['gamma_correction_time'] = gamma_time

    if return_intermediate or report_time or return_params:
      output['output'] = x_srgb_out
      if return_params:
        output['pred_gamma'] = gamma
    if return_intermediate or report_time or return_params:
      return output
    return x_srgb_out

  def add_3d_lut(self):
    """Adds 3D RGB LuT."""
    self._3d_lut = RGB3DLUT().to(device=self._device)

  def remove_3d_lut(self):
    """Removes 3D RGB LuT."""
    self._3d_lut = None

  def rgb_to_ycbcr(self, rgb):
    """Converts linear RGB [0,1] to Y′CbCr (BT.709)."""
    rgb_prime = rgb.clamp(0.0, 1.0)
    matrix = self._rgb_to_ycbcr_matrix
    rgb_prime = rgb_prime.permute(0, 2, 3, 1)
    ycbcr = torch.matmul(rgb_prime, matrix.T)
    return ycbcr.permute(0, 3, 1, 2)

  def ycbcr_to_rgb(self, ycbcr):
    """Converts Y′CbCr to linear RGB [0,1]."""
    matrix = self._ycbcr_to_rgb_matrix
    ycbcr = ycbcr.permute(0, 2, 3, 1)
    rgb_prime = torch.matmul(ycbcr, matrix.T)
    rgb = rgb_prime.clamp(0, 1)
    return rgb.permute(0, 3, 1, 2).clamp(0.0, 1.0)

  def print_num_of_params(self, show_message: Optional[bool] = True) -> str:
    """Prints number of parameters in the photofinishing model."""
    gain_num_params = self._gain_net.get_num_of_params()
    gtm_num_params = self._gtm_net.get_num_of_params()
    ltm_num_params = self._ltm_net.get_num_of_params()
    lut_num_params = self._lut_net.get_num_of_params()
    gamma_num_params = self._gamma_net.get_num_of_params()
    if self._3d_lut:
      rgb_lut_params = self._3d_lut.get_num_of_params()
    else:
      rgb_lut_params = 0

    total_num_params = (gain_num_params + gtm_num_params + ltm_num_params + lut_num_params + gamma_num_params +
                        rgb_lut_params)

    message = f'Total number of default network params: {total_num_params}\n'

    message = message + '---------------------------------------------------------\n'
    message = message + f'Gain network params: {gain_num_params}\n'
    message = message + f'LTM network params: {ltm_num_params}\n'
    message = message + f'GTM network params: {gtm_num_params}\n'
    if self._3d_lut:
      message = message + f'RGB LuT params: {rgb_lut_params}\n'
    message = message + f'LuT network params: {lut_num_params}\n'
    message = message + f'Gamma network params: {gamma_num_params}\n'
    if show_message:
      print(message)
    return message + '\n'

  def update_device(self, running_device: torch.device):
    """Update device."""
    if str(running_device).startswith('cuda') and not torch.cuda.is_available():
      message = 'CUDA is not available. Using CPU instead.'
      self._log_message(message)
      self._device = torch.device('cpu')
    else:
      self._device = torch.device(running_device)

    self._gain_net.to(device=self._device)
    self._gtm_net.to(device=self._device)
    self._ltm_net.to(device=self._device)
    self._lut_net.to(device=self._device)
    self._gamma_net.to(device=self._device)
    if self._3d_lut:
      self._3d_lut().to(device=self._device)
    self._rgb_to_ycbcr_matrix = self._rgb_to_ycbcr_matrix.to(device=self._device)
    self._ycbcr_to_rgb_matrix = self._ycbcr_to_rgb_matrix.to(device=self._device)

  @staticmethod
  def _apply_3d_lut_on_rgb(rgb: torch.Tensor, lut3d: torch.Tensor) -> torch.Tensor:
    """Applies a 3D LUT on linear RGB input."""
    run_on_cpu, (rgb, lut3d) = run_on_cpu_if_mps(rgb, lut3d)

    grid = rgb.permute(0, 2, 3, 1).unsqueeze(1)
    grid = grid * 2 - 1
    grid = torch.clamp(grid, -1.0, 1.0)
    lut3d = lut3d.expand(rgb.shape[0], -1, -1, -1, -1)
    out = F.grid_sample(lut3d, grid, mode='bilinear', align_corners=True)
    if run_on_cpu:
      out = out.to('mps')
    return out.squeeze(2)

  @staticmethod
  def _apply_gtm(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Applies the global tone mapping."""
    a = params[:, 0].view(-1, 1, 1, 1)
    b = params[:, 1].view(-1, 1, 1, 1)
    c = params[:, 2].view(-1, 1, 1, 1)
    x_gtm = PhotofinishingModule.apply_tm(x, a, b, c)
    return x_gtm

  @staticmethod
  def de_gamma(x_nonlinear, gamma_factor):
    """Undo gamma correction."""
    gamma_inv = 1.0 / (gamma_factor.clamp(min=1e-3))
    x_nonlinear = x_nonlinear.clamp(min=1e-5, max=1.0)
    x_lin = PhotofinishingModule._apply_gamma(x_nonlinear, gamma_inv)
    return x_lin

  @staticmethod
  def safe_pow(base: torch.Tensor, exp: Union[torch.Tensor, float], exp_clip_pos: int = 32,
               exp_clip_neg: int = 4, log_clip: Optional[float] = 40.0) -> torch.Tensor:
    """Stable power."""
    base = base.clamp_min(EPS)
    if not torch.is_tensor(exp):
      exp = torch.as_tensor(exp, dtype=base.dtype, device=base.device)
    exp = exp.clamp(min=-exp_clip_neg, max=exp_clip_pos)
    logb = torch.log(base).clamp(min=-log_clip, max=log_clip)
    z = (exp * logb).clamp(min=-log_clip, max=log_clip)
    y = torch.exp(z)
    return torch.nan_to_num(y, nan=0.0, posinf=1e20, neginf=0.0)

  @staticmethod
  def apply_tm(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Applies tone mapping."""
    run_on_cpu, (x, a, b, c) = run_on_cpu_if_mps(x, a, b, c)
    x_clamped = x.clamp(EPS, 1.0 - EPS)
    x_a = PhotofinishingModule.safe_pow(x_clamped, a)
    one_minus_x = (1.0 - x_clamped).clamp(EPS, 1.0)
    denom = x_a + PhotofinishingModule.safe_pow(c * one_minus_x, b)
    x_tm = x_a / (denom + EPS)
    if run_on_cpu:
      x_tm = x_tm.to('mps')
    return x_tm

  @staticmethod
  def _apply_ltm(x: torch.Tensor, x_gtm: torch.Tensor, ltm_coeffs: torch.Tensor) -> torch.Tensor:
    """Applies local tone mapping"""
    w = torch.sigmoid(ltm_coeffs[:, 0:1, :, :])
    a = ltm_coeffs[:, 1:2, :, :]
    b = ltm_coeffs[:, 2:3, :, :]
    c = ltm_coeffs[:, 3:4, :, :]
    g = ltm_coeffs[:, 4:5, :, :]
    x = x * g
    x_ltm = PhotofinishingModule.apply_tm(x, a, b, c)
    return (1 - w) * x_gtm + w * x_ltm

  @staticmethod
  def _apply_gain(x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """Applies digital gain."""
    return x * gain

  @staticmethod
  def _apply_gamma(x: torch.Tensor, gamma_factor: torch.Tensor) -> torch.Tensor:
    """Applies gamma correction."""
    return torch.pow(x, gamma_factor)

  @staticmethod
  def _apply_2d_lut_on_cbcr(cbcr: torch.Tensor, lut2d: torch.Tensor) -> torch.Tensor:
    """Applies 2D LuT on CbCr (chroma mapping)."""
    run_on_cpu, (cbcr, lut2d) = run_on_cpu_if_mps(cbcr, lut2d)
    b, _, h, w = cbcr.shape
    grid = torch.clamp(cbcr, -0.5, 0.5) * 2
    grid = grid.permute(0, 2, 3, 1)
    out = F.grid_sample(lut2d, grid, mode="bilinear", align_corners=True)

    if run_on_cpu:
      out = out.to('mps')

    return out




