"""
Author(s):
Zhongling Wang (z.wang2@samsung.com)

Description:
PyTorch implementation of the fast approximation version of the Bilateral Guided Upsampling (BGU).

This code is inspired by the original repository: https://github.com/google/bgu/
License: Apache License 2.0 (inherited from the original repository)

Original Paper:
Jiawen Chen, Andrew Adams, Neal Wadhwa, and Samuel W. Hasinoff. 2016.
Bilateral guided upsampling. ACM Trans. Graph. 35, 6, Article 203 (November 2016), 8 pages.
https://doi.org/10.1145/2980179.2982423
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _solve_safe(S_reg, T_reg):
  """Safe linalg.solve with MPS CPU fallback."""
  try:
    return torch.linalg.solve(
      S_reg.permute(0, 1, 3, 2),
      T_reg.permute(0, 1, 3, 2),
    ).permute(0, 1, 3, 2)

  except NotImplementedError as e:
    if S_reg.device.type == "mps":
      return torch.linalg.solve(
        S_reg.cpu().permute(0, 1, 3, 2),
        T_reg.cpu().permute(0, 1, 3, 2),
      ).permute(0, 1, 3, 2).to(S_reg.device)
    else:
      raise e


def bgu_fit(input_image: torch.Tensor, guide_image: torch.Tensor, output_image: torch.Tensor,
            weight_image: Optional[torch.Tensor] = None,
            grid_size: Optional[Tuple] = None,
            reg_lambda: Optional[float] = 1e-7,
            reg_T: Optional[bool] = True,
            use_float64: Optional[bool] = False) -> torch.Tensor:
  """
  Bilateral Guided Upsampling (grid fitting phase)

  Args:
      input_image: Tensor [B, C, H, W] (low-res input).
      guide_image: Tensor [B, H, W] or [B, 1, H, W] (single channel low-res guide, assumed normalized to [0, 1]).
      output_image: Tensor [B, out_ch, H, W] (low-res output).
      weight_image: Optional weight map (same shape as output_image); if None, uses ones.
      grid_size: Tuple (gh, gw, gd, out_ch, C+1). If None, defaults to [round(H/16), round(W/16), 16, out_ch, C+1].
      reg_lambda: Regularization constant for S (and T if reg_T=True). Larger values will make the model more robust to chanlenging cases but may decrease the overall performance.
      reg_T: Whether to regularize T (default True). Enable this will reduce the amount of artifacts. (The regularization for S is always applied to make sure the linear system has a solution).
      use_float64: Whether to use float64 for numerical stability (default False), will automaticaly switch to True if LinAlgError occurs.
  Returns:
      gamma: Tensor [B, gh, gw, gd, out_ch, C+1] containing the fitted low-res transform (affine models).
  """

  if use_float64:
    input_image = input_image.to(torch.float64)
    guide_image = guide_image.to(torch.float64)
    output_image = output_image.to(torch.float64)
  else:
    input_image = input_image.to(torch.float32)
    guide_image = guide_image.to(torch.float32)
    output_image = output_image.to(torch.float32)

  B, C, H, W = input_image.shape
  out_ch = output_image.size(1)
  device = input_image.device
  dtype = input_image.dtype

  if weight_image is None:
    weight_image = torch.ones_like(output_image)
  if guide_image.dim() == 4 and guide_image.size(1) == 1:
    guide_image = guide_image.squeeze(1)  # [B, H, W]

  # Default grid size: spatial bins ~ H/16, W/16 and depth 16.
  if grid_size is None:
    gh = round(H / 16)
    gw = round(W / 16)
    gd = 16
  else:
    gh, gw, gd, out_ch, _ = grid_size

  input_perm = input_image.permute(0, 2, 3, 1)  # [B, H, W, C]
  output_perm = output_image.permute(0, 2, 3, 1)  # [B, H, W, out_ch]
  ones = torch.ones((B, H, W, 1), device=device, dtype=dtype)
  input_aug = torch.cat([input_perm, ones], dim=-1)  # [B, H, W, C+1]

  # Create coordinate grid (shared across batch)
  ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
  xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
  grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]
  grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
  grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]

  # Map low-res pixel coordinates to bilateral grid space.
  gx = (grid_x + 0.5) * (gw - 1) / W  # [B, H, W]
  gy = (grid_y + 0.5) * (gh - 1) / H  # [B, H, W]
  gz = guide_image * (gd - 1)  # [B, H, W]

  # Compute floor and ceil indices for each coordinate.
  x0 = torch.floor(gx).long().clamp(0, gw - 1)
  y0 = torch.floor(gy).long().clamp(0, gh - 1)
  z0 = torch.floor(gz).long().clamp(0, gd - 1)
  x1 = (x0 + 1).clamp(max=gw - 1)
  y1 = (y0 + 1).clamp(max=gh - 1)
  z1 = (z0 + 1).clamp(max=gd - 1)

  # Compute fractional parts.
  wx = gx - x0.float()  # [B, H, W]
  wy = gy - y0.float()  # [B, H, W]
  wz = gz - z0.float()  # [B, H, W]

  # Flatten spatial dimensions: N = H*W.
  N = H * W
  x0_flat = x0.view(B, N)
  x1_flat = x1.view(B, N)
  y0_flat = y0.view(B, N)
  y1_flat = y1.view(B, N)
  z0_flat = z0.view(B, N)
  z1_flat = z1.view(B, N)
  wx_flat = wx.view(B, N)
  wy_flat = wy.view(B, N)
  wz_flat = wz.view(B, N)

  # Stack the eight corner indices along a new dimension.
  # Each is [B, N, 1] and then stack to get shape [B, N, 8].
  x_stack = torch.stack([x0_flat, x0_flat, x0_flat, x0_flat, x1_flat, x1_flat, x1_flat, x1_flat], dim=-1)
  y_stack = torch.stack([y0_flat, y0_flat, y1_flat, y1_flat, y0_flat, y0_flat, y1_flat, y1_flat], dim=-1)
  z_stack = torch.stack([z0_flat, z1_flat, z0_flat, z1_flat, z0_flat, z1_flat, z0_flat, z1_flat], dim=-1)

  # Similarly, stack the eight trilinear weights, shape [B, N, 8]
  w_stack = torch.stack([(1 - wx_flat) * (1 - wy_flat) * (1 - wz_flat),
                         (1 - wx_flat) * (1 - wy_flat) * wz_flat,
                         (1 - wx_flat) * wy_flat * (1 - wz_flat),
                         (1 - wx_flat) * wy_flat * wz_flat,
                         wx_flat * (1 - wy_flat) * (1 - wz_flat),
                         wx_flat * (1 - wy_flat) * wz_flat,
                         wx_flat * wy_flat * (1 - wz_flat),
                         wx_flat * wy_flat * wz_flat],
                        dim=-1)  # w000  # w001  # w010  # w011  # w100  # w101  # w110  # w111

  # Compute linear indices for each corner.
  # In the bilateral grid, linear index = y * (gw*gd) + x * gd + z.
  linear_idx = y_stack * (gw * gd) + x_stack * gd + z_stack  # [B, N, 8]

  # Compute outer products for each pixel.
  input_aug_flat = input_aug.reshape(B, N, C + 1)  # [B, N, C+1]
  output_flat = output_perm.reshape(B, N, out_ch)  # [B, N, out_ch]
  weight_flat = weight_image.reshape(B, N, out_ch).mean(dim=2)  # [B, N]

  # Compute outer products per pixel.
  outer_alpha = torch.einsum("bni,bnj->bnij", input_aug_flat, input_aug_flat)  # [B, N, C+1, C+1]
  outer_beta = torch.einsum("bnk,bnj->bnkj", output_flat, input_aug_flat)  # [B, N, out_ch, C+1]

  # Multiply each pixel’s contributions by its trilinear weights
  contrib = w_stack * weight_flat.unsqueeze(-1)  # [B, N, 8]
  contrib_alpha = contrib.unsqueeze(-1).unsqueeze(-1)  # [B, N, 8, 1, 1]
  contrib_beta = contrib.unsqueeze(-1).unsqueeze(-1)  # [B, N, 8, 1, 1]

  # Expand outer_alpha and outer_beta along a new dimension for corners.
  outer_alpha = outer_alpha.unsqueeze(2).expand(-1, -1, 8, -1, -1)  # [B, N, 8, C+1, C+1]
  outer_beta = outer_beta.unsqueeze(2).expand(-1, -1, 8, -1, -1)  # [B, N, 8, out_ch, C+1]

  # Multiply by contributions.
  outer_alpha = outer_alpha * contrib_alpha  # [B, N, 8, C+1, C+1]
  outer_beta = outer_beta * contrib_beta  # [B, N, 8, out_ch, C+1]

  # Flatten the corner dimension: total contributions per image: shape [B, N*8, ...]
  outer_alpha = outer_alpha.reshape(B, N * 8, C + 1, C + 1)  # [B, N*8, C+1, C+1]
  outer_beta = outer_beta.reshape(B, N * 8, out_ch, C + 1)  # [B, N*8, out_ch, C+1]
  linear_idx = linear_idx.reshape(B, N * 8)  # [B, N*8]

  # scatter-add the contributions into accumulators S and T.
  # Let S have shape [B, gh*gw*gd, C+1, C+1] and T have shape [B, gh*gw*gd, out_ch, C+1].
  num_grid_cells = gh * gw * gd
  S_accum = torch.zeros(B, num_grid_cells, C + 1, C + 1, device=device, dtype=dtype)
  T_accum = torch.zeros(B, num_grid_cells, out_ch, C + 1, device=device, dtype=dtype)

  # For vectorized scatter, we flatten the batch dimension.
  # Compute an offset for each batch element.
  offsets = (torch.arange(B, device=device) * num_grid_cells).view(B, 1)  # [B,1]
  linear_idx_offset = linear_idx + offsets  # [B, N*8]

  # Flatten batch: shape [B * num_contribs, ...]
  linear_idx_flat = linear_idx_offset.reshape(-1)  # [B*N*8]
  outer_alpha_flat = outer_alpha.reshape(B * N * 8, C + 1, C + 1)
  outer_beta_flat = outer_beta.reshape(B * N * 8, out_ch, C + 1)

  # Flatten S_accum and T_accum similarly.
  S_flat = S_accum.reshape(B * num_grid_cells, C + 1, C + 1)
  T_flat = T_accum.reshape(B * num_grid_cells, out_ch, C + 1)

  # Use scatter_add to accumulate contributions. We need to expand linear_idx_flat
  # to match the trailing dimensions.
  # For S: index shape becomes [B*N*8, C+1, C+1]
  idx_S = linear_idx_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, C + 1, C + 1)
  S_flat = S_flat.scatter_add(0, idx_S, outer_alpha_flat)

  # For T: index shape becomes [B*N*8, out_ch, C+1]
  idx_T = linear_idx_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, out_ch, C + 1)
  T_flat = T_flat.scatter_add(0, idx_T, outer_beta_flat)

  # Reshape S_accum and T_accum back.
  S_accum = S_flat.reshape(B, num_grid_cells, C + 1, C + 1)
  T_accum = T_flat.reshape(B, num_grid_cells, out_ch, C + 1)

  # --- Regularization ---
  # regularize S: make S invertible
  I_eye = torch.eye(C + 1, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # [1,1,C+1,C+1]
  counts = S_accum[..., -1, -1]
  weighted_lambda = reg_lambda * (counts + 1)  # B, num_grid_cells
  S_reg = S_accum + weighted_lambda[..., None, None] * I_eye  # [B, num_grid_cells, C+1, C+1]

  # regularize T: fill some of the zero entries in T with default grid (DefaultGamma)
  if reg_T:
    # global default transform
    S_sum_global = S_accum.sum(dim=1)  # Sum across all grid cells [B, C+1, C+1]
    T_sum_global = T_accum.sum(dim=1)  # Sum across all grid cells [B, out_ch, C+1]

    global_counts_sum = counts.sum(dim=1)  # Sum of counts across all cells
    weighted_lambda_global = reg_lambda * (global_counts_sum + 1.0)
    gain_global = T_sum_global[:, :, 3] / (S_sum_global[:, :-1, 3] + weighted_lambda_global)
    gain_global = gain_global.unsqueeze(1).repeat(1, num_grid_cells, 1)  # [B, M, 3]

    # local default transform
    gain_local = T_accum[..., :, 3] / (S_accum[..., :-1, 3] + weighted_lambda.unsqueeze(-1))

    # mixed local–global transform
    zero_count_mask = (counts == 0)  # [B, M]
    mixed_gain = torch.where(zero_count_mask.unsqueeze(-1), gain_global, gain_local)  # [B, M, 3]

    # diagonal matrix: processing each channel separately
    DefaultGamma = torch.zeros_like(T_accum)
    DefaultGamma[..., :3] = torch.diag_embed(mixed_gain)  # B, M, 3, 4

    T_reg = T_accum + weighted_lambda[..., None, None] * DefaultGamma
  else:
    T_reg = T_accum.clone()

  # solve γ: T = γS
  # Error handling: Automatically switch to float64 if necessary. This should rarely happen.
  try:
    # gamma_flat = torch.linalg.solve(S_reg.permute(0, 1, 3, 2), T_reg.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    gamma_flat = _solve_safe(S_reg, T_reg)
    gamma = gamma_flat.reshape(B, gh, gw, gd, out_ch, C + 1)
  except torch.linalg.LinAlgError as e:
    print(f"Singular matrix encountered: {e}, retry using float64")
    gamma = bgu_fit(input_image, guide_image, output_image, weight_image, grid_size, reg_lambda, reg_T,
                    use_float64=True)

  return gamma.to(torch.float32)


def bgu_slice(gamma: torch.Tensor, input_image: torch.Tensor, guide_image: torch.Tensor) -> torch.Tensor:
  """
  Bilateral Guided Upsampling (slicing and apply phase)

  Args:
      gamma: Tensor of shape [B, gh, gw, gd, out_ch, in_ch] containing the low-res affine models.
      input_image: Full-res input image of shape [B, C, H, W].
      guide_image: Full-res guide image of shape [B, H, W] or [B, 1, H, W], normalized to [0, 1].

  Returns:
      output: Full-res output image of shape [B, out_ch, H, W].

  Steps:
      1. Permute and reshape gamma to shape [B, out_ch*in_ch, gd, gh, gw] so that it can be
      interpreted as a 3D volume.
      2. Build a grid of sampling coordinates (one per full-res pixel) with shape [B, 1, H, W, 3].
      The coordinates are normalized with respect to the bilateral grid dimensions (gw, gh, gd)
      using align_corners=True.
      3. Use F.grid_sample to upsample from low-res gamma, output high-res gamma size: [B, H, W, out_ch, in_ch].
      4. Apply the high-res gamma (pixel-wise affine transform).
  """
  B, C, H, W = input_image.shape
  device = input_image.device
  dtype = gamma.dtype

  input_image = input_image.to(dtype)
  guide_image = guide_image.to(dtype)

  # If guide_image has only one channel, squeeze it.
  if guide_image.dim() == 4 and guide_image.size(1) == 1:
    guide_image = guide_image.squeeze(1)  # [B, H, W]

  _, gh, gw, gd, out_ch, in_ch = gamma.shape  # in_ch = input_channels + 1

  # Reshape gamma for grid_sample
  gamma_perm = gamma.permute(0, 4, 5, 3, 1, 2)  # [B, out_ch, in_ch, gd, gh, gw]
  gamma_reshaped = gamma_perm.reshape(B, out_ch * in_ch, gd, gh, gw)  # [B, out_ch*in_ch, gd, gh, gw]

  # --- Build sampling grid ---
  # We want to sample at one location per full-res pixel, producing an output volume of shape [B, 1, H, W, 3]
  # The grid_sample function expects grid coordinates in order (x, y, z) normalized to [-1,1]
  # We want to map each pixel center (x+0.5) to a coordinate in the gamma grid (align_corners=True).
  xs = torch.arange(W, device=device, dtype=dtype)
  ys = torch.arange(H, device=device, dtype=dtype)
  grid_x = (xs + 0.5) * (2.0 / W) - 1.0  # [W]
  grid_y = (ys + 0.5) * (2.0 / H) - 1.0  # [H]
  grid_z = guide_image * 2.0 - 1.0  # [B, H, W]

  grid_y2, grid_x2 = torch.meshgrid(grid_y, grid_x, indexing="ij")  # both [H, W]
  grid_y2 = grid_y2.unsqueeze(0).repeat(B, 1, 1)  # [B, H, W]
  grid_x2 = grid_x2.unsqueeze(0).repeat(B, 1, 1)  # [B, H, W]

  # The order is (x, y, z).
  grid = torch.stack([grid_x2, grid_y2, grid_z], dim=-1)  # [B, H, W, 3]
  # Expand to batch and add a dummy depth dimension (D_out = 1)
  grid = grid.unsqueeze(1).expand(B, 1, H, W, 3)  # [B, 1, H, W, 3]

  # --- Sample from gamma using grid_sample ---
  # grid_sample input: [B, C, D, H, W], grid: [B, D_out, H_out, W_out, 3]. 
  try:
    sampled = F.grid_sample(
      gamma_reshaped,
      grid,
      mode="bilinear",
      align_corners=True
    )  # [B, out_ch*in_ch, 1, H, W]
  except NotImplementedError:
    if gamma_reshaped.device.type == "mps":
      sampled = F.grid_sample(
        gamma_reshaped.cpu(),
        grid.cpu(),
        mode="bilinear",
        align_corners=True
      ).to(gamma_reshaped.device)
    else:
      raise
  sampled = sampled.squeeze(2)  # [B, out_ch*in_ch, H, W]
  sampled = sampled.reshape(B, out_ch, in_ch, H, W)  # [B, out_ch, in_ch, H, W]
  M = sampled.permute(0, 3, 4, 1, 2)  # [B, H, W, out_ch, in_ch]

  # --- Apply affine transform ---
  # Augment input_image: permute to [B, H, W, C] then add a ones channel.
  input_permuted = input_image.permute(0, 2, 3, 1)  # [B, H, W, C]
  ones = torch.ones((B, H, W, 1), device=device, dtype=dtype)
  input_aug = torch.cat([input_permuted, ones], dim=-1)  # [B, H, W, in_ch]

  # For each pixel, apply M: output[b,h,w] = M[b,h,w] @ input_aug[b,h,w].
  output = torch.einsum("bhwij,bhwj->bhwi", M, input_aug)  # [B, H, W, out_ch]
  output = output.permute(0, 3, 1, 2)  # [B, out_ch, H, W]
  return output


def rgb_to_luma(rgb: torch.Tensor) -> torch.Tensor:
  """
  Convert RGB a image to a luminance (grayscale) image.

  Args:
      rgb: [B, 3, H, W] (RGB image)

  Returns:
      luma: [B, 1, H, W] (luminance image)
  """
  return 0.2989 * rgb[:, 0:1] + 0.5870 * rgb[:, 1:2] + 0.1140 * rgb[:, 2:3]


class BGU(nn.Module):
  """
  Bilateral Guided Upsampling (BGU) Fast Approximation
  """

  def __init__(self, grid_size: Optional[Tuple] = None, reg_lambda: Optional[float] = 1e-7,
               reg_T: Optional[bool] = True):
    """
    Args:
        grid_size: Tuple (gh, gw, gd, out_ch, C+1). If None, defaults to [round(H/16), round(W/16), 16, out_ch, C+1].
        reg_lambda: Regularization constant for both S and T.
        reg_T: Whether to regularize T (default True). The regularization for S is always applied to make sure the linear system has a solution.
    """
    super().__init__()
    self.grid_size = grid_size
    self.reg_lambda = reg_lambda
    self.reg_T = reg_T

  def forward(self, high_res_input: torch.Tensor, low_res_input: torch.Tensor,
              low_res_target: torch.Tensor) -> torch.Tensor:
    """
    Two steps:
        1. bgu_fit: Fit the low-res transformation (grid)
        2. bgu_slice: Upsample the low-res transformation and apply it to the high-res input

    Args:
        high_res_input: Tensor [B, C, H, W] (full-res input image).
        low_res_input: Tensor [B, C, h, w] (low-res input image).
        low_res_target: Tensor [B, out_ch, h, w] (low-res target image).

    Returns:
        output: Tensor [B, out_ch, H, W] (upsampled full-res output image).
    """

    high_res_guide = rgb_to_luma(high_res_input)
    low_res_guide = rgb_to_luma(low_res_input)

    grid = bgu_fit(low_res_input, low_res_guide, low_res_target, grid_size=self.grid_size, reg_lambda=self.reg_lambda,
                   reg_T=self.reg_T)  # B, gh, gw, gd, 3, 4
    output = bgu_slice(grid, high_res_input, high_res_guide).clamp(0, 1)
    return output
