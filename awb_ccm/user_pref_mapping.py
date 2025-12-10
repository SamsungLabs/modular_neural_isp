"""
Copyright (c) 2025 Samsung Electronics Co., Ltd.

Author(s):
Luxi Zhao (lucy.zhao@samsung.com, lucyzhao.zlx@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

Functions for mapping illuminants of one aesthetic style to another aesthetic style in CIE XYZ space.
From paper: Learning Camera-Agnostic White-Balance Preferences (https://arxiv.org/abs/2507.01342)
"""

from typing import Optional
import torch.nn as nn
import torch
import numpy as np
from utils.color_utils import xyz2cct, interpolate_cst, raw_rgb_to_cct


def poly_kernel_torch(rgb):
  """Applies a kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb)."""
  out = torch.stack([rgb[:, 0],
    rgb[:, 1], 
    rgb[:, 2], 
    rgb[:, 0] * rgb[:, 1],
    rgb[:, 0] * rgb[:, 2], 
    rgb[:, 1] * rgb[:, 2], 
    rgb[:, 0] * rgb[:, 0],
    rgb[:, 1] * rgb[:, 1], 
    rgb[:, 2] * rgb[:, 2],
    rgb[:, 0] * rgb[:, 1] * rgb[:, 2]], dim=-1)  
  return out


class UserPrefIllumEstimator(nn.Module):
  def __init__(self, in_channels: int):
    super(UserPrefIllumEstimator, self).__init__()
    output_size = 3
    layer_depth = 16
    in_channels = 10
    activation = nn.ELU()
    self._fc1 = nn.Linear(in_channels, layer_depth, bias=True)
    self._fc2 = nn.Linear(layer_depth, layer_depth // 2, bias=True)
    self._fc3 = nn.Linear(layer_depth // 2, layer_depth, bias=True)
    self._fc4 = nn.Linear(layer_depth, output_size, bias=True)
    self._activation = activation
    self._bn1 = nn.BatchNorm1d(layer_depth)

  def forward(self, x):
    """Forward function."""
    x = poly_kernel_torch(x)
    x = self._activation(self._bn1(self._fc1(x)))
    x = self._activation(self._fc2(x))
    x = self._activation(self._fc3(x))
    estimated_illum = self._fc4(x)
    return estimated_illum

  def print_num_of_params(self, show_message: Optional[bool]=True) -> str:
    """Prints number of parameters in the model."""
    total_num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    message = f'Total number of network params: {total_num_params}'
    if show_message:
      print(message)
    return message + '\n'


def xyz_to_raw_cm_on_illum(color_matrix_1, color_matrix_2, xyz_illum, 
                           calib_illum_1=21, calib_illum_2=17):
    """
    Compute CST using ColorMatrices (CM), no Chromatic Adaptation (CA).
    Apply inverse CST on one single illum only.
    """

    xyz2cam1 = np.reshape(np.asarray(color_matrix_1), (3, 3))
    xyz2cam2 = np.reshape(np.asarray(color_matrix_2), (3, 3))
    cct = xyz2cct(xyz_illum)
    xyz2cam_interp = interpolate_cst(xyz2cam1, xyz2cam2, cct, calib_illum_1, calib_illum_2)
    raw_illum = xyz2cam_interp @ xyz_illum
    return raw_illum


def raw_to_xyz_cm_on_illum(color_matrix_1, color_matrix_2, raw_illum,
                           calib_illum_1=21, calib_illum_2=17):
    """
    Compute CST using ColorMatrices (CM), no Chromatic Adaptation (CA).
    Apply CST on one single illum only.
    """
    xyz2cam1 = np.reshape(np.asarray(color_matrix_1), (3, 3))
    xyz2cam2 = np.reshape(np.asarray(color_matrix_2), (3, 3))

    cct = raw_rgb_to_cct(raw_illum, xyz2cam1, xyz2cam2, calib_illum_1, calib_illum_2)
    xyz2cam_interp = interpolate_cst(xyz2cam1, xyz2cam2, cct, calib_illum_1, calib_illum_2)

    cam2xyz = np.linalg.inv(xyz2cam_interp)
    xyz_illum = cam2xyz @ raw_illum
    return xyz_illum


def map_illum(model, raw_illum, metadata):
  """Maps illum of one aesthetic style to another aesthetic style.
  
  Args:
      model: MLP mapping model.
      raw_illum: np.ndarray of shape (3,)
      metadata: extracted by extract_additional_dng_metadata and extract_raw_metadata

  Returns:
      np.ndarray: mapped raw illuminant of shape (3,)
  """
  # normalize by G
  raw_inp = raw_illum / raw_illum[1]  
  # convert RAW to XYZ
  xyz_inp = raw_to_xyz_cm_on_illum(
              metadata['color_matrix1'], 
              metadata['color_matrix2'], 
              raw_inp, 
              calib_illum_1=metadata['calibration_illuminant1'], 
              calib_illum_2=metadata['calibration_illuminant2'])
  xyz_inp = torch.tensor(np.array(xyz_inp).astype(np.float32)).to(next(model.parameters()).device)[None, ...]

  with torch.no_grad():
    xyz_est = model(xyz_inp)
    xyz_est = xyz_est / torch.linalg.norm(xyz_est, dim=-1)  # b, 3
  xyz_est = xyz_est[0].detach().cpu().numpy()

  # convert XYZ to RAW
  raw_est = xyz_to_raw_cm_on_illum(
              metadata['color_matrix1'], 
              metadata['color_matrix2'], 
              xyz_est, 
              calib_illum_1=metadata['calibration_illuminant1'], 
              calib_illum_2=metadata['calibration_illuminant2'])
  
  # normalize by G
  raw_est = raw_est / raw_est[1]
  return raw_est