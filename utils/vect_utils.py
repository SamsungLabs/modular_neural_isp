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

This file contains vector utility functions.
"""

from typing import Union, Optional
import torch
import numpy as np
from utils.constants import *


def min_max_normalization(data: np.ndarray, min_vector: np.ndarray, max_vector: np.ndarray):
  """Applies min-max normalization."""
  return ((data - min_vector) / (max_vector - min_vector + EPS)).astype(np.float32)

def rg_bg_to_illum_rgb(rgbg: Union[np.ndarray, torch.tensor],
                       normalize:Optional[bool]=True
                       ) -> Union[np.ndarray, torch.tensor]:
  """Converts an R/G B/G vector to normalized illuminant color."""

  def normalize_illum(illum: np.ndarray) -> np.ndarray:
    """Normalizes illuminant color."""
    return illum / vect_norm(illum)

  one_tensor = torch.tensor(1.0, dtype=rgbg.dtype, device=rgbg.device
                            ).expand(rgbg.size(0))
  illum = torch.stack([rgbg[:, 0], one_tensor, rgbg[:, 1]], dim=1)
  if normalize:
    return normalize_illum(illum)
  else:
    return illum

def angular_loss(predicted, gt, shrink=True):
  """Computes angular error between predicted and gt illuminant color(s)"""

  is_np = False
  if isinstance(predicted, list) or isinstance(predicted, np.ndarray):
    predicted = torch.tensor(predicted)
    is_np = True
  if isinstance(gt, list) or isinstance(gt, np.ndarray):
    gt = torch.tensor(gt)
    is_np = True

  if len(gt.shape) == 1:
    gt = gt.unsqueeze(0)

  if len(predicted.shape) == 1:
    predicted = predicted.unsqueeze(0)

  cossim = torch.clamp(torch.sum(predicted * gt, dim=1) / (
      torch.norm(predicted, dim=1) * torch.norm(gt, dim=1) + EPS), -1, max=1.0)
  if shrink:
    angle = torch.acos(cossim * SHRINK_FACTOR)
  else:
    angle = torch.acos(cossim)
  angular_error = 180.0 / PI * angle
  angular_error = torch.sum(angular_error) / angular_error.shape[0]
  return angular_error if not is_np else angular_error.numpy()

def vect_norm(vec: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
  """Computes vector norm."""
  is_tensor = torch.is_tensor(vec)
  return torch.linalg.vector_norm(vec, dim=1, keepdim=True) if is_tensor else np.linalg.norm(vec)
