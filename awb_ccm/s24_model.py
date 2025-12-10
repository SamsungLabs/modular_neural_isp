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

Network architecture presented in the paper:
"Time-Aware Auto White Balance in Mobile Photography", ICCV 2025 (https://arxiv.org/abs/2504.05623).
"""

from typing import Optional
import torch.nn as nn
import torch
from utils.vect_utils import rg_bg_to_illum_rgb

class HistNet(nn.Module):
  def __init__(self, hist_channels, out_channels, activation):
    super(HistNet, self).__init__()
    self._conv1 = nn.Conv2d(in_channels=hist_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
    self._conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
    self._conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
    self._activation = activation
    self._global_avg_pool = nn.AdaptiveAvgPool2d(1)
    self._fc = nn.Linear(16, out_channels)

  def forward(self, x):
    x = self._activation(self._conv1(x))
    x = self._activation(self._conv2(x))
    x = self._activation(self._conv3(x))
    x = self._global_avg_pool(x)
    x = torch.flatten(x, 1)
    x = self._fc(x)
    return x

class IllumEstimator(nn.Module):
  def __init__(self, in_channels: int, hist_channels: int):
    super(IllumEstimator, self).__init__()
    if hist_channels <= 0 or in_channels <= 0:
      raise ValueError(f'Invalid input channels ({hist_channels}, {in_channels}).')
    output_size = 2
    hist_latent_dim = 16
    feature_latent_dim = 16
    layer_depth = 32
    activation = nn.ELU()
    self._hist_net = HistNet(hist_channels, hist_latent_dim, activation)
    self._feature_to_latent = nn.Linear(in_channels, feature_latent_dim, bias=True)
    self._fc1 = nn.Linear(feature_latent_dim + hist_latent_dim, layer_depth, bias=True)
    self._fc2 = nn.Linear(layer_depth, layer_depth // 2, bias=True)
    self._fc3 = nn.Linear(layer_depth // 2, layer_depth, bias=True)
    self._fc4 = nn.Linear(layer_depth, output_size, bias=True)
    self._activation = activation
    self._bn1 = nn.BatchNorm1d(layer_depth)

  def forward(self, hist, capture_data):
    """Forward function."""
    hist_latent = self._hist_net(hist)
    capture_latent = self._feature_to_latent(capture_data)
    feature = torch.cat([hist_latent, capture_latent], dim=-1)
    x = self._activation(self._bn1(self._fc1(feature)))
    x = self._activation(self._fc2(x))
    x = self._activation(self._fc3(x))
    estimated_illum_rgbg = self._fc4(x)
    return rg_bg_to_illum_rgb(estimated_illum_rgbg)

  def print_num_of_params(self, show_message: Optional[bool]=True) -> str:
    """Prints number of parameters in the model."""
    total_num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    message = f'Total number of network params: {total_num_params}'
    if show_message:
      print(message)
    return message
