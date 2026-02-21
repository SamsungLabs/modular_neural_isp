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

Net architecture of the Raw-JPEG Adapter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import *
from utils.img_utils import tensor_to_img, imwrite, img_to_tensor
import zlib
import base64
import os
from typing import Optional, Tuple, Union, List, Dict
import math

import numpy as np

def safe_pow(base, exponent):
  """Applies torch.pow in a safe way, avoiding NaNs and overflows."""
  base = torch.clamp(base, min=EPS)
  if isinstance(exponent, torch.Tensor):
    exponent = torch.clamp(exponent, min=EPS, max=10.0)
  return torch.pow(base, exponent)

class ECABlock(nn.Module):
  """Efficient channel attention (ECA) block."""
  def __init__(self, k_size: int):
    super().__init__()
    self._avg_pool = nn.AdaptiveAvgPool2d(1)
    pad = (k_size - 1) // 2
    self._conv = nn.Conv1d(1, 1, kernel_size=k_size, bias=False, padding=pad)

  def forward(self, x):
    y = self._avg_pool(x)
    y = self._conv(y.squeeze(-1).transpose(-1, -2))
    y = torch.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
    return x * y.expand_as(x)

class LuTModule(nn.Module):
  def __init__(self, num_bins: Optional[int] = 64, num_channels: Optional[int] = 3):
    super().__init__()
    self._k = num_bins
    self._c = num_channels

  @staticmethod
  def normalize(un_normalized_y: torch.Tensor) -> torch.Tensor:
    """Converts unnormalized LuT heights to monotonic [0,1] mapping."""
    h = F.softplus(un_normalized_y)
    y = torch.cumsum(h, dim=2)
    y0 = y[:, :, :1]
    yn = y[:, :, -1:]
    return (y - y0) / (yn - y0)

  def forward(self, x: torch.Tensor, un_normalized_y: torch.Tensor, inverse: Optional[bool] = False) -> torch.Tensor:
    b, c, h, w = x.shape
    if un_normalized_y.shape[1] == 1 and c > 1:
      un_normalized_y = un_normalized_y.expand(b, c, un_normalized_y.shape[-1])
    assert un_normalized_y.shape[1] == c, f"LUT channels ({un_normalized_y.shape[1]}) != input channels ({c})"


    k = self._k
    n = h * w
    y_bc = self.normalize(un_normalized_y)
    flat = x.reshape(b, c, n)

    if not inverse:
      idx0 = (flat * k).floor().clamp(0, k - 1).long()
      idx1 = idx0 + 1
      y0 = torch.gather(y_bc, 2, idx0)
      y1 = torch.gather(y_bc, 2, idx1)
      alpha = flat * k - idx0.to(flat.dtype)
      out = y0 + alpha * (y1 - y0)
    else:
      ys = y_bc.contiguous()
      vals = flat.contiguous()
      idx = torch.searchsorted(ys, vals, right=False)
      idx = idx.clamp(1, k)
      idx0 = idx - 1
      idx1 = idx
      y0 = torch.gather(ys, 2, idx0)
      y1 = torch.gather(ys, 2, idx1)
      t0 = idx0.to(vals.dtype) / k
      t1 = idx1.to(vals.dtype) / k
      alpha = (vals - y0) / (y1 - y0).clamp(min=1e-6)
      out = t0 + alpha * (t1 - t0)

    return out.view(b, c, h, w)


class Encoder(nn.Module):
  """Encoder network."""
  def __init__(self, latent_channels: Optional[int] = 24, use_eca: Optional[bool] = True,
               return_intermediates: bool = False):
    super().__init__()
    self._use_eca = use_eca
    self._return_intermediates = return_intermediates

    self._conv1 = nn.Sequential(
      nn.ReflectionPad2d(1),
      nn.Conv2d(3, latent_channels // 2, kernel_size=3, stride=2, padding=0),
      nn.GELU(),
      ECABlock(5) if use_eca else nn.Identity()
    )
    self._conv2 = nn.Sequential(
      nn.ReflectionPad2d(1),
      nn.Conv2d(latent_channels // 2, latent_channels, 3, stride=2, padding=0),
      nn.GELU(),
      ECABlock(5) if use_eca else nn.Identity()
    )
    self._conv3 = nn.Sequential(
      nn.Conv2d(latent_channels, latent_channels, 1, stride=1, padding=0),
      nn.GELU()
    )

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Union[List[torch.Tensor], None]]:
    if self._return_intermediates:
      feats = []
    else:
      feats = None
    x = self._conv1(x)
    if self._return_intermediates:
      feats.append(x)
    x = self._conv2(x)
    if self._return_intermediates:
      feats.append(x)
    x = self._conv3(x)
    return x, feats


class Decoder(nn.Module):
  """Decoder network."""
  def __init__(self, gamma: bool, scale_dct: bool, lut: bool, latent_channels: Optional[int] = 24,
               use_eca: Optional[bool] = False, use_skip: bool = False,
               lut_size: Optional[int] = JPEG_ADAPT_LUT_BINS,
               lut_channels: Optional[int] = JPEG_ADAPT_LUT_CHS,
               dct_block: Optional[callable]=None):
    super().__init__()
    self._use_eca = use_eca
    self._use_skip = use_skip
    self._scale_dct = scale_dct
    self._gamma = gamma
    self._lut = lut
    self._dct_block = dct_block

    self._conv = nn.Sequential(
      nn.Conv2d(latent_channels, latent_channels // 2, 1, padding=0),
      nn.GELU(),
      ECABlock(5) if self._use_eca else nn.Identity()
    )
    self._up1 = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
      nn.ReflectionPad2d(1),
      nn.Conv2d(latent_channels // 2 + latent_channels if self._use_skip else latent_channels // 2,
                latent_channels // 2, 3, padding=0),
      nn.GELU(),
      ECABlock(5) if self._use_eca else nn.Identity()
    )

    self._up2 = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
      nn.ReflectionPad2d(1),
      nn.Conv2d(latent_channels if self._use_skip else latent_channels // 2, latent_channels // 2, 3,
                padding=0),
      nn.GELU(),
    )

    if self._gamma:
      self._gamma_output = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(latent_channels // 2, 1, 3, padding=0)
      )

    if self._scale_dct:
      self._freq_decoder = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(latent_channels, latent_channels // 2, 3, padding=0),
        nn.BatchNorm2d(latent_channels // 2),
        nn.GELU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(latent_channels // 2, latent_channels, 3, padding=0),
        nn.GELU(),
        nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False)
      )

      self._scale_dct_output = nn.Conv2d(latent_channels, 3, 1)
    else:
      self._scale_dct_output = self._freq_decoder = None

    if self._lut:
      self._lut_channels = lut_channels
      self._lut_bins = lut_size
      self._lut_net = nn.Sequential(
        nn.Conv2d(latent_channels, latent_channels // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(latent_channels // 2),
        nn.GELU(),
        nn.Upsample(size=(2, 2), mode='bilinear', align_corners=False)
      )
      self._lut_head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(4 * latent_channels // 2, self._lut_channels * (self._lut_bins + 1))
      )

  def forward(self, z: torch.Tensor, skips: Optional[List[torch.Tensor]] = None
              ) -> Dict[str, Union[torch.Tensor, None]]:
    x = self._conv(z)

    if self._use_skip and skips is not None:
      x = torch.cat([x, skips[-1]], dim=1)
    x = self._up1(x)

    if self._use_skip and skips is not None:
      x = torch.cat([x, skips[-2]], dim=1)
    x = self._up2(x)

    if self._gamma:
      gamma_map = torch.exp(2.0 * torch.tanh(self._gamma_output(x))) # [0.14, 7.4]
    else:
      gamma_map = None

    if self._scale_dct:
      z_freq = self._freq_decoder(z)
      scale_dct = self._dct_block(torch.exp(0.7 * torch.tanh(self._scale_dct_output(z_freq))))  # [0.4966, 2.0138]
    else:
      scale_dct = None

    if self._lut:
      lut = self._lut_head(self._lut_net(z)).view(-1, self._lut_channels, self._lut_bins + 1)
    else:
      lut = None

    return {'gamma_map': gamma_map, 'scale_dct': scale_dct, 'lut': lut}


class JPEGAdapter(nn.Module):
  """JPEG Adapter network."""
  def __init__(self, latent_dim: int=24, target_img_size: Tuple[int, int]=(64, 64), quality: int=75,
               use_gamma: Optional[bool]=False, use_scale_dct: Optional[bool]=False,
               use_lut: Optional[bool]=False, use_eca: Optional[bool]=False,
               lut_size: Optional[int] = JPEG_ADAPT_LUT_BINS, lut_channels: Optional[int] = JPEG_ADAPT_LUT_CHS):
    super().__init__()
    assert target_img_size[0] >= 64 and target_img_size[1] >= 64
    self._target_map_sz = target_img_size
    self._torch_dtype = torch.float32
    self._quality = quality
    self._latent_dim = latent_dim
    self._eca = use_eca
    self._gamma = use_gamma
    self._scale_dct = use_scale_dct
    self._lut = use_lut

    if not (self._gamma or self._scale_dct or self._lut):
      raise ValueError('At least one of "use_gamma", "use_scale_dct", or/and "use_lut" must be True.')
    self._encoder = Encoder(latent_channels=self._latent_dim, use_eca=self._eca, return_intermediates=True
                            ).to(self._torch_dtype)

    self._decoder = Decoder(latent_channels=latent_dim, use_eca=use_eca, gamma=self._gamma,
                            scale_dct=self._scale_dct, lut=self._lut, use_skip=True,
                            lut_size=lut_size, lut_channels=lut_channels, dct_block=self._dct2_blockwise
                            ).to(self._torch_dtype)
    if self._lut:
      self._lut_module = LuTModule(num_bins=lut_size, num_channels=lut_channels)
    else:
      self._lut_module = None

  def get_config(self):
    """Returns network configuration."""
    return {'target_map_size': self._target_map_sz, 'jpeg_quality': self._quality, 'latent_dim': self._latent_dim,
            'eca': self._eca, 'gamma': self._gamma,
            'scale_dct': self._scale_dct, 'lut': self._lut, 'dtype': self._dtype}

  def forward(self, raw_img: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, None]]]:
    """Forward function that emits the coeffs of pre-processing operators, applies them to the input raw image."""
    raw_img_resized = F.interpolate(raw_img, size=self._target_map_sz, mode='bilinear', align_corners=False)
    latent, intermediate = self._encoder(raw_img_resized)
    output_params = self._decoder(latent, intermediate)
    raw_img_proc = self._pre_process_jpeg(raw_img=raw_img, gamma_map=output_params['gamma_map'],
                                          scale_dct=output_params['scale_dct'],
                                          lut=output_params['lut'])
    return raw_img_proc, output_params

  def preprocess_raw(self, raw: torch.Tensor, gamma_map: Optional[torch.Tensor] = None,
                     scale_dct: Optional[torch.Tensor] = None, lut: Optional[torch.Tensor] = None,
                     dtype: Optional[torch.dtype]=torch.float32,
                     ) -> torch.Tensor:
    """Pre-process input raw images."""

    if gamma_map is not None:
      gamma_map = gamma_map.to(dtype=dtype)
    if lut is not None:
      lut = lut.to(dtype=dtype)
    raw_img_proc = self._pre_process_jpeg(raw_img=raw, gamma_map=gamma_map, scale_dct=scale_dct, lut=lut)
    return raw_img_proc

  def reconstruct_raw(self, raw_jpeg: torch.Tensor, gamma_map: Optional[torch.Tensor] = None,
                      scale_dct: Optional[torch.Tensor] = None, lut: Optional[torch.Tensor]=None) -> torch.Tensor:
    """Reconstructs the original raw image from the processed JPEG file using given maps."""
    if gamma_map is None and scale_dct is None and lut is None:
      required = []
      if self._gamma:
        required.append("'gamma_map'")
      if self._scale_dct:
        required.append("'scale_dct'")
      if self._lut:
        required.append("'lut'")
      if required:
        msg = ", ".join(required[:-1])
        if len(required) > 1:
          msg += f" and {required[-1]}"
        else:
          msg = required[0]
        raise ValueError(f'{msg} must be provided.')

    raw_img = self._post_process_jpeg(raw_jpeg=raw_jpeg, gamma_map=gamma_map, scale_dct=scale_dct, lut=lut)
    return raw_img

  def _pre_process_jpeg(self, raw_img: torch.Tensor, gamma_map: torch.Tensor, scale_dct: Optional[torch.Tensor]=None,
                        lut: Optional[torch.Tensor]=None) -> torch.Tensor:
    """Applies pre-processing to raw before saving to JPEG file."""

    # 1) LuT
    if lut is not None:
      raw_img = self._lut_module(raw_img, lut)

    # 2) DCT
    if scale_dct is not None:
      _, _, h, w = raw_img.shape
      padded_raw_img = self._pad_to_block(raw_img)
      raw_img_dct = self._dct2_blockwise(padded_raw_img)
      raw_img_dct = raw_img_dct * scale_dct
      raw_img = self._idct2_blockwise(raw_img_dct, image_size=padded_raw_img.shape[2:])[:, :, :h, :w]

    # 3) Spatial operator
    if gamma_map is not None:
      gamma_map_resized = F.interpolate(gamma_map, size=raw_img.shape[2:4], mode='bilinear', align_corners=False)
    else:
      gamma_map_resized = 1.0

    raw_img = safe_pow(raw_img, gamma_map_resized)

    return raw_img

  def _post_process_jpeg(self, raw_jpeg: torch.Tensor, gamma_map: torch.Tensor,
                         scale_dct: Optional[torch.Tensor]=None, lut: Optional[torch.Tensor] = None
                         ) -> torch.Tensor:
    """Applies the inverse of the pre-processing operators to the raw JPEG to reconstruct the original raw image."""

    # 1) Spatial operators
    if gamma_map is not None:
      gamma_map_resized = F.interpolate(gamma_map, size=raw_jpeg.shape[2:4], mode='bilinear', align_corners=False)
    else:
      gamma_map_resized = 1.0

    rec_raw_img = (safe_pow(raw_jpeg, 1.0 / gamma_map_resized)).clamp(min=0.0, max=1.0)

    # 2) DCT
    if scale_dct is not None:
      _, _, h, w = rec_raw_img.shape
      padded_rec_raw_img = self._pad_to_block(rec_raw_img)
      raw_img_dct = self._dct2_blockwise(padded_rec_raw_img)
      raw_img_dct /= scale_dct
      rec_raw_img = self._idct2_blockwise(raw_img_dct, image_size=padded_rec_raw_img.shape[2:])[:, :, :h, :w]

    # 3) LuT
    if lut is not None:
      rec_raw_img = self._lut_module(rec_raw_img, lut, inverse=True)

    return rec_raw_img.clamp(min=0.0, max=1.0)

  def print_num_of_params(self, show_message: Optional[bool]=True) -> str:
    """Prints number of parameters in the model."""
    total_num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    message = f'Total number of network params: {total_num_params}'
    if show_message:
      print(message)
    return message + '\n'

  @staticmethod
  def normalize_lut(lut: torch.Tensor) -> torch.Tensor:
    """Normalizes LuT."""
    return LuTModule.normalize(lut)

  @staticmethod
  def encode_params(params: Dict[str, Union[torch.Tensor, None]]) -> str:
    """Encodes predicted parameters to a string for JPEG comment embedding."""
    if params['gamma_map'] is not None:
      gamma_map = tensor_to_img(params['gamma_map'].squeeze(0)).astype(np.float32)
      s_shape = np.array(gamma_map.shape, dtype=np.uint16).tobytes()
      s_shape_str = base64.b64encode(s_shape).decode('utf-8')
      gamma_map_data = gamma_map.flatten().tobytes()
      gamma_map_encoded = base64.b64encode(zlib.compress(gamma_map_data, level=9)).decode('utf-8')
    else:
      gamma_map_encoded = '$'
      s_shape_str = '$'
    if params['scale_dct'] is not None:
      scale_dct = params['scale_dct'].detach().cpu().numpy().astype(np.float32)
      scale_dct_data = scale_dct.flatten().tobytes()
      scale_dct_encoded = base64.b64encode(zlib.compress(scale_dct_data, level=9)).decode('utf-8')
    else:
      scale_dct_encoded = '$'
    if params['lut'] is not None:
      lut = params['lut'].detach().cpu().numpy().astype(np.float32)
      lut_shape = np.array(lut.shape, dtype=np.uint16).tobytes()
      lut_shape_str = base64.b64encode(lut_shape).decode('utf-8')
      lut_data = lut.flatten().tobytes()
      lut_encoded = base64.b64encode(zlib.compress(lut_data, level=9)).decode('utf-8')
    else:
      lut_shape_str = '$'
      lut_encoded = '$'

    if all(x == '$' for x in [scale_dct_encoded, gamma_map_encoded, lut_encoded]):
      raise ValueError('All maps are None!')

    return f'{s_shape_str}:{gamma_map_encoded}:{scale_dct_encoded}:{lut_shape_str}:{lut_encoded}'


  @staticmethod
  def decode_params(comment: str, dtype: Optional[torch.dtype]=torch.float32,
                    device: Optional[torch.device]=torch.device('cuda')) -> Dict[str, Union[torch.Tensor, None]]:
    """Decodes JPEG Comment and return stored operators' parameters."""

    def decode_field(encoded: str, target_shape: Union[Tuple[int], Tuple[int, int, int, int]], dtype: np.dtype
                     ) -> Union[np.ndarray, None]:
      if encoded == '$':
        return None
      decompressed = zlib.decompress(base64.b64decode(encoded))
      array = np.frombuffer(decompressed, dtype=dtype)
      return np.array(array.reshape(target_shape))


    fields = comment.split(':')
    if len(fields) != 5:
      raise ValueError(f'Invalid comment format. Expected 5 fields, got {len(fields)}')
    s_shape_str, gamma_encoded, scale_dct_encoded, lut_shape_str, lut_encoded = fields
    if s_shape_str != '$':
      shape_bytes = base64.b64decode(s_shape_str)
      map_shape = tuple(np.frombuffer(shape_bytes, dtype=np.uint16))
    else:
      map_shape = None

    if lut_shape_str != '$':
      shape_bytes = base64.b64decode(lut_shape_str)
      lut_shape = tuple(np.frombuffer(shape_bytes, dtype=np.uint16))
    else:
      lut_shape = None
    gamma_map = decode_field(gamma_encoded, map_shape, np.float32) if gamma_encoded != '$' else None
    scale_dct = decode_field(scale_dct_encoded, (3, 1, 8, 8), np.float32)

    lut = decode_field(lut_encoded, lut_shape, np.float32) if lut_encoded != '$' else None

    return {
      'gamma_map': img_to_tensor(gamma_map).to(dtype=dtype, device=device
                                               ).unsqueeze(0) if gamma_map is not None else gamma_map,
      'scale_dct':  torch.from_numpy(scale_dct).to(dtype=dtype, device=device
                                                   ).unsqueeze(0) if scale_dct is not None else scale_dct,
      'lut': torch.from_numpy(lut).to(dtype=dtype, device=device) if lut is not None else lut,
    }

  def test_embedding_jpeg_comment(self) -> bool:
    """Tests if current design fits in JPEG comment."""
    input_img = torch.rand(1, 3, 3000, 4000).to(dtype=self._torch_dtype, device=next(self.parameters()).device)
    _, output_params = self(input_img)
    comment = self.encode_params(output_params)
    print(f'Size of comment = {len(comment)} Bytes')
    if len(comment) > 65533:
      print(f'Comment too large ({len(comment)} Bytes) for JPEG metadata.')
      return False
    try:
      imwrite(
        image=np.random.rand(256, 256, 3),
        output_path='image_with_comment',
        format='jpg',
        quality=self._quality,
        comment=comment
      )
      if not os.path.exists('image_with_comment.jpg'):
        return False
      os.remove('image_with_comment.jpg')
      return True
    except Exception as e:
      print(f'Failed to write image with comment: {e}')
      return False

  @staticmethod
  def _create_dct_basis(n: int, device: torch.device) -> torch.Tensor:
    """Creates nxn DCT (Type II) basis matrix."""
    basis = torch.zeros((n, n), device=device)
    for k in range(n):
      for i in range(n):
        alpha = math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n)
        basis[k, i] = alpha * math.cos(math.pi * (2 * i + 1) * k / (2 * n))
    return basis


  @staticmethod
  def _pad_to_block(x: torch.Tensor, block: int = 8) -> torch.Tensor:
    """Pads image to nearest multiple of `block`."""
    b, c, h, w = x.shape
    pad_h = (block - h % block) % block
    pad_w = (block - w % block) % block
    padded_x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant')
    return padded_x

  @staticmethod
  def _dct2_blockwise(raw_img: torch.Tensor, block: Optional[int]=8) -> torch.Tensor:
    """Applies block-wise 2D DCT to each 8x8 block in the image."""
    b, c, h, w = raw_img.shape
    assert h % block == 0 and w % block == 0, "Image must be divisible by block size"

    patches = F.unfold(raw_img, kernel_size=block, stride=block)
    patches = patches.view(b, c, block, block, -1).permute(0, 1, 4, 2, 3)
    dct_basis = JPEGAdapter._create_dct_basis(block, raw_img.device)
    out = torch.matmul(dct_basis, patches)  # DCT over rows
    out = torch.matmul(out.transpose(-2, -1), dct_basis)
    out = out.transpose(-2, -1)
    return out

  @staticmethod
  def _idct2_blockwise(coeffs: torch.Tensor, image_size: Tuple, block: Optional[int]=8) -> torch.Tensor:
    """Applies inverse block-wise DCT to reconstruct the image."""
    b, c, n, _, _ = coeffs.shape
    dct_basis = JPEGAdapter._create_dct_basis(block, coeffs.device)
    out = torch.matmul(dct_basis.T, coeffs)
    out = torch.matmul(out.transpose(-2, -1), dct_basis.T)
    out = out.transpose(-2, -1)
    out = out.permute(0, 1, 3, 4, 2).reshape(b, c * block * block, -1)
    img_recon = F.fold(out, output_size=image_size, kernel_size=block, stride=block)

    return img_recon
