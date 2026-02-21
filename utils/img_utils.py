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

This file contains image utility functions.
"""

from typing import List, Union, Dict, Any, Optional, Tuple

from fontTools.varLib.errors import UnsupportedFormat
import rawpy
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
import torch
import numpy as np
import cv2
import random
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image, ExifTags
from torchvision import transforms
from skimage import color
from utils.constants import *
from sklearn.linear_model import LinearRegression
import exiftool
from utils.exif_utils import parse_exif
import exifread
from fractions import Fraction
import itertools
from sklearn.linear_model import Ridge
from concurrent.futures import ThreadPoolExecutor


def normalize_raw(img: np.ndarray, black_level: Union[List, np.ndarray],
                  white_level: float) -> np.ndarray:
  """Normalizes raw image using black level and white level values.

  Args:
    img: Demosaiced/mosaiced RGB raw image in the format (height x width) or (height x width x 4).
    black_level: 4D vector of black levels.
    white_level: A scalar value of white level.

  Returns:
    Normalized image after black level correction (BLC).
  """
  raw = img.astype(np.float32)
  if raw.shape[-1] == 4:
      raw = (raw - black_level) / white_level
  else:
    height, width = raw.shape
    idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for i in range(len(black_level)):
        raw[idx[i][0]:height:2, idx[i][1]:width:2] = (raw[idx[i][0]:height:2,
                                                      idx[i][1]:width:2] - black_level[i]
                                                      ) / white_level
  return raw

def clip(x: np.ndarray, min_v: Optional[float] = 0.0,
         max_v: Optional[float] = 1.0) -> np.ndarray:
  """Limits the values in x by min_v and max_v."""
  return np.clip(x, a_min=min_v, a_max=max_v).astype(np.float32)

def extract_raw_metadata(file: str) -> Dict[str, Any]:
  """Extracts DNG metadata."""
  def wb_gains_to_illum_color(wb_gains: Union[List, np.ndarray]) -> np.ndarray:
    if not isinstance(wb_gains, np.ndarray):
      wb_gains = np.array(wb_gains)
    wb_gains = wb_gains.flatten()
    if len(wb_gains) == 4:
      if wb_gains[3] == 0:
        wb_gains = wb_gains[:-1]
      else:
        wb_gains[1] = wb_gains[1] + wb_gains[2]
        wb_gains[2] = wb_gains[3]
        wb_gains = wb_gains[:-1]
    illum_color = 1 / np.fmax(wb_gains, EPS)
    return illum_color / np.fmax(np.linalg.norm(illum_color), EPS)

  def get_orientation(file_path):
    image = Image.open(file_path)
    exif_data = image.getexif()
    data = {
      ExifTags.TAGS.get(tag, tag): value
      for tag, value in exif_data.items()
    }
    return data['Orientation']

  def process_color_matrix(ccm: np.ndarray) -> np.ndarray:
    if not isinstance(ccm, np.ndarray):
      ccm = np.array(ccm)
    if len(ccm.shape) == 1:
      ccm = ccm.reshape(3, -1)
    if ccm.size > 9:
      ccm = ccm[..., :-1]
    return ccm

  def get_pattern(
          pattern: str,
          raw_pattern: Optional[Union[List, np.ndarray]]=None) -> str:
    if raw_pattern is None:
      return pattern
    return ''.join([pattern[i] for i in np.array(raw_pattern).flatten()])

  try:
    orientation = get_orientation(file)
  except:
    orientation = 1

  with rawpy.imread(file) as raw:
    return {'black_level': raw.black_level_per_channel,
            'white_level': float(raw.white_level),
            'color_matrix': process_color_matrix(raw.color_matrix),
            'pattern': get_pattern(raw.color_desc.decode('utf-8'),  raw.raw_pattern),
            'illum_color': wb_gains_to_illum_color(raw.camera_whitebalance),
            'daylight_illum_color': wb_gains_to_illum_color(raw.daylight_whitebalance),
            'color_desc': raw.color_desc.decode('utf-8'),
            'raw_pattern': raw.raw_pattern,
            'orientation': orientation
            }

def apply_exif_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
  """Applies EXIF orientation correction to input RGB image."""
  if orientation == 1:
    return img
  elif orientation == 2:
    return np.fliplr(img)
  elif orientation == 3:
    return np.rot90(img, 2)
  elif orientation == 4:
    return np.flipud(img)
  elif orientation == 5:
    return np.rot90(np.fliplr(img), -1)
  elif orientation == 6:
    return np.rot90(img, -1)
  elif orientation == 7:
    return np.rot90(np.fliplr(img), 1)
  elif orientation == 8:
    return np.rot90(img, 1)
  else:
    return img

def undo_exif_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
  """Reverses the EXIF orientation correction (undoes rotation/flip)."""
  if orientation == 1:
    return img
  elif orientation == 2:
    return np.fliplr(img)
  elif orientation == 3:
    return np.rot90(img, -2)
  elif orientation == 4:
    return np.flipud(img)
  elif orientation == 5:
    return np.fliplr(np.rot90(img, 1))
  elif orientation == 6:
    return np.rot90(img, 1)
  elif orientation == 7:
    return np.fliplr(np.rot90(img, -1))
  elif orientation == 8:
    return np.rot90(img, -1)
  else:
    return img

def extract_image_from_dng(file: str) -> np.ndarray:
  """Extracts raw image from a DNG file."""
  raw = rawpy.imread(file)
  return raw.raw_image_visible.astype(np.float32)


def demosaice(img: np.ndarray, cfa_pattern: str, tile_mode: Optional[bool] = False,
              tile_size: Optional[int] = 256, overlap: Optional[int] = 16,
              num_workers: Optional[int] = 8) -> np.ndarray:
  """Performs image demosaicing to input image.

  Args:
    img: Input Bayer image.
    cfa_pattern: Color filter array (CFA) pattern (e.g., 'RGGB')
    tile_mode: Whether to process image in tiles (for speed/memory)
    tile_size: Size of each tile (square)
    overlap: Overlap (in pixels) between adjacent tiles
    num_workers: Number of threads for parallel processing

  Returns:
    RGB image after demosaicing.
  """
  cfa_pattern = cfa_pattern.upper()

  if not tile_mode:
    return clip(demosaicing_CFA_Bayer_Menon2007(img, cfa_pattern))

  h, w = img.shape
  rgb_out = np.zeros((h, w, 3), dtype=np.float32)

  def process_tile(y0: int, x0: int, y1: int, x1: int) -> Tuple[int, int, np.ndarray]:
    sub_img = img[y0:y1, x0:x1]
    demosaiced_rgb = demosaicing_CFA_Bayer_Menon2007(sub_img, cfa_pattern)
    return y0, x0, demosaiced_rgb

  coords = []
  for y in range(0, h, tile_size - overlap):
    for x in range(0, w, tile_size - overlap):
      y_0, x_0 = y, x
      y_1 = min(y + tile_size, h)
      x_1 = min(x + tile_size, w)
      coords.append((y_0, x_0, y_1, x_1))

  with ThreadPoolExecutor(max_workers=num_workers) as ex:
    for (y0, x0, y1, x1), (_, _, rgb) in zip(coords, ex.map(lambda c: process_tile(*c), coords)):
      h_t, w_t = rgb.shape[:2]
      oy0 = y0 + (overlap // 2 if y0 > 0 else 0)
      ox0 = x0 + (overlap // 2 if x0 > 0 else 0)
      oy1 = min(y0 + h_t - (overlap // 2 if y1 < h else 0), h)
      ox1 = min(x0 + w_t - (overlap // 2 if x1 < w else 0), w)

      cy0 = oy0 - y0
      cx0 = ox0 - x0
      cy1 = cy0 + (oy1 - oy0)
      cx1 = cx0 + (ox1 - ox0)
      rgb_out[oy0:oy1, ox0:ox1] = rgb[cy0:cy1, cx0:cx1]

    return clip(rgb_out)


def imwrite(image: np.ndarray, output_path: str, format: str, quality: Optional[int]=95, comment: Optional[
  Union[str, bytes]]=None
            ) -> str:
  """Saves an image to a file in the specified format.

  Args:
    image: An array representing the image (height x width x channel).
    output_path: File path without extension where the image will be saved.
    format: The desired file format: 'PNG-16', 'PNG-8', or 'JPEG'.
    quality: JPEG quality level (0-100), default is 95.
    comment: Optional JPEG comment to embed (must be ASCII, max ~64KB). Can be str or bytes.

  Raises:
    UnsupportedFormat: If the specified format is invalid.
  """
  format = format.lower()
  image = clip(image)
  ext_map = {
    'png-16': ('.png', np.uint16, 16),
    'png-8': ('.png', np.uint8, 8),
    'jpeg': ('.jpg', np.uint8, 8),
    'jpg': ('.jpg', np.uint8, 8),
  }

  if format in ext_map:
    ext, dtype, bit_depth = ext_map[format]
    image = (image * (2 ** bit_depth - 1)).astype(dtype)
  else:
    raise UnsupportedFormat(f"Format '{format}' is not supported.")

  output_file = f"{os.path.splitext(output_path)[0]}{ext}"
  if format in ['jpeg', 'jpg']:
    if comment is None:
      cv2.imwrite(output_file, convert_bgr_rgb(image), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
      pil_img = Image.fromarray(image)
      if isinstance(comment, str):
        comment_bytes = comment.encode('ascii', errors='ignore')
      else:
        comment_bytes = comment
      pil_img.save(output_file, format='JPEG', quality=quality, comment=comment_bytes)

  else:
    cv2.imwrite(output_file, convert_bgr_rgb(image))

  return output_file

def imencode(image: np.ndarray, quality: Optional[int]=95) -> np.ndarray:
  image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
  image = convert_bgr_rgb(image)
  success, jpeg_buf = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
  if not success:
    raise RuntimeError('Failed to encode sRGB image.')
  return jpeg_buf

def imdecode(image_but: np.ndarray) -> np.ndarray:
  image = cv2.imdecode(image_but, cv2.IMREAD_UNCHANGED)
  image = convert_bgr_rgb(image)
  return im2double(image)


def raw_to_lsrgb(img: Union[np.ndarray, torch.Tensor],
                 illum_color: Union[np.ndarray, List, torch.Tensor],
                 ccm: Union[np.ndarray, List, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
  """Converts image from raw space to linear sRGB.

  Args:
    img: Input image as a NumPy array (H x W x 3) or torch tensor (B x 3 x H x W).
    illum_color: Illuminant color as a numpy/list (3,) or torch tensor (B x 3).
    ccm: Color correction matrix as numpy/list (3 x 3) or torch tensor (B x 3 x 3).

  Returns:
    Linear sRGB image(s) in the same type and shape of input image (img).
  """
  if (isinstance(img, np.ndarray) and (isinstance(illum_color, np.ndarray) or isinstance(illum_color, list)) and
       (isinstance(ccm, np.ndarray) or isinstance(ccm, list))):
    if isinstance(illum_color, list):
      illum_color = np.array(illum_color, dtype=img.dtype)
    if isinstance(ccm, list):
      ccm = np.array(ccm, dtype=img.dtype)

    img_reshaped = img.reshape(-1, 3)
    wb_gain = illum_color[1] / illum_color
    color_matrix = ccm @ np.diag(wb_gain)
    img_out = (img_reshaped @ color_matrix.T).reshape(img.shape)
    return np.clip(img_out, 0.0, 1.0)


  elif isinstance(img, torch.Tensor) and isinstance(illum_color, torch.Tensor) and isinstance(ccm, torch.Tensor):
    b, c, h, w = img.shape
    wb_gain = illum_color[:, 1].unsqueeze(-1) / (illum_color + EPS)
    color_matrix = ccm * wb_gain.unsqueeze(1)
    img_flat = img.permute(0, 2, 3, 1).reshape(b, -1, 3)
    img_out = torch.bmm(img_flat, color_matrix.transpose(1, 2))
    img_out = img_out.view(b, h, w, 3).permute(0, 3, 1, 2)
    return torch.clamp(img_out, 0.0, 1.0)

  else:
    raise TypeError('Input must be a NumPy array or Torch tensor.')

def raw_to_srgb(img: Union[np.ndarray, torch.Tensor],
                illum_color: Union[np.ndarray, List, torch.Tensor],
                ccm: Union[np.ndarray, List, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
  """Converts image from raw space to sRGB (simple rendering).

  Args:
    img: Input image as a NumPy array (H x W x 3) or torch tensor (B x 3 x H x W).
    illum_color: Illuminant color as a numpy/list (3,) or torch tensor (B x 3).
    ccm: Color correction matrix as numpy/list (3 x 3) or torch tensor (B x 3 x 3).

  Returns:
    sRGB image(s) in the same type and shape of input image (img).
  """
  lsrgb_img = raw_to_lsrgb(img=img, illum_color=illum_color, ccm=ccm)
  return lsrgb_img ** (1 / 2.2)


def im2double(img: np.ndarray) -> np.ndarray:
  """ Converts image to floating-point format [0-1]."""
  if img[0].dtype == 'uint8':
    max_value = 255
  elif img[0].dtype == 'uint16':
    max_value = 65535
  else:
    raise UnsupportedFormat
  return img.astype('float') / max_value

def imread(img_file: str, single_channel: Optional[bool] = False, normalize: Optional[bool] = True,
           load_comment: Optional[bool] = False, return_dtype: Optional[bool]=False
           ) -> Union[
  np.ndarray, Tuple[np.ndarray, bytes], Tuple[np.ndarray, np.dtype], Tuple[np.ndarray, bytes, np.dtype]]:
  """Reads RGB image file with optional normalization and comment extraction (JPEG only)."""
  if not os.path.exists(img_file):
    raise FileNotFoundError(f'Image not found: {img_file}')
  ext = os.path.splitext(img_file)[-1].lower()
  if load_comment and ext in ['.jpg', '.jpeg']:
    img = Image.open(img_file)
    comment = img.info.get('comment', b"")
    img = np.array(img)
    if img.ndim == 2 and not single_channel:
      img = np.stack([img] * 3, axis=-1)
  else:
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    comment = None
    if img is None:
      raise FileNotFoundError(f'Cannot load image: {img_file}')
    if not single_channel:
      img = convert_bgr_rgb(img)
  if return_dtype:
    dtype = img.dtype
  else:
    dtype = None
  if normalize:
    img = im2double(img)
  if load_comment and return_dtype:
    additional_output = (comment, dtype)
  elif load_comment:
    additional_output = comment
  elif return_dtype:
    additional_output = dtype
  else:
    additional_output = None
  if img.ndim > 2 and img.shape[2] > 3:
    # Handling PNG images with alpha channels
    img = img[:, :, 1:]
  return (img, additional_output) if additional_output is not None else img

def img_to_tensor(img):
  """Converts a given ndarray image to torch tensor image."""
  dims = len(img.shape)
  assert (dims == 3 or dims == 4)
  if dims == 3:
    img = img.transpose((2, 0, 1))
  elif dims == 4:
    img = img.transpose((3, 2, 0, 1))
  else:
    raise NotImplementedError
  return torch.from_numpy(img)

def tensor_to_img(img):
  """Converts a given torch tensor image to an ndarray image."""
  dims = len(img.shape)
  assert (dims == 3 or dims == 4)
  if dims == 3:
    img = img.permute(1, 2, 0)
  elif dims == 4:
    img = img.permute(2, 3, 1, 0).squeeze(dim=-1)
  else:
    raise NotImplementedError
  return img.detach().cpu().numpy()

def convert_bgr_rgb(img: np.ndarray) -> np.ndarray:
  """Converts BGR/RGB image to RGB/BGR image."""
  return img[..., ::-1]

def shift_image(img: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
  """Shifts an image by a given amount along the x and y axes.

  Args:
    img: Input image as a NumPy array (H x W x C).
    shift_x: Number of pixels to shift along the x-axis. Positive values shift right, negative left.
    shift_y: Number of pixels to shift along the y-axis. Positive values shift down, negative up.

  Returns:
    A shifted version of the input image.
  """
  h, w, c = img.shape
  translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
  shifted_img = cv2.warpAffine(img, translation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
  return shifted_img

def augment_img(img: np.ndarray, gt: Optional[np.ndarray]=None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
  """Applies simple augmentations: horizontal/vertical flip and random shift."""
  if random.random() < 0.5:
    img = np.flip(img, axis=0)
    if gt is not None:
      gt = np.flip(gt, axis=0)
  if random.random() < 0.5:
    img = np.flip(img, axis=1)
    if gt is not None:
      gt = np.flip(gt, axis=1)
  max_shift = 5
  shift_x = random.randint(-max_shift, max_shift)
  shift_y = random.randint(-max_shift, max_shift)
  img = shift_image(img, shift_x, shift_y)
  if gt is not None:
    gt = shift_image(gt, shift_x, shift_y)
    return img, gt
  return img


def get_ssim(source: np.ndarray, reference: np.ndarray) -> float:
  """Computes the SSIM between two color images."""
  return structural_similarity(source, reference, multichannel=True,
                               channel_axis=2, data_range=1)

def get_psnr(source: np.ndarray, reference: np.ndarray) -> float:
  """Computes PSNR between two color images."""
  return peak_signal_noise_ratio(source, reference)

def get_lpips(source: np.ndarray, reference: np.ndarray,
              lpips_model: object, image_size: Optional[int]=1024,
              device: Optional[str]='gpu') -> float:
  """Computes LPIPS between two color images."""

  def preprocess_image(image: np.ndarray, target_size: Optional[int]=1024
                       ) -> torch.Tensor:
    """Pre-process image for LPIPS"""
    transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((target_size, target_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

  device = torch.device('cuda' if device.lower() == 'gpu' else 'cpu')

  source_tensor = preprocess_image(source, target_size=image_size).to(device=device)
  reference_tensor = preprocess_image(reference, target_size=image_size).to(device=device)
  source_tensor = source_tensor.unsqueeze(0)
  reference_tensor = reference_tensor.unsqueeze(0)
  return lpips_model(source_tensor, reference_tensor).item()

def get_delta_e(source: np.ndarray, reference: np.ndarray) -> float:
  """Computes DeltaE 2000 between two color images."""

  def delta_e_2000(input_img: np.ndarray, gt_img: np.ndarray) -> float:
    """Computes the DeltaE 2000 color difference between two Lab color representations."""
    kl, kc, kh = 1, 1, 1
    l_source, a_source, b_source = input_img[:, 0], input_img[:, 1], input_img[:, 2]
    l_target, a_target, b_target = gt_img[:, 0], gt_img[:, 1], gt_img[:, 2]
    norm_source = np.sqrt(a_source ** 2 + b_source ** 2)
    norm_target = np.sqrt(a_target ** 2 + b_target ** 2)
    avg_norm = (norm_source + norm_target) / 2
    g = 0.5 * (1 - np.sqrt(np.power(avg_norm, 7) / (
       np.power(avg_norm, 7) + np.power(25, 7))))
    ap_source = (1 + g) * a_source
    ap_target = (1 + g) * a_target
    cp_source = np.sqrt(ap_source ** 2 + b_source ** 2)
    cp_target = np.sqrt(ap_target ** 2 + b_target ** 2)
    cp_prob = cp_target * cp_source
    zcidx = np.argwhere(cp_prob == 0)
    hpstd = np.arctan2(b_source, ap_source)
    hpstd[np.abs(ap_source) + np.abs(b_source) == 0] = 0
    hpsample = np.arctan2(b_target, ap_target)
    hpsample = np.mod(hpsample + 2 * np.pi, 2 * np.pi)
    hpsample[np.abs(ap_target) + np.abs(b_target) == 0] = 0
    dL = l_target - l_source
    dC = cp_target - cp_source
    dhp = hpsample - hpstd
    dhp = np.mod(dhp + np.pi, 2 * np.pi) - np.pi
    dhp[zcidx] = 0
    dH = 2 * np.sqrt(cp_prob) * np.sin(dhp / 2)
    Lp = (l_target + l_source) / 2
    Cp = (cp_source + cp_target) / 2
    hp = (hpstd + hpsample) / 2
    hp = np.mod(hp - (np.abs(hpstd - hpsample) > np.pi) * np.pi, 2 * np.pi)
    hp[zcidx] = hpsample[zcidx] + hpstd[zcidx]
    Lpm502 = (Lp - 50) ** 2
    Sl = 1 + 0.015 * Lpm502 / np.sqrt(20 + Lpm502)
    Sc = 1 + 0.045 * Cp
    T = 1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp) + 0.32 * np.cos(
      3 * hp + np.pi / 30) - 0.20 * np.cos(
      4 * hp - 63 * np.pi / 180)
    Sh = 1 + 0.015 * Cp * T
    delthetarad = (30 * np.pi / 180) * np.exp(
      -((180 / np.pi * hp - 275) / 25) ** 2)
    Rc = 2 * np.sqrt(Cp ** 7 / (Cp ** 7 + 25 ** 7))
    RT = - np.sin(2 * delthetarad) * Rc
    klSl = kl * Sl
    kcSc = kc * Sc
    khSh = kh * Sh
    de00 = np.sqrt(
      (dL / klSl) ** 2 + (dC / kcSc) ** 2 + (
         dH / khSh) ** 2 + RT * (dC / kcSc) * (dH / khSh)
    )
    return de00

  source = color.rgb2lab(source)
  reference = color.rgb2lab(reference)
  source = np.reshape(source, [-1, 3]).astype(np.float32)
  reference = np.reshape(reference, [-1, 3]).astype(np.float32)

  return np.mean(delta_e_2000(source, reference))

def poly_kernel(rgb):
  """Applies a kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)."""
  return (np.transpose((rgb[:, 0], rgb[:, 1], rgb[:, 2], rgb[:, 0] * rgb[:, 1],
                        rgb[:, 0] * rgb[:, 2], rgb[:, 1] * rgb[:, 2], rgb[:, 0] * rgb[:, 0],
                        rgb[:, 1] * rgb[:, 1], rgb[:, 2] * rgb[:, 2],
                        rgb[:, 0] * rgb[:, 1] * rgb[:, 2],
                        np.repeat(1, np.shape(rgb)[0]))))

def get_mapping_func(source: np.ndarray, target: np.ndarray,
                     alpha: Optional[float]=0.0,
                     discard_saturated: Optional[bool]=False,
                     saturation_percentile: Optional[float]=0.9,
                     p_kernel: Optional[bool]=True):
  """Computes a polynomial mapping to map the source image to the target image.

  Args:
    source: The source image with shape (H, W, 3) or (N, 3), where H and W are height and width, and N is # of pixels.
    target: The target image with shape (H, W, 3) or (N, 3).
    alpha: Regularization strength for Ridge regression. Default is 0.0 (no regularization) -- uses LinearRegression.
    discard_saturated: If True, pixels with extreme values (saturated) in the source or target images are discarded
      before fitting the model.
    saturation_percentile: The percentile used to identify saturated pixels. Default is 0.9.
    p_kernel: If True, polynomial mapping is applied to the data before fitting the model. Default is True.

  Returns:
    Fitted model (matrix) that can be used to transform source images to the target space.
  """
  source = np.reshape(source, [-1, 3])
  target = np.reshape(target, [-1, 3])

  if discard_saturated:
    source_avg = np.mean(source, axis=1)
    target_avg = np.mean(target, axis=1)
    source_threshold = np.percentile(source_avg, saturation_percentile * 100)
    target_threshold = np.percentile(target_avg, saturation_percentile * 100)
    saturated_mask = (source_avg > source_threshold) | (source_avg < (1 - source_threshold)) | \
                     (target_avg > target_threshold) | (target_avg < (1 - target_threshold))
    source = source[~saturated_mask]
    target = target[~saturated_mask]

  if alpha:
    m = Ridge(alpha=alpha)
    m.fit(poly_kernel(source), target) if p_kernel else m.fit(source, target)
  else:
    m = LinearRegression().fit(poly_kernel(source), target) if p_kernel else LinearRegression().fit(source, target)
  return m

def apply_mapping_func(image, m, p_kernel: Optional[bool]=True):
  """Applies the mapping transformation to an image using the fitted model.

  Args:
    image: The image to be transformed, with shape (H, W, 3) or (N, 3).
    m: The fitted mapping model obtained from `get_mapping_func`. Can be a Ridge or Linear regression model.
    p_kernel: If True, polynomial mapping is applied to the image before transformation. Default is True.

  Returns:
    The transformed image, with the same shape as the input image.
  """
  sz = image.shape
  image = np.reshape(image, [-1, 3])
  result = m.predict(poly_kernel(image)) if p_kernel else m.predict(image)
  result = np.reshape(result, sz)
  return result

def extract_additional_dng_metadata(dng_file: str) -> Dict[str, Any]:
  """Extracts additional DNG metadata."""

  def warn(msg):
    print(f'[WARN] {msg}: {dng_file}')

  def convert_to_list(value: str) -> Any:
    """Helper function to convert a string to a np.array, or return None if invalid."""
    try:
        if value is None:
          return value
        return np.array([float(x) for x in value.split()])
    except ValueError:
        return None

  def _get_tag_from_ifd(ifd, tag_id):
    try:
      if ifd and tag_id in ifd.tags:
        return ifd.tags[tag_id].values
    except Exception:
      pass
    return None

  def _bytes_to_str(x):
    try:
      if isinstance(x, (list, tuple)):
        chars = []
        for v in x:
          if isinstance(v, bytes):
            chars.append(v.decode('ascii', errors='ignore'))
          elif isinstance(v, int):
            chars.append(chr(v))
        return ''.join(chars).strip('\x00').strip()
      if isinstance(x, bytes):
        return x.decode('ascii', errors='ignore').strip("\x00").strip()
      if isinstance(x, str):
        return x
    except Exception:
      pass
    return None

  def _to_float_scalar(x):
    try:
      if isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
      if isinstance(x, Fraction):
        return float(x)
      if isinstance(x, (int, float, np.number)):
        return float(x)
    except Exception:
      pass
    return None

  def _to_int_scalar(x):
    try:
      if isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
      if isinstance(x, (int, np.integer)):
        return int(x)
    except Exception:
      pass
    return None

  def _to_array(x):
    try:
      if x is None:
        return None
      return np.array(x, dtype=np.float32)
    except Exception:
      return None

  def _to_3x3(x):
    try:
      if x is None:
        return None
      x = np.asarray(x, dtype=np.float32).reshape(-1)
      if x.size == 9:
        return x.reshape(3, 3)
      if x.size == 18:
        return x.reshape(3, 6)[:, :3]
    except Exception:
      pass
    return None

  def _exifread_float(tags, name):
    try:
      tag = tags.get(name)
      if tag is None:
        return None
      val = tag.values
      if isinstance(val, list) and len(val) == 1:
        val = val[0]
      if isinstance(val, Fraction):
        return float(val)
      if isinstance(val, (int, float)):
        return float(val)
    except Exception:
      pass
    return None

  try:
    with exiftool.ExifTool(EXIFTOOL_PATH) as et:
      metadata_list = et.execute_json(dng_file)

    metadata = metadata_list[0] if metadata_list else {}
    data = {
      'make': metadata.get('EXIF:Make', None),
      'model': metadata.get('EXIF:Model', None),
      'exposure_time': metadata.get('EXIF:ExposureTime', None),
      'f_number': metadata.get('EXIF:FNumber', None),
      'iso': metadata.get('EXIF:ISO', None),
      'focal_length': metadata.get('EXIF:FocalLength', None),
      'color_matrix1': convert_to_list(metadata.get('EXIF:ColorMatrix1', None)),
      'color_matrix2': convert_to_list(metadata.get('EXIF:ColorMatrix2', None)),
      'camera_calibration1': convert_to_list(metadata.get('EXIF:CameraCalibration1', None)),
      'camera_calibration2': convert_to_list(metadata.get('EXIF:CameraCalibration2', None)),
      'as_shot_neutral': convert_to_list(metadata.get('EXIF:AsShotNeutral', None)),
      'calibration_illuminant1': metadata.get('EXIF:CalibrationIlluminant1', None),
      'calibration_illuminant2': metadata.get('EXIF:CalibrationIlluminant2', None),
      'forward_matrix1': convert_to_list(metadata.get('EXIF:ForwardMatrix1', None)),
      'forward_matrix2': convert_to_list(metadata.get('EXIF:ForwardMatrix2', None)),
      'noise_profile': convert_to_list(metadata.get('EXIF:NoiseProfile', None)),
      'aperture': metadata.get('Composite:Aperture', None),
      'shutter_speed': metadata.get('Composite:ShutterSpeed', None),
      'fov': metadata.get('Composite:FOV', None),
      'width': metadata.get('EXIF:ImageWidth', None),
      'height': metadata.get('EXIF:ImageHeight', None),
    }
    return data

  except Exception as e:
    warn(f'ExifTool failed for {dng_file}: {e}. Resolving metadata extraction...\n')
    try:
      ifds = parse_exif(dng_file, verbose=False)
    except Exception as e:
      warn(f'Failed to parse EXIF ({e})')
      return {}

    ifd0 = None
    exif_ifd = None
    for ifd in ifds.values():
      if 0x010F in ifd.tags or 0x0110 in ifd.tags:
        ifd0 = ifd
        break
    exif_tags = {0x829A, 0x829D, 0x8827, 0x920A}
    for ifd in ifds.values():
      if any(t in ifd.tags for t in exif_tags):
        exif_ifd = ifd
        break
    try:
      cm1 = _to_3x3(_get_tag_from_ifd(ifd0, 0xC621))
      fm1 = _to_3x3(_get_tag_from_ifd(ifd0, 0xC714))
      asn = _to_array(_get_tag_from_ifd(ifd0, 0xC628))

      if cm1 is None or fm1 is None or asn is None:
        raise ValueError('Missing required DNG color metadata')
    except Exception as e:
      warn(str(e))
      return {}
    make = _bytes_to_str(_get_tag_from_ifd(ifd0, 0x010F))
    model = _bytes_to_str(_get_tag_from_ifd(ifd0, 0x0110))

    exposure_time = _to_float_scalar(_get_tag_from_ifd(exif_ifd, 0x829A))
    f_number = _to_float_scalar(_get_tag_from_ifd(exif_ifd, 0x829D))
    iso = _to_int_scalar(_get_tag_from_ifd(exif_ifd, 0x8827))
    focal_length = _to_float_scalar(_get_tag_from_ifd(exif_ifd, 0x920A))

    width = _to_int_scalar(_get_tag_from_ifd(ifd0, 0xA002))
    height = _to_int_scalar(_get_tag_from_ifd(ifd0, 0xA003))

    calib_illum_1 = _to_int_scalar(_get_tag_from_ifd(ifd0, 0xC65A))
    calib_illum_2 = _to_int_scalar(_get_tag_from_ifd(ifd0, 0xC65B))
    try:
      if exposure_time is None or f_number is None or focal_length is None:
        with open(dng_file, 'rb') as f:
          tags = exifread.process_file(f, details=False)

        if exposure_time is None:
          exposure_time = _exifread_float(tags, 'EXIF ExposureTime')

        if f_number is None:
          f_number = _exifread_float(tags, 'EXIF FNumber')

        if focal_length is None:
          focal_length = _exifread_float(tags, 'EXIF FocalLength')

    except Exception as e:
      warn(f'Exifread fallback failed ({e})')
    aperture = f_number
    shutter_speed = exposure_time

    return {
      'make': make,
      'model': model,
      'exposure_time': exposure_time,
      'f_number': f_number,
      'iso': iso,
      'focal_length': focal_length,
      'width': width,
      'height': height,
      'color_matrix1': cm1.flatten(),
      'color_matrix2': _to_array(_get_tag_from_ifd(ifd0, 0xC622)),
      'forward_matrix1': fm1.flatten(),
      'forward_matrix2': _to_array(_get_tag_from_ifd(ifd0, 0xC715)),
      'camera_calibration1': _to_array(_get_tag_from_ifd(ifd0, 0xC623)),
      'camera_calibration2': _to_array(_get_tag_from_ifd(ifd0, 0xC624)),
      'as_shot_neutral': asn,
      'calibration_illuminant1': calib_illum_1,
      'calibration_illuminant2': calib_illum_2,
      'aperture': aperture,
      'shutter_speed': shutter_speed,
      'fov': None,
      'noise_profile': None,
    }

def imresize(img: np.ndarray, height: int, width: int, interpolation_method: Optional[str] = 'linear') -> np.ndarray:
  """Resizes an image to a target size with a specified interpolation method.

  Args:
      img: Input image.
      height: Target height.
      width: Target width.
      interpolation_method: One of 'linear', 'bicubic', or 'nearest'. Defaults to 'linear'.

  Returns:
      np.ndarray: Resized image.
  """
  if img.shape[:2] == (height, width):
    return img

  interpolation_methods = {
    'linear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'nearest': cv2.INTER_NEAREST
  }
  interpolation = interpolation_methods.get(interpolation_method.lower(), cv2.INTER_LINEAR)
  return cv2.resize(img, (width, height), interpolation=interpolation)


def rgb_to_rgbg(rgb: np.ndarray) -> np.ndarray:
  """Converts RGB colors to R/G, B/G."""
  return np.stack([rgb[..., 0] / (rgb[..., 1] + EPS),
                   rgb[..., 2] / (rgb[..., 1] + EPS)], axis=-1)

def rgb_to_uv(rgb: np.ndarray) -> np.ndarray:
  """Converts RGB colors to uv-log space."""
  log_rgb = np.log(rgb + EPS)
  u = log_rgb[:, 1] - log_rgb[:, 0]
  v = log_rgb[:, 1] - log_rgb[:, 2]
  return np.stack([u, v], axis=-1)

def compute_edges(img: np.ndarray) -> np.ndarray:
  """Calculates the gradient intensity for a given image."""
  edge_img = np.zeros(img.shape)
  img_pad = cv2.copyMakeBorder(img, top=1, bottom=1, left=1, right=1,
                               borderType=cv2.BORDER_REFLECT)
  offsets = [-1, 0, 1]
  for dx, dy in itertools.product(offsets, repeat=2):
    if dx == 0 and dy == 0:
      continue
    edge_img[:, :, :] += np.abs(
      img[:, :, :] - img_pad[1 + dx: img.shape[0] + 1 + dx,
                     1 + dy: img.shape[1] + 1 + dy, :]
    )
  edge_img /= 8
  return edge_img


def compute_2d_rgbg_histogram(img: np.ndarray, hist_boundaries: List[float], hist_bins: int, edge_hist: bool,
                              uv_coord: bool) -> np.ndarray:
  """Returns 2D R/G B/G histogram stats of given image(s).

  Args:
    img: Input image as a NumPy array (H x W x C).
    hist_boundaries: A list of min u (R/G), min v (B/G), max u, and max v values of the histogram boundaries.
    hist_bins: Number of bins in each direction of the 2D histogram.
    edge_hist: A flag to create histogram of edge image.
    uv_coord: A flag to append u/v positional info to the 2D histogram feature.

  Returns:
    A 2D histogram of color image, and optionally edge image & uv_coordinates.
  """
  img_reshaped = img.reshape([-1, 3])
  u_min, v_min, u_max, v_max = hist_boundaries
  brightness = np.linalg.norm(img_reshaped, axis=1)
  img_chroma = rgb_to_rgbg(img_reshaped)
  hist, _, _ = np.histogram2d(img_chroma[:, 0], img_chroma[:, 1], bins=hist_bins,
                              range=((u_min, u_max), (v_min, v_max)),
                              weights=brightness)
  norm_factor = np.sum(hist) + EPS
  hist = np.sqrt(hist / norm_factor)
  if uv_coord:
    x_bin_indices, y_bin_indices = np.meshgrid(np.arange(hist_bins), np.arange(hist_bins),
                                               indexing="ij")
    x_bin_indices = x_bin_indices / hist_bins
    y_bin_indices = y_bin_indices / hist_bins
    hist = np.stack([hist, x_bin_indices, y_bin_indices], axis=-1)

  if edge_hist:
    edge_img = compute_edges(img)
    edge_chroma = rgb_to_rgbg(edge_img).reshape([-1, 2])
    edge_hist, _, _ = np.histogram2d(edge_chroma[:, 0], edge_chroma[:, 1], bins=hist_bins,
                                     range=((u_min, u_max), (v_min, v_max)),
                                     weights=brightness)
    norm_factor = np.sum(edge_hist) + EPS
    edge_hist = np.sqrt(edge_hist / norm_factor)
    if uv_coord:
      hist = np.concatenate([hist, np.expand_dims(edge_hist, axis=-1)], axis=-1)
    else:
      hist = np.stack([hist, edge_hist], axis=-1)
  return hist

def compute_snr(img: np.ndarray, patch_size: Optional[int] = 15,
                stride: Optional[int] = 1) -> np.ndarray:
  """Computes signal-to-noise-ratio (SNR) image of a given image."""
  img_tensor = torch.from_numpy(img).float()
  if img_tensor.ndimension() != 3:
    raise ValueError("Input image must be a 3D array with shape (H, W, C).")
  img_tensor = img_tensor.permute(2, 0, 1)
  pad_height = (patch_size - stride) // 2
  pad_width = (patch_size - stride) // 2
  img_padded = torch.nn.functional.pad(img_tensor, (pad_width, pad_width, pad_height, pad_height),
                                       mode='reflect')

  patches = img_padded.unfold(1, patch_size, stride).unfold(
    2, patch_size, stride)
  patches = patches.contiguous().view(img_tensor.size(0), -1,
                                      patch_size * patch_size)

  signal_mean = patches.mean(dim=-1, keepdim=True)
  noise_std = patches.std(dim=-1, keepdim=True)
  snr_map = 10 * torch.log10(signal_mean / (noise_std + EPS))
  num_patches_height = (img_tensor.size(1) + 2 * pad_height - patch_size) // stride + 1
  num_patches_width = (img_tensor.size(2) + 2 * pad_width - patch_size) // stride + 1
  snr_map = snr_map.view(img_tensor.size(0), num_patches_height, num_patches_width)
  snr_map = snr_map.permute(1, 2, 0)
  snr_map = torch.clamp(snr_map / 100, 0, 1)
  return snr_map.numpy()

def extract_non_overlapping_patches(img: np.ndarray, gt_img: Optional[np.ndarray] = None,
                                    patch_size: Optional[int] = 228, num_patches: Optional[int] = 0,
                                    allow_overlap: Optional[bool] = False,
                                    add_resized_patch: Optional[bool] = False) -> Dict[str, List[np.ndarray]]:
  """
  Extracts non-overlapping patches from an input image and optionally from a ground-truth image.
  If allow_overlap is True, extra patches are extracted near the borders to cover the full image.

  Args:
    img: Input image of shape (height, width, [channels]).
    gt_img: Optional ground-truth image with the same dimensions as 'img'.
    patch_size: Size of square patches (width = height).
    num_patches: Number of patches to extract. If set to 0, all valid patches are returned.
    allow_overlap: If True, adds extra overlapping patches to cover remaining border regions.
    add_resized_patch: If True, resize the full image to patch size and include it as one more patch.

  Returns:
    A dictionary containing:
      - 'img': A list of extracted patches from 'img'.
      - 'xy': A list of (x, y) coordinates for the top-left corner of each extracted patch.
      - 'gt': A list of extracted patches from 'gt_img' (if provided).
  """
  h, w = img.shape[:2]

  if gt_img is not None:
    assert gt_img.shape[:2] == (h, w), 'Input image and ground-truth image must have the same dimensions.'
  y_steps = list(range(0, h - patch_size + 1, patch_size))
  x_steps = list(range(0, w - patch_size + 1, patch_size))
  if allow_overlap:
    if h % patch_size != 0:
      y_steps.append(h - patch_size)
    if w % patch_size != 0:
      x_steps.append(w - patch_size)
  patch_positions = [(y, x) for y in y_steps for x in x_steps]
  if num_patches == 0 or num_patches >= len(patch_positions):
    chosen_positions = patch_positions
  else:
    chosen_positions = random.sample(patch_positions, num_patches)
  img_patches = [img[y:y + patch_size, x:x + patch_size, ...] for y, x in chosen_positions]
  patch_coords = [(x, y) for y, x in chosen_positions]
  if add_resized_patch:
    resized_img = imresize(img, height=patch_size, width=patch_size, interpolation_method='bicubic')
    img_patches.append(resized_img)
    patch_coords.append((-1, -1))  # use (-1, -1) to indicate resized full image
  patches = {'img': img_patches, 'xy': patch_coords}
  if gt_img is not None:
    gt_patches = [gt_img[y:y + patch_size, x:x + patch_size, ...] for y, x in chosen_positions]
    if add_resized_patch:
      resized_gt = imresize(gt_img, height=patch_size, width=patch_size, interpolation_method='bicubic')
      gt_patches.append(resized_gt)
    patches['gt'] = gt_patches
  return patches




