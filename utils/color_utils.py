"""
Copyright (c) 2025 Samsung Electronics Co., Ltd.

Author(s):
Luxi Zhao (lucy.zhao@samsung.com, lucyzhao.zlx@gmail.com)
Mahmoud Afifi (m.afifi1@samsung.com, m.3afifi@gmail.com)
Abdelrahman Abdelhamed (a.abdelhamed@samsung.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

Utility functions for working with correlated color temperatures and color space conversions.
"""

import numpy as np
import sys
from typing import Optional, Union, List, Tuple, Sequence
from utils.constants import XYZ_TO_SRGB_D50, EPS, CALIB_ILLUM1, CALIB_ILLUM2

list_or_np_array = Union[List, np.ndarray]

rt = [  # /* reciprocal temperature (K) */
  sys.float_info.min, 10.0e-6, 20.0e-6, 30.0e-6, 40.0e-6, 50.0e-6,
  60.0e-6, 70.0e-6, 80.0e-6, 90.0e-6, 100.0e-6, 125.0e-6,
  150.0e-6, 175.0e-6, 200.0e-6, 225.0e-6, 250.0e-6, 275.0e-6,
  300.0e-6, 325.0e-6, 350.0e-6, 375.0e-6, 400.0e-6, 425.0e-6,
  450.0e-6, 475.0e-6, 500.0e-6, 525.0e-6, 550.0e-6, 575.0e-6,
  600.0e-6
]

uvt = [
  [0.18006, 0.26352, -0.24341],
  [0.18066, 0.26589, -0.25479],
  [0.18133, 0.26846, -0.26876],
  [0.18208, 0.27119, -0.28539],
  [0.18293, 0.27407, -0.30470],
  [0.18388, 0.27709, -0.32675],
  [0.18494, 0.28021, -0.35156],
  [0.18611, 0.28342, -0.37915],
  [0.18740, 0.28668, -0.40955],
  [0.18880, 0.28997, -0.44278],
  [0.19032, 0.29326, -0.47888],
  [0.19462, 0.30141, -0.58204],
  [0.19962, 0.30921, -0.70471],
  [0.20525, 0.31647, -0.84901],
  [0.21142, 0.32312, -1.0182],
  [0.21807, 0.32909, -1.2168],
  [0.22511, 0.33439, -1.4512],
  [0.23247, 0.33904, -1.7298],
  [0.24010, 0.34308, -2.0637],
  [0.24792, 0.34655, -2.4681],
  [0.25591, 0.34951, -2.9641],
  [0.26400, 0.35200, -3.5814],
  [0.27218, 0.35407, -4.3633],
  [0.28039, 0.35577, -5.3762],
  [0.28863, 0.35714, -6.7262],
  [0.29685, 0.35823, -8.5955],
  [0.30505, 0.35907, -11.324],
  [0.31320, 0.35968, -15.628],
  [0.32129, 0.36011, -23.325],
  [0.32931, 0.36038, -40.770],
  [0.33724, 0.36051, -116.45]
]

def dot(x, y):
  return np.sum(x * y, axis=-1)


def norm(x):
  return np.sqrt(dot(x, x))


def lerp(a, b, c):
  return (b - a) * c + a


def xyz2cct(xyz):
  """
  Implementation of Robertson's method for converting XYZ to CCT.
  Reference:
  "Color Science: Concepts and Methods, Quantitative Data and Formulae", Second Edition,
  Gunter Wyszecki and W. S. Stiles, John Wiley & Sons, 1982, pp. 227, 228.
  """
  di = 0
  i = 0
  if (xyz[0] < 1.0e-20) and (xyz[1] < 1.0e-20) and (xyz[2] < 1.0e-20):
    return -1  # /* protect against possible divide-by-zero failure */
  us = (4.0 * xyz[0]) / (xyz[0] + 15.0 * xyz[1] + 3.0 * xyz[2])
  vs = (6.0 * xyz[1]) / (xyz[0] + 15.0 * xyz[1] + 3.0 * xyz[2])
  dm = 0.0
  for i in range(31):
    di = (vs - uvt[i][1]) - uvt[i][2] * (us - uvt[i][0])
    if (i > 0) and (((di < 0.0) and (dm >= 0.0)) or ((di >= 0.0) and (dm < 0.0))):
      break  # /* found lines bounding (us, vs) : i-1 and i */
    dm = di

  if i == 31:
    # /* bad XYZ input, color temp would be less than minimum of 1666.7 degrees, or too far towards blue */
    return -1
  di = di / np.sqrt(1.0 + uvt[i][2] * uvt[i][2])
  dm = dm / np.sqrt(1.0 + uvt[i - 1][2] * uvt[i - 1][2])
  p = dm / (dm - di)  # /* p = interpolation parameter, 0.0 : i-1, 1.0 : i */
  p = 1.0 / (lerp(rt[i - 1], rt[i], p))
  cct = p
  return cct  # /* success */


def get_cct_from_exif(light_source):
  """
  EXIF LightSource legal values to standard illuminant:
  https://exiftool.org/TagNames/EXIF.html#LightSource
  Standard illuminant to CCT:
  https://en.wikipedia.org/wiki/Standard_illuminant
  Day White Fluorescent:
  https://onlinemanual.nikonimglib.com/d7500/en/16_white_balance_01.html

  :param light_source: EXIF LightSource legal value
  :return: correlated color temperature
  """
  cct_by_ls = {
    12: 6430,  # Daylight Fluorescent
    13: 5000,  # Day White Fluorescent
    14: 4230,  # Cool White Fluorescent
    15: 3450,  # White Fluorescent
    16: 2940,  # Warm White Fluorescent
    17: 2856,  # Standard Light A
    18: 4874,  # Standard Light B
    19: 6774,  # Standard Light C
    20: 5503,  # D55
    21: 6504,  # D65
    22: 7504,  # D75
    23: 5003,  # D50
  }
  return cct_by_ls[light_source]


def raw_rgb_to_xyz(raw_rgb, temp, xyz2cam1, xyz2cam2, calib_illum_1=CALIB_ILLUM1, calib_illum_2=CALIB_ILLUM2):
  # RawRgbToXyz Convert raw-RGB triplet to corresponding XYZ
  cct1 = get_cct_from_exif(calib_illum_1)
  cct2 = get_cct_from_exif(calib_illum_2)
  cct1inv = 1 / cct1
  cct2inv = 1 / cct2
  temp_inv = 1 / temp
  g = (temp_inv - cct2inv) / (cct1inv - cct2inv)
  h = 1 - g
  xyz2cam = g * xyz2cam1 + h * xyz2cam2
  xyz = np.matmul(np.linalg.inv(xyz2cam), np.transpose(raw_rgb))
  return xyz


def raw_rgb_to_cct(raw_rgb, xyz2cam1, xyz2cam2, calib_illum_1=CALIB_ILLUM1, calib_illum_2=CALIB_ILLUM2):
  """Convert raw-RGB triplet to corresponding correlated color temperature (CCT)"""
  pxyz = [.3, .3, .3]
  loss = 1e10
  k = 1
  cct = 6500  # default
  while loss > 1e-4 and k < 100:
    cct = xyz2cct(pxyz)
    xyz = raw_rgb_to_xyz(raw_rgb, cct, xyz2cam1, xyz2cam2, calib_illum_1, calib_illum_2)
    loss = norm(xyz - pxyz)
    pxyz = xyz
    k = k + 1
  return cct


def interpolate_cst(xyz2cam1: np.ndarray, xyz2cam2: np.ndarray, temp: float, calib_illum_1: Optional[int]=CALIB_ILLUM1,
                    calib_illum_2: Optional[int]=CALIB_ILLUM2) -> np.ndarray:
  """
  Interpolate xyz2cam matrices based on inverse correlated color temperature.
  :param xyz2cam1: derived from ColorMatrix1, corresponding to CalibrationIlluminant1
  :param xyz2cam2: derived from ColorMatrix2, corresponding to CalibrationIlluminant2
  :param temp: correlated color temperature of illuminant
  :param calib_illum_1: CalibrationIlluminant1
  :param calib_illum_2: CalibrationIlluminant2
  :return: interpolated xyz2cam
  """
  # RawRgbToXyz Convert raw-RGB triplet to corresponding XYZ
  cct1 = get_cct_from_exif(calib_illum_1)
  cct2 = get_cct_from_exif(calib_illum_2)

  if cct2 > cct1:
    temp = np.clip(temp, cct1, cct2)
  else:
    temp = np.clip(temp, cct2, cct1)
  cct1inv = 1 / cct1
  cct2inv = 1 / cct2
  tempinv = 1 / temp
  g = (tempinv - cct2inv) / (cct1inv - cct2inv)
  h = 1 - g
  xyz2cam = g * xyz2cam1 + h * xyz2cam2
  return xyz2cam


def compute_ccm(forward_matrix_1: list_or_np_array, forward_matrix_2: list_or_np_array,
                illuminant: Optional[np.ndarray]=None,
                color_matrix_1: Optional[list_or_np_array]=None, color_matrix_2: Optional[list_or_np_array]=None,
                calib_illum_1: Optional[int]=CALIB_ILLUM1, calib_illum_2: Optional[int]=CALIB_ILLUM2) -> np.ndarray:
  """Computes the color correction matrix (CCM) based on a given illuminant."""

  xyz2srgb = np.array(XYZ_TO_SRGB_D50)
  cam2xyz1 = np.reshape(np.asarray(forward_matrix_1), (3, 3))
  cam2xyz2 = np.reshape(np.asarray(forward_matrix_2), (3, 3))
  xyz2cam1 = np.reshape(np.asarray(color_matrix_1), (3, 3))
  xyz2cam2 = np.reshape(np.asarray(color_matrix_2), (3, 3))

  illuminant = illuminant / illuminant[1]

  # interpolate between CSTs based on illuminant
  # Adobe SDK uses ColorMatrices to compute the CCT even when ForwardMatrices are present
  cct = raw_rgb_to_cct(illuminant, xyz2cam1, xyz2cam2, calib_illum_1, calib_illum_2)
  cam2xyz = interpolate_cst(cam2xyz1, cam2xyz2, cct, calib_illum_1, calib_illum_2)
  ccm = xyz2srgb @ cam2xyz
  return ccm


def xyz_to_uv(xyz: Union[Sequence[float], np.ndarray]) -> np.ndarray:
  """XYZ to uv conversion."""
  x, y, z = xyz
  de_nom = (x + 15 * y + 3 * z) + 1e-12
  u = 4 * x / de_nom
  v = 6 * y / de_nom
  return np.array([u, v])


def uv_to_xyz(u: float, v: float, y: Optional[float] = 1.0) -> np.ndarray:
  """Inverse for CIE 1960 UCS (u,v)."""
  v = float(v) if np.isscalar(v) else (v + 1e-12)  # guard
  x = 1.5 * (u / v) * y
  z = ((2.0 - 0.5 * u) / v - 5.0) * y
  return np.array([x, y, z])


def cct_tint_from_raw_rgb(raw_rgb: list_or_np_array, xyz2cam1: np.ndarray, xyz2cam2: np.ndarray,
                          calib_illum_1: Optional[int]=CALIB_ILLUM1, calib_illum_2: Optional[int]=CALIB_ILLUM2
                          ) -> Tuple[Union[float, None], Union[float, None]]:
  """Estimates correlated color temperature (CCT) and tint from a raw-RGB illuminant vector."""
  raw_rgb = raw_rgb / (raw_rgb[1] + 1e-12)
  cct = raw_rgb_to_cct(raw_rgb, xyz2cam1, xyz2cam2, calib_illum_1, calib_illum_2)
  if cct < 0:
    return None, None
  xyz = raw_rgb_to_xyz(raw_rgb, cct, xyz2cam1, xyz2cam2, calib_illum_1, calib_illum_2)
  uv = xyz_to_uv(xyz)
  us, vs = uv
  dm = 0.0
  di = 1.0
  for i in range(31):
    di = (vs - uvt[i][1]) - uvt[i][2] * (us - uvt[i][0])
    if (i > 0) and (((di < 0) and (dm >= 0)) or ((di >= 0) and (dm < 0))):
      break
    dm = di
  di_n = di / np.sqrt(1.0 + uvt[i][2] ** 2)
  dm_n = dm / np.sqrt(1.0 + uvt[i - 1][2] ** 2)
  p = dm_n / (dm_n - di_n)
  u_bb = (1 - p) * uvt[i - 1][0] + p * uvt[i][0]
  v_bb = (1 - p) * uvt[i - 1][1] + p * uvt[i][1]
  du = uvt[i][0] - uvt[i - 1][0]
  dv = uvt[i][1] - uvt[i - 1][1]
  n = np.array([-dv, du]) / (np.linalg.norm([du, dv]) + 1e-12)
  tint = np.dot(uv - np.array([u_bb, v_bb]), n)
  return cct, tint

def raw_rgb_from_cct_tint(cct: float, tint: float, xyz2cam1: np.ndarray, xyz2cam2: np.ndarray,
                          calib_illum_1: Optional[int]=CALIB_ILLUM1, calib_illum_2: Optional[int]=CALIB_ILLUM2
                          ) -> np.ndarray:
  """Reconstructs a camera raw-RGB illuminant vector from correlated color temperature (CCT) and tint."""
  cct = float(cct)
  cct_inv = 1.0 / cct
  xyz2cam = interpolate_cst(xyz2cam1, xyz2cam2, cct, calib_illum_1, calib_illum_2)
  i = 1.0
  for i in range(1, len(rt)):
    if rt[i] >= cct_inv:
      break
  p = (cct_inv - rt[i - 1]) / (rt[i] - rt[i - 1])
  u_bb = (1 - p) * uvt[i - 1][0] + p * uvt[i][0]
  v_bb = (1 - p) * uvt[i - 1][1] + p * uvt[i][1]
  du = uvt[i][0] - uvt[i - 1][0]
  dv = uvt[i][1] - uvt[i - 1][1]
  n = np.array([-dv, du]) / (np.linalg.norm([du, dv]) + EPS)
  uv = np.array([u_bb, v_bb]) + tint * n
  xyz = uv_to_xyz(uv[0], uv[1], y=1.0)
  raw = xyz2cam @ xyz
  raw = raw / (raw[1] + EPS)
  return raw


