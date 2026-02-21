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

This file contains the modular neural ISP pipeline, including the backend functionalities of the photo-editing tool.
"""

import sys
import os
import re
import tempfile

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from photofinishing.photofinishing_model import PhotofinishingModule
from denoising.nafnet_arch import NAFNet
from awb_ccm.c5_model import IllumEstimator as CCIllumEstimator
from awb_ccm.s24_model import IllumEstimator as S24IllumEstimator
from awb_ccm.user_pref_mapping import map_illum, UserPrefIllumEstimator
from io_.raw_jpeg_adapter_model import JPEGAdapter
from io_.linearization import CIEXYZNet
from upsampling.bilateral_guided_upsampling import BGU

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Union, Dict, List, Any, Tuple
import time
import json
from utils.img_utils import (tensor_to_img, img_to_tensor, raw_to_lsrgb, compute_2d_rgbg_histogram, imresize,
                             compute_snr, compute_edges, imencode, imdecode, imread)
from utils.img_utils import apply_exif_orientation, imwrite

from utils.file_utils import read_json_file, write_json_file
from utils.color_utils import compute_ccm, cct_tint_from_raw_rgb, raw_rgb_from_cct_tint
from utils.constants import *
from utils.vect_utils import min_max_normalization


class PipeLine(nn.Module):
  """Modular neural pipeline."""

  _SOBEL_X = torch.tensor([[-1, 0, 1]], dtype=torch.float32).view(1, 1, 1, 3)
  _SOBEL_Y = torch.tensor([[-1], [0], [1]], dtype=torch.float32).view(1, 1, 3, 1)
  
  def __init__(self, denoising_model_path: Optional[str] = None,
               denoising_model_config_path: Optional[str]=None,
               generic_denoising_model_path: Optional[str] = None,
               generic_denoising_model_config_path: Optional[str] = None,
               enhancement_model_path: Optional[str] = None,
               enhancement_model_config_path: Optional[str] = None,
               enhancement_style_model_path: Optional[List[str]] = None,
               enhancement_style_model_config_path: Optional[List[str]] = None,
               photofinishing_model_path: Optional[str] = None,
               photofinishing_model_config_path: Optional[str]=None,
               photofinishing_style_model_paths: Optional[List[str]] = None,
               photofinishing_style_model_config_paths: Optional[List[str]] = None,
               s24_awb_model_path: Optional[str] = None,
               s24_awb_model_config_path: Optional[str]=None, cc_awb_model_path: Optional[str] = None,
               post_awb_model_path: Optional[str] = None, raw_jpeg_adapter_model_path: Optional[str] = None,
               raw_jpeg_adapter_model_config_path: Optional[str]=None,
               linearization_model_path: Optional[str] = None,
               running_device: Optional[torch.device] = torch.device('cuda'), log: Optional[str] = None):
    """
    Pipeline constructor.

    Args:
        denoising_model_path: Optional path to the camera-specific denoising model.
        denoising_model_config_path: Optional path to the config file of the camera-specific denoising model. If not
           provided but 'denoising_model_path' is given, it is assumed that the config is located in a folder named
           'configs' at the same root directory as the model (e.g., if the model is in `./models/model_lite.pth`, the
           config is expected at `./configs/lite.json`).
        generic_denoising_model_path: Optional path to the generic denoising model.
        generic_denoising_model_config_path: Optional path to the generic model's config file. If not provided but the
           generic model path is given, the config is assumed to follow the same convention as above.
        enhancement_model_path: Optional path to the default detail-enhancement model.
        enhancement_model_config_path: Optional path to the config file of the default-style enhancement model. If not
           provided while the model path is given, the config is assumed to be located in a 'configs' folder at the same
           root directory as the model, sharing the same basename (e.g., 'models/model.pth' -> 'configs/model.json').
        enhancement_style_model_path: Optional list of paths to style-specific detail-enhancement models. Each entry
           corresponds to a model used by the photofinishing style models in 'photofinishing_style_model_paths' (see
           below).
        enhancement_style_model_config_path: Optional list of config paths for the style-specific enhancement models.
           If not provided, each config is assumed to follow the same conventions described above for enhancement model.
        photofinishing_model_path: Optional path to the photofinishing model for the default style.
        photofinishing_model_config_path: Optional path to the config file of the photofinishing model. If not provided
           but the model is given, the config is assumed to be in a folder named 'config', located at the same root
           directory as the model. For example, if the model is in '../models/style0.pth', the config is expected in
           '../config/style0.json'.
        photofinishing_style_model_paths: Optional list of paths to photofinishing models corresponding to non-default
           styles.
        photofinishing_style_model_config_paths: Optional list of config paths for the style photofinishing models.
           If omitted, configs follow the same 'config/model.json' convention as for the default photofinishing model.
        s24_awb_model_path: Optional path to the camera-specific (S24 main raw space) illuminant estimation model for
           auto white balance (AWB).
        s24_awb_model_config_path: Optional path to the S24 AWB config file. If not provided but the model path is
           given, the config is assumed to share the same folder and basename (i.e., 'model.pt' -> 'model.json').
        cc_awb_model_path: Optional path to the cross-camera illuminant estimation model.
        post_awb_model_path: Optional path to the post-estimation illuminant-mapping model used for user-preference AWB
           adjustments.
        raw_jpeg_adapter_model_path: Optional path to the Raw-JPEG Adapter model used for embedding raw information into
           the output sRGB JPEG.
        raw_jpeg_adapter_model_config_path: Optional config file for the Raw-JPEG Adapter model. If not provided but the
           model path is given, the config is assumed to be in a 'config' folder with a matching basename.
        linearization_model_path: Optional path to the CIE XYZ Net model for linearizing third-party sRGB inputs.
        running_device: Optional torch device (e.g., 'torch.device("cuda")' or 'torch.device("cpu")').
        log: Optional string container for storing log messages. If not provided and logging is enabled at inference
           time, logs are printed directly to the console.
    Note:
        All model paths can be updated later through the 'update_model(...)' function.
    """
    super().__init__()
    self._denoising_model = None
    self._generic_denoising_model = None
    self._enhancement_model = None
    self._enhancement_style_models = None
    self._photofinishing_model = None
    self._photofinishing_style_models = None
    self._s24_awb_model = None
    self._cc_awb_model = None
    self._post_awb_model = None
    self._raw_jpeg_adapter_model = None
    self._linearization_model = None
    self._is_s24 = None
    self._raw_jpeg_quality = None
    self._log = log
    self._device = running_device
    self._denoising_path = denoising_model_path
    self._denoising_config_path = denoising_model_config_path
    self._generic_denoising_path = generic_denoising_model_path
    self._generic_denoising_config_path = generic_denoising_model_config_path

    self._enhancement_path = enhancement_model_path
    self._enhancement_config_path = enhancement_model_config_path
    self._enhancement_style_paths = enhancement_style_model_path
    self._enhancement_style_config_paths = enhancement_style_model_config_path

    self._photofinishing_path = photofinishing_model_path
    self._photofinishing_config_path = photofinishing_model_config_path
    self._s24_awb_path = s24_awb_model_path
    self._s24_awb_config_path = s24_awb_model_config_path
    self._cc_awb_path = cc_awb_model_path
    self._post_awb_path = post_awb_model_path
    self._raw_jpeg_path = raw_jpeg_adapter_model_path
    self._raw_jpeg_config_path = raw_jpeg_adapter_model_config_path
    self._linearization_path = linearization_model_path
    self._photofinishing_style_paths = photofinishing_style_model_paths
    self._photofinishing_style_config_paths = photofinishing_style_model_config_paths

    self._assert_inputs()

    # Raw denoising models
    if self._denoising_path is not None:
      self._load_denoising_model()

    if self._generic_denoising_path is not None:
      self._load_generic_denoising_model()

    # AWB models
    if self._s24_awb_path is not None:
      self._load_s24_awb_model()

    if self._cc_awb_path is not None:
      self._load_cc_awb_model()

    if self._post_awb_path is not None:
      self._load_post_awb_model()

    # Photofinishing models
    if self._photofinishing_path is not None:
      self._load_photofinishing_model()

    if self._photofinishing_style_paths is not None:
      self._load_photofinishing_style_model()

    # Enhancement
    if self._enhancement_path is not None:
      self._load_enhancement_model()

    if self._enhancement_style_paths is not None:
      self._load_enhancement_style_model()

    # Raw-JPEG adapter model
    if self._raw_jpeg_path is not None:
      self._load_raw_jpeg_model()

    # Linearization
    if self._linearization_path is not None:
      self._load_linearization_model()

    # Upsampling
    self._upsample_model = BGU(reg_lambda=1e-7, reg_T=True)

    self.update_device(running_device)

  def set_log(self, log: str):
    """Sets log."""
    self._log = log

  def get_log(self):
    return self._log

  def _trim_log(self):
    """Keep self._log at max lines by trimming 10 from the top."""
    if self._log is None:
      return
    lines = self._log.splitlines()
    if len(lines) > LOG_MAX_LENGTH:
      lines = lines[-LOG_MAX_LENGTH:]
      self._log = '\n'.join(lines)

  def _assert_inputs(self, wo_assert: Optional[bool]=False):
    """Asserts model and config file paths."""

    def check_model_with_config(model_path: str, c_path: Optional[str] = None
                                ) -> Tuple[Union[str, None], Union[str, None]]:
      if not wo_assert:
        assert os.path.exists(model_path)
      else:
        if not os.path.exists(model_path):
          return None, None
      if c_path is None:
        c_path = model_path.replace('.pth', '.json').replace('.pt', '.json').replace(
          'model-awb', 'config-awb')
        if not os.path.exists(c_path):
          c_path = None
      if c_path is None or not os.path.exists(c_path):
        base_dir, filename = os.path.split(model_path)
        config_folders = ['config', 'configs']
        config_path = None
        for config_folder in config_folders:
          temp_config_path = os.path.join(os.path.dirname(base_dir), config_folder,
                                          os.path.splitext(filename)[0] + '.json')
          if os.path.exists(temp_config_path):
            config_path = temp_config_path
          elif re.search(r'_(base|large|lite)', filename):
            c_f = filename.rsplit('_', 1)[-1].removesuffix('.pth')
            temp_config_path = os.path.join(os.path.dirname(base_dir), config_folder, f'{c_f}.json')
            if os.path.exists(temp_config_path):
              config_path = temp_config_path
      else:
        config_path = c_path
        if not wo_assert:
          assert config_path is not None
        else:
          if config_path is None:
            return None, None
      return model_path, config_path

    if self._denoising_path is not None:
      self._denoising_path, self._denoising_config_path = check_model_with_config(
        self._denoising_path, self._denoising_config_path)

    if self._generic_denoising_path is not None:
      self._generic_denoising_path, self._generic_denoising_config_path = check_model_with_config(
        self._generic_denoising_path, self._generic_denoising_config_path)

    if self._enhancement_path is not None:
      self._enhancement_path, self._enhancement_config_path = check_model_with_config(
        self._enhancement_path, self._enhancement_config_path)


    if self._photofinishing_path is not None:
      self._photofinishing_path, self._photofinishing_config_path = check_model_with_config(
        self._photofinishing_path, self._photofinishing_config_path)

    if self._s24_awb_path is not None:
      self._s24_awb_path, self._s24_awb_config_path = check_model_with_config(
        self._s24_awb_path, self._s24_awb_config_path)

    if self._cc_awb_path is not None:
      assert os.path.exists(self._cc_awb_path)

    if self._post_awb_path is not None:
      assert os.path.exists(self._post_awb_path)

    if self._raw_jpeg_path is not None:
      self._raw_jpeg_path, self._raw_jpeg_config_path = check_model_with_config(
        self._raw_jpeg_path, self._raw_jpeg_config_path)

    if self._linearization_path is not None:
      assert os.path.exists(self._linearization_path)

    if self._photofinishing_style_paths is not None:
      assert isinstance(self._photofinishing_style_paths, list)
    if self._photofinishing_style_config_paths is not None:
      assert isinstance(self._photofinishing_style_config_paths, list)
    elif self._photofinishing_style_paths is not None:
      self._photofinishing_style_config_paths = [None] * len(self._photofinishing_style_paths)
    if self._photofinishing_style_paths is not None:
      assert len(self._photofinishing_style_config_paths) <= len(self._photofinishing_style_paths)
      if len(self._photofinishing_style_config_paths) < len(self._photofinishing_style_paths):
        while len(self._photofinishing_style_config_paths) != len(self._photofinishing_style_paths):
          self._photofinishing_style_config_paths.append(None)

      for i, (photofinishing_style_path, photofinishing_style_config_path) in enumerate(
         zip(self._photofinishing_style_paths, self._photofinishing_style_config_paths)):
        self._photofinishing_style_paths[i], self._photofinishing_style_config_paths[i] = check_model_with_config(
          photofinishing_style_path, photofinishing_style_config_path)

    if self._enhancement_style_paths is not None:
      assert isinstance(self._enhancement_style_paths, list)
    if self._enhancement_style_config_paths is not None:
      assert isinstance(self._enhancement_style_config_paths, list)
    elif self._enhancement_style_paths is not None:
      self._enhancement_style_config_paths = [None] * len(self._enhancement_style_paths)
    if self._enhancement_style_paths is not None:
      assert len(self._enhancement_style_config_paths) <= len(self._enhancement_style_paths)
      if len(self._enhancement_style_config_paths) < len(self._enhancement_style_paths):
        while len(self._enhancement_style_config_paths) != len(self._enhancement_style_paths):
          self._enhancement_style_config_paths.append(None)

      for i, (enhancement_style_path, enhancement_style_config_path) in enumerate(
         zip(self._enhancement_style_paths, self._enhancement_style_config_paths)):
        self._enhancement_style_paths[i], self._enhancement_style_config_paths[i] = check_model_with_config(
          enhancement_style_path, enhancement_style_config_path)

    if self._enhancement_style_paths is not None and self._photofinishing_style_paths is not None:
      assert len(self._enhancement_style_paths) == len(self._photofinishing_style_paths)


  def update_model(self, denoising_model_path: Optional[str] = None,
                   denoising_model_config_path: Optional[str] = None,
                   generic_denoising_model_path: Optional[str] = None,
                   generic_denoising_model_config_path: Optional[str] = None,
                   enhancement_model_path: Optional[str] = None,
                   enhancement_model_config_path: Optional[str] = None,
                   enhancement_style_model_path: Optional[List[str]] = None,
                   enhancement_style_model_config_path: Optional[List[str]] = None,
                   s24_awb_model_path: Optional[str] = None,
                   s24_awb_model_config_path: Optional[str] = None,
                   cc_awb_model_path: Optional[str] = None,
                   post_awb_model_path: Optional[str] = None,
                   photofinishing_model_path: Optional[str] = None,
                   photofinishing_model_config_path: Optional[str] = None,
                   photofinishing_style_model_path: Optional[List[str]] = None,
                   photofinishing_style_model_config_path: Optional[List[str]] = None,
                   raw_jpeg_model_path: Optional[str] = None,
                   raw_jpeg_model_config_path: Optional[str] = None,
                   linearization_model_path: Optional[str] = None,
                   ):

    if denoising_model_path is not None:
      self._denoising_path = denoising_model_path
      self._denoising_config_path = denoising_model_config_path

    if generic_denoising_model_path is not None:
      self._generic_denoising_path = generic_denoising_model_path
      self._generic_denoising_config_path = generic_denoising_model_config_path

    if s24_awb_model_path is not None:
      self._s24_awb_path = s24_awb_model_path
      self._s24_awb_config_path = s24_awb_model_config_path

    if cc_awb_model_path is not None:
      self._cc_awb_path = cc_awb_model_path

    if post_awb_model_path is not None:
      self._post_awb_path = post_awb_model_path

    if photofinishing_model_path is not None:
      self._photofinishing_path = photofinishing_model_path
      self._photofinishing_config_path = photofinishing_model_config_path

    if linearization_model_path is not None:
      self._linearization_path = linearization_model_path

    if enhancement_model_path is not None:
      self._enhancement_path = enhancement_model_path
      self._enhancement_config_path = enhancement_model_config_path

    if enhancement_style_model_path is not None:
      self._enhancement_style_paths = enhancement_style_model_path
      self._enhancement_style_config_paths = enhancement_style_model_config_path

    if raw_jpeg_model_path is not None:
      self._raw_jpeg_path = raw_jpeg_model_path
      self._raw_jpeg_config_path = raw_jpeg_model_config_path

    if photofinishing_style_model_path is not None:
      self._photofinishing_style_paths = photofinishing_style_model_path
      self._photofinishing_style_config_paths = photofinishing_style_model_config_path

    # Assertions
    self._assert_inputs()

    # Denoising model
    if denoising_model_path is not None:
      self._load_denoising_model()

    # Generic denoising model
    if generic_denoising_model_path is not None:
      self._load_generic_denoising_model()

    # AWB models
    if s24_awb_model_path is not None:
      self._load_s24_awb_model()

    if cc_awb_model_path is not None:
      self._load_cc_awb_model()

    if post_awb_model_path:
      self._load_post_awb_model()

    # Photofinishing models
    if photofinishing_model_path is not None:
      self._load_photofinishing_model()

    if photofinishing_style_model_path is not None:
      self._load_photofinishing_style_model()

    # Enhancement model
    if enhancement_model_path is not None:
      self._load_enhancement_model()

    if self._enhancement_style_paths is not None:
      self._load_enhancement_style_model()

    # Raw-JPEG adapter model
    if self._raw_jpeg_path is not None:
      self._load_raw_jpeg_model()

    # Linearization model
    if self._linearization_path is not None:
      self._load_linearization_model()


  def _forward_asserts(self, illum: Union[torch.Tensor, np.ndarray], denoising_strength: float,
                       chroma_denoising_strength: float, luma_denoising_strength: float,
                       enhancement_strength: float, ccm: Union[torch.Tensor, np.ndarray], img_metadata: dict,
                       use_cc_awb: bool, use_generic_denoiser: bool, style_id: int,
                       awb_user_pref: bool, contrast_amount: float, vibrance_amount: float, saturation_amount: float,
                       sharpening_amount: float, highlight_amount: float, shadow_amount: float, target_cct: float,
                       target_tint: float, ev_scale: float, skip_ps_assert: Optional[bool]=False,
                       apply_orientation: Optional[bool]=False, gain_blending_weight: Optional[float]=None,
                       gtm_blending_weight: Optional[float]=None, ltm_blending_weight: Optional[float]=None,
                       chroma_blending_weight: Optional[float]=None, gamma_blending_weight: Optional[float]=None):
    if (illum is None or ccm is None) and img_metadata is None:
      raise ValueError('Expected image metadata since illuminant and/or CCM are missing.')

    if apply_orientation and img_metadata is None:
      raise ValueError('Expected image metadata to apply orientation.')

    if img_metadata is not None:
      self._is_s24 = PipeLine._is_24_camera(img_metadata=img_metadata)
    else:
      self._is_s24 = False  # use generic models if camera is unknown

    if awb_user_pref:
      assert img_metadata is not None
      assert self._post_awb_model is not None

    if self._is_s24:
      assert self._generic_denoising_model is not None or self._denoising_model is not None
      if use_generic_denoiser:
        assert self._generic_denoising_model is not None
    else:
      assert self._generic_denoising_model is not None

    if style_id != 0:
      assert self._photofinishing_style_models is not None and style_id <= len(self._photofinishing_style_models)
      if self._enhancement_style_models is not None:
        assert style_id <= len(self._enhancement_style_models)

    elif not skip_ps_assert:
      assert self._photofinishing_model is not None

    if gain_blending_weight is not None:
      assert 0.0 <= gain_blending_weight <= 1.0

    if gtm_blending_weight is not None:
      assert 0.0 <= gtm_blending_weight <= 1.0

    if ltm_blending_weight is not None:
      assert 0.0 <= ltm_blending_weight <= 1.0

    if chroma_blending_weight is not None:
      assert 0.0 <= chroma_blending_weight <= 1.0

    if gamma_blending_weight is not None:
      assert 0.0 <= gamma_blending_weight <= 1.0


    if denoising_strength is not None:
      assert 0.0 <= denoising_strength <= 1.0

    if chroma_denoising_strength is not None:
      assert 0.0 <= chroma_denoising_strength <= 1.0

    if luma_denoising_strength is not None:
      assert 0.0 <= luma_denoising_strength <= 1.0

    if enhancement_strength is not None:
      assert 0.0 <= enhancement_strength <= 1.0

    if contrast_amount:
      assert -1 <= contrast_amount <= 1.0

    if vibrance_amount:
      assert -1 <= vibrance_amount <= 1.0

    if saturation_amount:
      assert -1 <= saturation_amount <= 1.0

    if sharpening_amount:
      assert 0 < sharpening_amount <= 50.0

    assert EV_MIN <= ev_scale <= EV_MAX

    if highlight_amount:
      assert -1 <= highlight_amount <= 1.0

    if shadow_amount:
      assert -1 <= shadow_amount <= 1.0

    apple_pro_raw = self._is_apple_pro_raw(img_metadata)
    # Apple ProRAW produces unreliable CCT/Tint values; skip validation
    if not apple_pro_raw:
      if target_cct:
        assert CCT_MIN <= target_cct <= CCT_MAX

      if target_tint:
        assert TINT_MIN <= target_tint <= TINT_MAX

    if illum is None:
      if use_cc_awb or not self._is_s24:
        assert self._cc_awb_model is not None
      else:
        assert (self._s24_awb_model is not None or self._cc_awb_model is not None)

    if ccm is None:
      assert img_metadata is not None
      keys = ['forward_matrix1', 'forward_matrix2', 'color_matrix1', 'color_matrix2',
              'calibration_illuminant1', 'calibration_illuminant2']
      for key_i in keys:
        if 'additional_metadata' in img_metadata:
          assert (key_i in img_metadata or key_i in img_metadata['additional_metadata'])
        else:
          assert key_i in img_metadata or key_i


  def _color_correction(
     self, raw: torch.Tensor, denoised_raw: torch.Tensor, illum: torch.Tensor, ccm: torch.Tensor,
     img_metadata: dict, use_cc_awb: bool, log_messages: bool, report_time: bool, awb_user_pref: bool,
     adjust_ccm: bool, target_cct: float, target_tint: float) -> Dict[str, Union[torch.Tensor, float]]:
    """Performs color correction on given raw image to map from raw space to linear sRGB space."""
    if illum is None and (target_cct is None or target_tint is None):
      computed_illum = True
      model_name = 's24' if self._is_s24 and not use_cc_awb else 'cc'
      if self._is_s24 and 'noise_stats' in self._s24_awb_config['capture_data']:
        img_metadata['noise_stats'] = PipeLine._compute_noise_stats(raw, denoised_raw)

      img_stats = self._get_awb_stats(raw_img=self._to_np(raw), model=model_name, metadata=img_metadata)
      if report_time:
        awb_time_start = time.perf_counter()
      else:
        awb_time_start = 0
      if self._is_s24 and not use_cc_awb:
        if log_messages:
          message = 'Calling S24 AWB model...\n'
          self._log_message(message)
        illum = self._s24_awb_model(img_stats['hist_stats'], img_stats['capture_data_stats'])
      else:
        if log_messages:
          message = 'Calling CC AWB model...\n'
          self._log_message(message)

        illum = self._cc_awb_model(img_stats['hist_stats'], inference=True)
      illum = illum.to(dtype=raw.dtype)
      if awb_user_pref and not self._is_apple_pro_raw(img_metadata):
        if 'color_matrix1' in img_metadata:
          with torch.no_grad():
            illum = self._to_tensor(map_illum(self._post_awb_model, self._to_np(illum), img_metadata)
                                    ).to(dtype=raw.dtype)
        else:
          print('Cannot apply AWB preference: missing required metadata.')
      elif awb_user_pref:
        illum = self._to_tensor(img_metadata['as_shot_neutral']).to(dtype=raw.dtype)
      if report_time:
        awb_time_end = time.perf_counter()
        awb_time = awb_time_end - awb_time_start
      else:
        awb_time = None
    else:
      computed_illum = False
      awb_time = 0

    if target_cct is not None or target_tint is not None:
      process_target_cct_tint = True
      if log_messages:
        if target_cct is not None:
          message = f'Adjusting illuminant color to match target CCT {target_cct}'
        else:
          message = 'Adjusting illuminant color to match'
        if target_tint is not None:
          message += f' target tint {target_tint} ...\n'
        else:
          message += ' ...\n'
        self._log_message(message)
      if not ('color_matrix1' in img_metadata and 'color_matrix2' in img_metadata):
        process_target_cct_tint = False
        if log_messages:
          message = 'Cannot process CCT/tint adjustment, because required metadata is missing.'
          self._log_message(message)
    else:
      process_target_cct_tint = False

    if process_target_cct_tint:
      computed_illum = True
      cct, tint = 0, 0
      xyz2cam1 = np.reshape(np.asarray(img_metadata['color_matrix1']), (3, 3))
      xyz2cam2 = np.reshape(np.asarray(img_metadata['color_matrix2']), (3, 3))
      calib_illum_1 = CALIB_ILLUM1 if 'calibration_illuminant1' not in img_metadata else img_metadata[
        'calibration_illuminant1']
      calib_illum_2 = CALIB_ILLUM2 if 'calibration_illuminant2' not in img_metadata else img_metadata[
        'calibration_illuminant2']
      if target_cct is None or target_tint is None:
        if illum is not None and isinstance(illum, torch.Tensor):
          illum = illum.detach().cpu().numpy().astype(np.float32).squeeze()
        cct, tint = cct_tint_from_raw_rgb(illum, xyz2cam1, xyz2cam2, calib_illum_1, calib_illum_2)
        if target_cct is None:
          target_cct = cct
      if target_tint is None:
        target_tint = tint
      else:
        target_tint = target_tint * TINT_SCALE
      target_illum = raw_rgb_from_cct_tint(target_cct, target_tint, xyz2cam1, xyz2cam2, calib_illum_1, calib_illum_2)
      illum = self._to_tensor(target_illum).to(dtype=raw.dtype)
    elif img_metadata is not None and 'color_matrix1' in img_metadata and 'color_matrix2' in img_metadata:
      xyz2cam1 = np.reshape(np.asarray(img_metadata['color_matrix1']), (3, 3))
      xyz2cam2 = np.reshape(np.asarray(img_metadata['color_matrix2']), (3, 3))
      calib_illum_1 = CALIB_ILLUM1 if 'calibration_illuminant1' not in img_metadata else img_metadata[
        'calibration_illuminant1']
      calib_illum_2 = CALIB_ILLUM2 if 'calibration_illuminant2' not in img_metadata else img_metadata[
        'calibration_illuminant2']
      if illum is not None:
        if isinstance(illum, torch.Tensor):
          illum_temp = illum.detach().cpu().numpy().astype(np.float32).squeeze()
        else:
          illum_temp = illum
      target_cct, target_tint = cct_tint_from_raw_rgb(illum_temp, xyz2cam1, xyz2cam2, calib_illum_1, calib_illum_2)
      # Constrain CCT/tint values to prevent unnatural color outputs in failure cases
      if awb_user_pref and not self._is_apple_pro_raw(img_metadata):
        cct_min, cct_max = 2500, 8000
      else:
        cct_min, cct_max = 1500, 9000
      if target_cct < cct_min or target_cct > cct_max and not self._is_apple_pro_raw(img_metadata):
        target_cct = max(min(target_cct, cct_max), cct_min)
      if abs(target_tint / TINT_SCALE) > 5 and awb_user_pref and not self._is_apple_pro_raw(img_metadata):
        target_tint = 5 * TINT_SCALE
      illum = raw_rgb_from_cct_tint(target_cct, target_tint, xyz2cam1, xyz2cam2, calib_illum_1, calib_illum_2)
      illum = self._to_tensor(illum.astype(np.float32)).to(dtype=raw.dtype, device=raw.device)
    else:
      target_cct, target_tint = None, None

    if ccm is None:
      if computed_illum or adjust_ccm:
        required_keys = [
          'forward_matrix1', 'forward_matrix2', 'color_matrix1', 'color_matrix2',
          'calibration_illuminant1', 'calibration_illuminant2']
        has_all = all(img_metadata.get(k) is not None for k in required_keys)
        if has_all:
          ccm = compute_ccm(
            forward_matrix_1=img_metadata['forward_matrix1'],
            forward_matrix_2=img_metadata['forward_matrix2'],
            illuminant=self._to_np(illum) if isinstance(illum, torch.Tensor) else illum,
            color_matrix_1=img_metadata['color_matrix1'],
            color_matrix_2=img_metadata['color_matrix2'],
            calib_illum_1=img_metadata['calibration_illuminant1'],
            calib_illum_2=img_metadata['calibration_illuminant2'],
          )
        elif 'color_matrix' in img_metadata:
          # Fall back to a fixed CCM if provided
          ccm = np.array(img_metadata['color_matrix'], dtype=np.float32)
        else:
          raise ValueError('Cannot compute/find CCM: missing metadata fields.')
      else:
        ccm = np.array(img_metadata['color_matrix'], dtype=np.float32)

      ccm = self._to_tensor(ccm).to(dtype=raw.dtype)

    if report_time:
      raw_to_lsrgb_time_start = time.perf_counter()
    else:
      raw_to_lsrgb_time_start = 0
    lsrgb = raw_to_lsrgb(denoised_raw, illum_color=illum, ccm=ccm)
    if report_time:
      raw_to_lsrgb_time_end = time.perf_counter()
      raw_to_lsrgb_time = raw_to_lsrgb_time_end - raw_to_lsrgb_time_start
    else:
      raw_to_lsrgb_time = None

    return {'lsrgb': lsrgb, 'awb_time': awb_time, 'raw_to_lsrgb_time': raw_to_lsrgb_time, 'cct': target_cct,
            'tint': target_tint if target_tint is None else target_tint / TINT_SCALE, 'illum': illum, 'ccm': ccm}


  def get_ps_params(self, lsrgb: torch.Tensor, style_id: Optional[int] = 0, downscale_ps: Optional[bool]=True,
                    post_process_ltm: Optional[bool]=False, solver_iter: Optional[int]=BILATERAL_SOLVER_ITERS,
                    contrast_amount: Optional[float]=0.0, vibrance_amount: Optional[float]=0.0,
                    saturation_amount: Optional[float]=0.0, highlight_amount: Optional[float]=0.0,
                    shadow_amount: Optional[float]=0.0, gain_param: Optional[torch.Tensor]=None,
                    gtm_param: Optional[torch.Tensor]=None, ltm_param: Optional[torch.Tensor] = None,
                    chroma_lut_param: Optional[torch.Tensor]=None, gamma_param: Optional[torch.Tensor] = None
                    ) -> Dict[str, torch.Tensor]:
    """Gets photofinishing parameters."""
    if downscale_ps:
      lsrgb_ds = F.interpolate(lsrgb, scale_factor=0.25, mode='bilinear', align_corners=True)
    else:
      lsrgb_ds = lsrgb

    if style_id:
      call_func = self._photofinishing_style_models[style_id - 1]
    else:
      call_func = self._photofinishing_model
    ps_output = call_func(
      lsrgb_ds, post_process_ltm=post_process_ltm, solver_iter=solver_iter,
      contrast_amount=contrast_amount, vibrance_amount=vibrance_amount, saturation_amount=saturation_amount,
      highlight_amount=highlight_amount, shadow_amount=shadow_amount, input_gain_factor=gain_param,
      input_gtm_params=gtm_param, input_ltm_params=ltm_param, input_chroma_lut=chroma_lut_param,
      input_gamma_factor=gamma_param, return_params=True
    )

    return {'gain_param': ps_output['pred_gain'], 'gtm_param': ps_output['pred_gtm'],
            'ltm_param': ps_output['pred_ltm'], 'chroma_lut_param': ps_output['pred_lut'],
            'gamma_param': ps_output['pred_gamma']}


  def forward(self, raw: Union[torch.Tensor, np.ndarray], denoised_raw: Optional[torch.Tensor]=None,
              lsrgb: Optional[torch.Tensor]=None, illum: Optional[Union[torch.Tensor, np.ndarray]] = None,
              denoising_strength: Optional[float] = None, chroma_denoising_strength: Optional[float] = None,
              luma_denoising_strength: Optional[float] = None, enhancement_strength: Optional[float] = None,
              ccm: Optional[Union[torch.Tensor, np.ndarray]] = None,
              img_metadata: Optional[dict] = None, use_cc_awb: Optional[bool]=False,
              use_generic_denoiser: Optional[bool]=False,
              style_id: Optional[int] = 0, downscale_ps: Optional[bool]=True, log_messages: Optional[bool]=False,
              report_time: Optional[bool]=False, post_process_ltm: Optional[bool]=False,
              awb_user_pref: Optional[bool]=False, adjust_ccm: Optional[bool]=False,
              solver_iter: Optional[int]=BILATERAL_SOLVER_ITERS, auto_exposure: Optional[bool]=False,
              contrast_amount: Optional[float]=0.0, vibrance_amount: Optional[float]=0.0,
              saturation_amount: Optional[float]=0.0, sharpening_amount: Optional[float]=0.0,
              highlight_amount: Optional[float]=0.0, shadow_amount: Optional[float]=0.0,
              target_cct: Optional[float]=None, target_tint: Optional[float]=None,
              gain_param: Optional[torch.Tensor]=None, gtm_param: Optional[torch.Tensor]=None,
              ltm_param: Optional[torch.Tensor] = None, chroma_lut_param: Optional[torch.Tensor]=None,
              gamma_param: Optional[torch.Tensor] = None, ev_scale: Optional[float] = 0.0,
              return_intermediate: Optional[bool]=False, apply_orientation: Optional[bool]=False,
              always_return_np: Optional[bool]=False, photofinishing: Optional[bool]=True,
              gain_blending_weight: Optional[float] = None, gtm_blending_weight: Optional[float] = None,
              ltm_blending_weight: Optional[float] = None, chroma_blending_weight: Optional[float] = None,
              gamma_blending_weight: Optional[float] = None
              ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:

    """
    Forward function of the modular neural ISP pipeline.

    Args:
        raw: Input raw image (torch.Tensor or numpy.ndarray).
        denoised_raw: Optional pre-computed denoised raw image (torch.Tensor). Useful for the interactive tool to avoid
           re-running denoising.
        lsrgb: Optional linear sRGB image (torch.Tensor) to bypass re-computation.
        illum: Optional illuminant vector (torch.Tensor or numpy.ndarray) used for white balancing.
        denoising_strength: Optional denoising strength in [0, 1]; 0 disables denoising, 1 applies full denoising.
        chroma_denoising_strength: Optional chroma denoising strength in [0, 1].
        luma_denoising_strength: Optional luma denoising strength in [0, 1].
        enhancement_strength: Optional detail-enhancement strength in [0, 1]; 0 disables detail enhancement, 1 applies
           full detail enhancement.
        ccm: Optional 3x3 color correction matrix (torch.Tensor or numpy.ndarray).
        img_metadata: Optional dictionary of image metadata (e.g., from DNG files). Required if neither 'illum' nor
           'ccm' are provided.
        use_cc_awb: Flag indicating whether to use the cross-camera AWB model for S24 images when illuminant or
           'target_cct'/'target_tint' are not given.
        use_generic_denoiser: Flag to use the generic denoiser for S24 images instead of the camera-specific model.
        style_id: Style identifier for photofinishing; 0 is the default style, and higher values index into
           the style of models in 'self._photofinishing_style_models'.
        downscale_ps: If True, downscales the image to 1/4 resolution before photofinishing (default True).
        log_messages: If True, logs processing steps.
        report_time: If True, reports processing time for each module.
        post_process_ltm: If True, enables multi-scale and refinement of the LTM coefficients to mitigate halo artifacts
          (Sec. B.1, supplementary).
        awb_user_pref: Applies user-preference AWB mapping when neither 'illum' nor manual 'target_cct'/'target_tint'
           are provided.
        adjust_ccm: Recomputes the CCM based on pre-calibrated matrices in 'img_metadata'.
        solver_iter: Number of bilateral-solver iterations used for LTM post-processing.
        auto_exposure: If True, applies automatic exposure adjustment.
        contrast_amount: Contrast adjustment amount in [-1, 1].
        vibrance_amount: Vibrance adjustment amount in [-1, 1].
        saturation_amount: Saturation adjustment amount in [-1, 1].
        sharpening_amount: Sharpening strength in [0, 50]; 0 disables sharpening.
        highlight_amount: Highlight adjustment amount in [-1, 1].
        shadow_amount: Shadow adjustment amount in [-1, 1].
        target_cct: Target correlated color temperature (Kelvin) for manual white balance. Range: [CCT_MIN, CCT_MAX]
           (see utils.constants.py).
        target_tint: Target tint for manual white balance. Range: [TINT_MIN, TINT_MAX] (see utils.constants.py).
        gain_param: Optional pre-computed digital gain (torch.Tensor).
        gtm_param: Optional pre-computed global tone-mapping coefficients (torch.Tensor).
        ltm_param: Optional pre-computed local tone-mapping (LTM) coefficients (torch.Tensor).
        chroma_lut_param: Optional pre-computed 2D chroma-mapping LuT (torch.Tensor).
        gamma_param: Optional pre-computed gamma parameter (torch.Tensor).
        ev_scale: Manual exposure adjustment. Range: [EV_MIN, EV_MAX] (see utils.constants.py).
        return_intermediate: If True, returns intermediate stage outputs.
        apply_orientation: If True, applies orientation in img_metadata.
        always_return_np: If True, returns outputs as numpy arrays.
        photofinishing: If True, runs the photofinishing module (default True).
        gain_blending_weight: Blending weight for digital gain in [0, 1].
        gtm_blending_weight: Blending weight for GTM in [0, 1].
        ltm_blending_weight: Blending weight for LTM in [0, 1].
        chroma_blending_weight: Blending weight for chroma mapping in [0, 1].
        gamma_blending_weight: Blending weight for gamma correction in [0, 1].

    Returns:
        A dictionary containing:
          - final output image,
          - intermediate metadata,
          - raw and denoised raw inputs,
          - linear sRGB image,
          - estimated/given CCT, tint, and illuminant,
          - color correction matrix (CCM),
          - photofinishing parameters (None if photofinishing is False),
          - optional intermediate stage outputs.
    """


    output = {}

    if img_metadata is not None:
      if 'additional_metadata' in img_metadata:
        for key in img_metadata['additional_metadata']:
          img_metadata.update({key: img_metadata['additional_metadata'][key]})

    output['metadata'] = img_metadata

    if (gain_param is not None and gtm_param is not None and ltm_param is not None and chroma_lut_param is not None
       and gamma_param is not None):
      skip_ps_assert = True
    else:
      skip_ps_assert = False

    self._forward_asserts(
      illum=illum, denoising_strength=denoising_strength, chroma_denoising_strength=chroma_denoising_strength,
      luma_denoising_strength=luma_denoising_strength, enhancement_strength=enhancement_strength, ccm=ccm,
      img_metadata=img_metadata, use_cc_awb=use_cc_awb, use_generic_denoiser=use_generic_denoiser,
      style_id=style_id, awb_user_pref=awb_user_pref, contrast_amount=contrast_amount,
      vibrance_amount=vibrance_amount, saturation_amount=saturation_amount, sharpening_amount=sharpening_amount,
      highlight_amount=highlight_amount, shadow_amount=shadow_amount, target_cct=target_cct, target_tint=target_tint,
      ev_scale=ev_scale, skip_ps_assert=skip_ps_assert, apply_orientation=apply_orientation,
      gain_blending_weight=gain_blending_weight, gtm_blending_weight=gtm_blending_weight,
      ltm_blending_weight=ltm_blending_weight, chroma_blending_weight=chroma_blending_weight,
      gamma_blending_weight=gamma_blending_weight)


    if isinstance(raw, np.ndarray):
      np_input = True
      raw = self._to_tensor(raw)
    else:
      np_input = False

    if always_return_np:
      np_input = True

    output['raw'] = raw

    if ev_scale and denoised_raw is None and lsrgb is None:
      raw = raw * (2 ** ev_scale)

    if lsrgb is None and illum is not None and (isinstance(illum, np.ndarray) or isinstance(illum, list)):
      if isinstance(illum, list):
        illum = np.array(illum, dtype=np.float32)
      illum = self._to_tensor(illum)

    if lsrgb is None and ccm is not None and (isinstance(ccm, np.ndarray) or isinstance(ccm, list)):
      if isinstance(ccm, list):
        ccm = np.array(ccm, dtype=np.float32)
      ccm = self._to_tensor(ccm)

    # raw denoising
    if denoised_raw is None and lsrgb is None:
      if report_time:
        denoising_time_start = time.perf_counter()
      else:
        denoising_time_start = 0
      if self._is_s24 and not use_generic_denoiser:
        if log_messages:
          message = 'Calling S24 raw denoising model...\n'
          self._log_message(message)
        denoised_raw = self._denoising_model(raw)
      else:
        if log_messages:
          message = 'Calling generic raw denoising model...\n'
          self._log_message(message)
        denoised_raw = self._generic_denoising_model(raw)
      output['denoised_raw'] = denoised_raw
    else:
      denoising_time_start = None
      output['denoised_raw'] = denoised_raw
    if denoising_strength is not None:
      denoised_raw = (1 - denoising_strength) * raw + denoising_strength * denoised_raw
    if report_time:
      if denoising_time_start is not None:
        denoising_time_end = time.perf_counter()
        denoising_time = denoising_time_end - denoising_time_start
      else:
        denoising_time = 0
    else:
      denoising_time = None

    # color correction
    if lsrgb is None:
      color_correction_output = self._color_correction(
        raw=raw, denoised_raw=denoised_raw, illum=illum, ccm=ccm, img_metadata=img_metadata, use_cc_awb=use_cc_awb,
        log_messages=log_messages, report_time=report_time, awb_user_pref=awb_user_pref, adjust_ccm=adjust_ccm,
        target_cct=target_cct, target_tint=target_tint)
      lsrgb = color_correction_output['lsrgb']
      awb_time = color_correction_output['awb_time']
      raw_to_lsrgb_time = color_correction_output['raw_to_lsrgb_time']
      output['cct'] = color_correction_output['cct']
      output['tint'] = color_correction_output['tint']
      output['illum'] = color_correction_output['illum']
      output['ccm'] = color_correction_output['ccm']
      output['lsrgb'] = lsrgb
      if log_messages:
        message = f'(CCT, tint) = ({output["cct"]}, {output["tint"]})\n'
        self._log_message(message)
    else:
      awb_time = 0
      raw_to_lsrgb_time = 0
      output['cct'] = None
      output['tint'] = None
      output['illum'] = None
      output['ccm'] = None
      output['lsrgb'] = lsrgb
    # post-capture auto-exposure:
    if auto_exposure:
      if log_messages:
        message = 'Auto exposure...\n'
        self._log_message(message)
      lsrgb, ev = self._auto_exposure(lsrgb)
      if log_messages:
        message = f'Applied EV = {ev.cpu().numpy().squeeze()}\n'
        self._log_message(message)
      output['ev'] = ev
    else:
      output['ev'] = torch.zeros(lsrgb.shape[0], device=self._device, dtype=lsrgb.dtype)

    # luma/chroma denoising
    if (chroma_denoising_strength is not None and chroma_denoising_strength > 0) or (
       luma_denoising_strength is not None and luma_denoising_strength > 0):
      if log_messages:
        message = 'Luma/chroma denoising...\n'
        self._log_message(message)
      denoised_ycbcr = PipeLine._chroma_luma_denoise(self._photofinishing_model.rgb_to_ycbcr(lsrgb),
                                                     chroma_strength=chroma_denoising_strength,
                                                     luma_strength=luma_denoising_strength)
      lsrgb = self._photofinishing_model.ycbcr_to_rgb(denoised_ycbcr)


    # photofinishing
    if photofinishing:
      if log_messages:
        message = 'Calling photofinishing model...\n'
        self._log_message(message)

      if downscale_ps:
        lsrgb_ds = F.interpolate(lsrgb, scale_factor=0.25, mode='bilinear', align_corners=True)
      else:
        lsrgb_ds = lsrgb
      if report_time:
        ps_time_start = time.perf_counter()
      else:
        ps_time_start = 0
      if style_id > 0:
        ps_output = self._photofinishing_style_models[style_id - 1](
          lsrgb_ds, report_time=report_time, post_process_ltm=post_process_ltm, solver_iter=solver_iter,
          contrast_amount=contrast_amount, vibrance_amount=vibrance_amount, saturation_amount=saturation_amount,
          highlight_amount=highlight_amount, shadow_amount=shadow_amount,
          input_gain_factor=gain_param, input_gtm_params=gtm_param, input_ltm_params=ltm_param,
          input_chroma_lut=chroma_lut_param, input_gamma_factor=gamma_param, return_params=True,
          return_intermediate=return_intermediate, gain_blending_weight=gain_blending_weight,
          gtm_blending_weight=gtm_blending_weight, ltm_blending_weight=ltm_blending_weight,
          gamma_blending_weight=gamma_blending_weight)
      else:
        ps_output = self._photofinishing_model(
          lsrgb_ds, report_time=report_time, post_process_ltm=post_process_ltm, solver_iter=solver_iter,
          contrast_amount=contrast_amount, vibrance_amount=vibrance_amount, saturation_amount=saturation_amount,
          highlight_amount=highlight_amount, shadow_amount=shadow_amount,
          input_gain_factor=gain_param, input_gtm_params=gtm_param, input_ltm_params=ltm_param,
          input_chroma_lut=chroma_lut_param, input_gamma_factor=gamma_param, return_params=True,
          return_intermediate=return_intermediate, gain_blending_weight=gain_blending_weight,
          gtm_blending_weight=gtm_blending_weight, ltm_blending_weight=ltm_blending_weight,
          gamma_blending_weight=gamma_blending_weight)
      srgb_ds = ps_output['output']
      output['gain_param'] = ps_output['pred_gain']
      output['gtm_param'] = ps_output['pred_gtm']
      output['ltm_param'] = ps_output['pred_ltm']
      output['chroma_lut_param'] = ps_output['pred_lut']
      output['gamma_param'] = ps_output['pred_gamma']
      if return_intermediate:
        output['lsrgb_gain'] = ps_output['lsrgb_gain']
        output['lsrgb_gtm'] = ps_output['lsrgb_gtm']
        output['lsrgb_ltm'] = ps_output['lsrgb_ltm']
        output['processed_lsrgb'] = ps_output['processed_lsrgb']
        output['gamma'] = ps_output['output']

      if report_time:
        ps_detailed_time = f"Gain time: {ps_output['gain_time']}\n"
        ps_detailed_time += f"Global tone mapping time: {ps_output['gtm_time']}\n"
        ps_detailed_time += f"Local tone mapping time: {ps_output['ltm_time']}\n"
        ps_detailed_time += f"Chroma mapping time: {ps_output['chroma_mapping_time']}\n"
        ps_detailed_time += f"Gamma correction time: {ps_output['gamma_correction_time']}\n"
        ps_time_end = time.perf_counter()
        ps_time = ps_time_end - ps_time_start
      else:
        ps_detailed_time = None
        ps_time = None

      if downscale_ps:
        if log_messages:
          message = 'Upsampling...\n'
          self._log_message(message)

        if report_time:
          upsampling_time_start = time.perf_counter()
        else:
          upsampling_time_start = 0
        with torch.no_grad():
          srgb = self._upsample_model(lsrgb, lsrgb_ds, srgb_ds)
        if report_time:
          upsampling_time_end = time.perf_counter()
          upsampling_time = upsampling_time_end - upsampling_time_start
        else:
          upsampling_time = None
      else:
        srgb = srgb_ds
        upsampling_time = None
    else:
      upsampling_time = None
      ps_detailed_time = None
      ps_time = None
      srgb = lsrgb
      output['gain_param'] = None
      output['gtm_param'] = None
      output['ltm_param'] = None
      output['chroma_lut_param'] = None
      output['gamma_param'] = None
      if return_intermediate:
        output['lsrgb_gain'] = None
        output['lsrgb_gtm'] = None
        output['lsrgb_ltm'] = None
        output['processed_lsrgb'] = None


    # Enhancement
    if self._enhancement_model is not None:
      if log_messages:
        message = 'Calling enhancement model...\n'
        self._log_message(message)

      if report_time:
        enhancement_time_start = time.perf_counter()
      else:
        enhancement_time_start = 0

      if style_id > 0 and self._enhancement_style_models is not None:
        enhanced_srgb = self._enhancement_style_models[style_id - 1](srgb).clamp(0, 1)
      else:
        enhanced_srgb = self._enhancement_model(srgb).clamp(0, 1)

      if enhancement_strength is not None:
       enhanced_srgb = (1 - enhancement_strength) * srgb + enhancement_strength * enhanced_srgb
      if report_time:
        enhancement_time_end = time.perf_counter()
        enhancement_time = enhancement_time_end - enhancement_time_start
      else:
        enhancement_time = None

    else:
      enhanced_srgb = srgb
      enhancement_time = 0

    # Sharpening
    if sharpening_amount:
      enhanced_srgb = self._sharpen(enhanced_srgb, amount=sharpening_amount)

    if np_input:
      enhanced_srgb = self._to_np(enhanced_srgb)
    if apply_orientation:
      enhanced_srgb = apply_exif_orientation(enhanced_srgb, img_metadata['orientation'])
    output['srgb'] = enhanced_srgb

    if log_messages and report_time:
      message = f'Denoising time: {denoising_time}\n'
      message += f'AWB time: {awb_time}\n'
      message += f'Raw-to-LsRGB time: {raw_to_lsrgb_time}\n'
      message += f'Photofinishing total time: {ps_time}\n'
      message += f'Photofinishing detailed time:\n{ps_detailed_time}'
      if downscale_ps:
        message += f'Upsampling time: {upsampling_time}\n'
      if self._enhancement_model is not None:
        message += f'Enhancement time: {enhancement_time}\n'
      self._log_message(message)


    return output

  def _log_message(self, message: str):
    if self._log is None:
      print(message)
    else:
      self._log += message + '\n'
      self._trim_log()

  def update_ltm_input_sz(self, input_size: int):
    """Updates input size for LTM model."""
    self._photofinishing_model.update_ltm_input_sz(input_size=input_size)

  def update_device(self, running_device: torch.device):
    """Update device."""
    if str(running_device).startswith('cuda') and not torch.cuda.is_available():
      message = 'CUDA is not available. Using CPU instead.'
      self._log_message(message)
      self._device = torch.device('cpu')
    else:
      self._device = torch.device(running_device)

    if self._denoising_model is not None:
      self._denoising_model.to(device=self._device)
    if self._generic_denoising_model is not None:
      self._generic_denoising_model.to(device=self._device)
    if self._photofinishing_model is not None:
      self._photofinishing_model.update_device(running_device=self._device)
    if self._s24_awb_model is not None:
      self._s24_awb_model.to(device=self._device)
    if self._cc_awb_model is not None:
      self._cc_awb_model.to(device=self._device)
    if self._post_awb_model is not None:
      self._post_awb_model.to(device=self._device)
    if self._photofinishing_style_models is not None:
      for i in range(len(self._photofinishing_style_models)):
        self._photofinishing_style_models[i].update_device(running_device=self._device)
    self._upsample_model.to(device=self._device)
    if self._enhancement_model is not None:
      self._enhancement_model.to(device=self._device)
    if self._enhancement_style_models is not None:
      for i in range(len(self._enhancement_style_models)):
        self._enhancement_style_models[i].to(device=self._device)
    if self._raw_jpeg_adapter_model is not None:
      self._raw_jpeg_adapter_model.to(device=self._device)
    if self._linearization_model is not None:
      self._linearization_model.to(device=self._device)

  def _load_denoising_model(self):
    """Loads raw denoising model."""
    if not os.path.exists(self._denoising_config_path):
      return False
    if not os.path.exists(self._denoising_path):
      return False
    config = read_json_file(self._denoising_config_path)
    self._denoising_model = NAFNet(width=config['width'], middle_block_num=config['middle_block_num'],
                                   encoder_block_nums=config['encoder_block_nums'],
                                   decoder_block_nums=config['decoder_block_nums']).to(self._device)

    self._denoising_model.load_state_dict(torch.load(self._denoising_path, map_location=self._device,
                                                     weights_only=True))
    message = 'Raw denoising model parameters:\n'
    message += f'Total number of network params: {sum(p.numel() for p in self._denoising_model.parameters())}\n'
    self._log_message(message)
    self._denoising_model.eval()

  def _load_generic_denoising_model(self):
    """Loads generic raw denoising model."""
    if not os.path.exists(self._generic_denoising_config_path):
      return False
    if not os.path.exists(self._generic_denoising_path):
      return False
    config = read_json_file(self._generic_denoising_config_path)
    self._generic_denoising_model = NAFNet(width=config['width'], middle_block_num=config['middle_block_num'],
                                           encoder_block_nums=config['encoder_block_nums'],
                                           decoder_block_nums=config['decoder_block_nums']).to(self._device)

    self._generic_denoising_model.load_state_dict(
      torch.load(self._generic_denoising_path, map_location=self._device, weights_only=True))
    message = 'Generic raw denoising model parameters:\n'
    message += (f'Total number of network params: '
                f'{sum(p.numel() for p in self._generic_denoising_model.parameters())}\n')
    self._log_message(message)
    self._generic_denoising_model.eval()


  def _load_enhancement_model(self):
    """Loads enhancement model."""
    if not os.path.exists(self._enhancement_config_path):
      return False
    if not os.path.exists(self._enhancement_path):
      return False
    config = read_json_file(self._enhancement_config_path)
    self._enhancement_model = NAFNet(width=config['width'], middle_block_num=config['middle_block_num'],
                                     encoder_block_nums=config['encoder_block_nums'],
                                     decoder_block_nums=config['decoder_block_nums']).to(self._device)

    self._enhancement_model.load_state_dict(torch.load(self._enhancement_path, map_location=self._device,
                                                       weights_only=True))
    message = 'Enhancement model parameters:\n'
    message += f'Total number of network params: {sum(p.numel() for p in self._enhancement_model.parameters())}\n'
    self._log_message(message)
    self._enhancement_model.eval()


  def _load_photofinishing_model(self):
    """Loads photofinishing model."""
    if not os.path.exists(self._photofinishing_config_path):
      return False
    if not os.path.exists(self._photofinishing_path):
      return False
    config = read_json_file(self._photofinishing_config_path)
    self._photofinishing_model = PhotofinishingModule(device=self._device, use_3d_lut=config['use_3d_lut'])
    self._photofinishing_model.load_state_dict(torch.load(self._photofinishing_path, map_location=self._device,
                                                          weights_only=True))
    message = 'Photofinishing model parameters:\n'
    message += self._photofinishing_model.print_num_of_params(show_message=False)
    self._log_message(message)
    self._photofinishing_model.eval()

  def _load_enhancement_style_model(self):
    """Loads style enhancement models."""
    self._enhancement_style_models = nn.ModuleList()
    for model_path, config_path in zip(self._enhancement_style_paths, self._enhancement_style_config_paths):
      if not os.path.exists(config_path):
        return False
      if not os.path.exists(model_path):
        return False
      config = read_json_file(config_path)
      model = NAFNet(width=config['width'], middle_block_num=config['middle_block_num'],
                     encoder_block_nums=config['encoder_block_nums'],
                     decoder_block_nums=config['decoder_block_nums']).to(self._device)

      model.load_state_dict(torch.load(model_path, map_location=self._device, weights_only=True))
      self._enhancement_style_models.append(model)

    for i in range(len(self._photofinishing_style_models)):
      message = f'Photofinishing style model parameters (for style #{i}):\n'
      message += self._photofinishing_style_models[i].print_num_of_params(show_message=False)
      self._log_message(message)
      self._photofinishing_style_models[i].eval()

  def _load_photofinishing_style_model(self):
    """Loads style photofinishing models."""
    self._photofinishing_style_models = nn.ModuleList()
    for model_path, config_path in zip(self._photofinishing_style_paths, self._photofinishing_style_config_paths):
      if not os.path.exists(config_path):
        return False
      if not os.path.exists(model_path):
        return False
      config = read_json_file(config_path)
      model = PhotofinishingModule(device=self._device, use_3d_lut=config['use_3d_lut'])
      model.load_state_dict(torch.load(model_path, map_location=self._device, weights_only=True))
      self._photofinishing_style_models.append(model)

    for i in range(len(self._photofinishing_style_models)):
      message = f'Photofinishing style model parameters (for style #{i}):\n'
      message += self._photofinishing_style_models[i].print_num_of_params(show_message=False)
      self._log_message(message)
      self._photofinishing_style_models[i].eval()

  def _load_raw_jpeg_model(self):
    """Loads Raw-JPEG Adapter model."""
    if not os.path.exists(self._raw_jpeg_config_path):
      return False
    if not os.path.exists(self._raw_jpeg_path):
      return False
    config = read_json_file(self._raw_jpeg_config_path)
    self._raw_jpeg_adapter_model = JPEGAdapter(
      latent_dim=config['latent_dim'], target_img_size=config['map_size'], use_eca=config['eca'],
      quality=config['quality'], use_scale_dct=config['scale_dct'], use_gamma=config['gamma'], use_lut=config['lut'],
      lut_size=config['lut_size'], lut_channels=config['lut_channels']).to(device=self._device)
    self._raw_jpeg_adapter_model.load_state_dict(torch.load(self._raw_jpeg_path, map_location=self._device,
                                                            weights_only=True))
    self._raw_jpeg_quality = config['quality']
    message = 'Raw JPEG Adapter model parameters:\n'
    message += self._raw_jpeg_adapter_model.print_num_of_params(show_message=False)
    self._log_message(message)
    self._raw_jpeg_adapter_model.eval()

  def _load_linearization_model(self):
    """Loads sRGB linearization model."""
    if not os.path.exists(self._linearization_path):
      return False

    self._linearization_model = CIEXYZNet().to(device=self._device)
    self._linearization_model.load_state_dict(torch.load(self._linearization_path, map_location=self._device,
                                                         weights_only=True))
    message = 'Linearization model parameters: '
    message += f'{self._linearization_model.get_num_of_params()}\n'
    self._log_message(message)
    self._linearization_model.eval()

  def _load_s24_awb_model(self):
    """Loads S24 main camera's AWB model."""
    if not os.path.exists(self._s24_awb_config_path):
      return False
    if not os.path.exists(self._s24_awb_path):
      return False

    s24_awb_model_config = read_json_file(self._s24_awb_config_path)
    capture_data_size = AWB_CAPTURE_DATA_SIZE[AWB_S24_MODELS.index(
      os.path.splitext(os.path.basename(self._s24_awb_path))[0].replace('model-', ''))]
    self._s24_awb_model = S24IllumEstimator(in_channels=capture_data_size, hist_channels=4).to(device=self._device)
    self._s24_awb_model.load_state_dict(torch.load(self._s24_awb_path, weights_only=True, map_location=self._device))
    self._s24_awb_model.eval()
    message = 'S24 AWB model parameters:\n'
    message += self._s24_awb_model.print_num_of_params(show_message=False) + '\n'
    self._log_message(message)
    capture_data = s24_awb_model_config['capture_data']
    norm_values = s24_awb_model_config['norm_values']
    capture_data_min_vector = np.array(norm_values['capture-data-min'])
    capture_data_max_vector = np.array(norm_values['capture-data-max'])
    histogram_boundaries = s24_awb_model_config['hist_boundaries']
    hist_bins = s24_awb_model_config['hist_bins']
    target_size = s24_awb_model_config['target_size']
    self._s24_awb_config = {'capture_data': capture_data,
                            'norm_min': capture_data_min_vector, 'norm_max': capture_data_max_vector,
                            'hist_bounds': histogram_boundaries, 'hist_bins': hist_bins, 'target_size': target_size}


  def _load_cc_awb_model(self):
    """Loads cross-camera AWB model."""
    if not os.path.exists(self._cc_awb_path):
      return False
    self._cc_awb_model = CCIllumEstimator(device=self._device)
    self._cc_awb_model.load_state_dict(torch.load(self._cc_awb_path, weights_only=True, map_location=self._device))
    self._cc_awb_model.eval()
    message = 'CC AWB model parameters:\n'
    message += self._cc_awb_model.print_num_of_params(show_message=False)
    self._log_message(message)
    self._cc_awb_config = {'hist_bins': 48, 'target_size': [256, 384]}

  def _load_post_awb_model(self):
    """Loads preference-aware post AWB model."""
    if not os.path.exists(self._post_awb_path):
      return False
    self._post_awb_model = UserPrefIllumEstimator(in_channels=3)
    self._post_awb_model.load_state_dict(torch.load(self._post_awb_path, weights_only=True, map_location=self._device))
    message = 'Post AWB model parameters:\n'
    message += self._post_awb_model.print_num_of_params(show_message=False)
    self._log_message(message)
    self._post_awb_model.eval()

  def _get_lut_size(self) -> int:
    """Returns CbCr LuT size (num of bins)."""
    return self._photofinishing_model.get_cbcr_lut_size()

  @staticmethod
  def is_s24_camera(metadata: Dict[str, Any]) -> bool:
    """Checks if the input image is from S24 main camera."""
    return PipeLine._is_24_camera(metadata)

  @staticmethod
  def _is_apple_pro_raw(metadata: Dict[str, Any]) -> bool:
    """Heuristic check for Apple ProRaw images."""
    return (metadata.get('as_shot_neutral') is not None and
            np.allclose(metadata.get('as_shot_neutral'), [1.0, 1.0, 1.0]))
    
  @staticmethod
  def _is_24_camera(img_metadata: Dict[str, Any]) -> bool:
    """Checks if the input image is from S24 main camera."""
    def check_camera_specification(in_metadata: Dict[str, Any]) -> bool:
      if (in_metadata['make'] == 'samsung' and in_metadata['model'] == 'Galaxy S24 Ultra' and
         in_metadata['f_number'] == 1.7 and in_metadata['focal_length'] == 6.3):
        return True
      return False
    if 'make' in img_metadata:
      s24_model = check_camera_specification(img_metadata)
    elif 'additional_metadata' in img_metadata:
      img_metadata = img_metadata['additional_metadata']
      s24_model = check_camera_specification(img_metadata)
    else:
      s24_model = False
    return s24_model

  @staticmethod
  def _compute_noise_stats(raw: torch.Tensor, denoised_raw: torch.Tensor) -> np.ndarray:
    """Computes noise stats."""
    assert raw.shape[0] == 1, 'Batch size must be 1'
    diff = (raw - denoised_raw).abs()
    rgb_mean = diff.mean(dim=(0, 2, 3)).squeeze(0).detach().cpu().numpy()
    rgb_std = diff.std(dim=(0, 2, 3)).squeeze(0).detach().cpu().numpy()
    return np.concatenate([rgb_mean, rgb_std], axis=0)

  def _to_tensor(self, x):
    """Converts numpy to tensor."""
    x = x.astype(np.float32)
    if len(x.shape) == 3:
      return img_to_tensor(x).unsqueeze(0).to(device=self._device, dtype=torch.float32)
    else:
      return torch.from_numpy(x).unsqueeze(0).to(device=self._device)

  def save_image(self, srgb: Union[torch.Tensor, np.ndarray], output_path: str,
                 raw: Optional[Union[torch.Tensor, np.ndarray]]=None,
                 srgb_jpeg_quality: Optional[int]=DEFAULT_SRGB_JPEG_QUALITY,
                 raw_jpeg_quality: Optional[int]=DEFAULT_RAW_JPEG_QUALITY, metadata: Optional[Dict]=None,
                 log_messages: Optional[bool]=False, report_time: Optional[bool]=False,
                 editing_settings: Optional[Dict]=None):
    """Saves an sRGB JPEG while embedding processed raw and metadata for re-rendering."""
    assert raw_jpeg_quality in RAW_JPEG_QUALITY_OPTIONS, f'Invalid value of raw JPEG quality: {raw_jpeg_quality}'
    assert 10 <= srgb_jpeg_quality <= 100
    if isinstance(srgb, torch.Tensor):
      srgb = self._to_np(srgb)
    if raw is not None:
      if raw_jpeg_quality != self._raw_jpeg_quality or self._raw_jpeg_adapter_model is None:
        if os.path.exists(os.path.join('..', 'io_', 'models', JPEG_RAW_MODELS[raw_jpeg_quality])):
          new_path = os.path.join('..', 'io_', 'models', JPEG_RAW_MODELS[raw_jpeg_quality])
        elif os.path.exists(os.path.join('io_', 'models', JPEG_RAW_MODELS[raw_jpeg_quality])):
          new_path = os.path.join('io_', 'models', JPEG_RAW_MODELS[raw_jpeg_quality])
        else:
          new_path = None
        if new_path is not None:
          current_model = self._raw_jpeg_adapter_model
          current_raw_jpeg_quality = self._raw_jpeg_quality
          self._raw_jpeg_path = new_path
          self._assert_inputs()
          if self._raw_jpeg_adapter_model is None:
            self._raw_jpeg_adapter_model = current_model
            self._raw_jpeg_quality = current_raw_jpeg_quality

      if isinstance(raw, np.ndarray):
        raw = self._to_tensor(raw)

      if self._raw_jpeg_adapter_model is None:
        raise ValueError('Raw JPEG Adapter model is not loaded.')
      if log_messages:
        message = 'Calling Raw-JPEG Adapter model...\n'
        self._log_message(message)

      if report_time:
        raw_encoding_start = time.perf_counter()
      else:
        raw_encoding_start = 0
      raw_img_proc, operator_params = self._raw_jpeg_adapter_model(raw)
      raw_img_proc = self._to_np(raw_img_proc)
      comment = self._raw_jpeg_adapter_model.encode_params(operator_params)
      with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_name = tmp.name
        imwrite(image=raw_img_proc, output_path=tmp_name, format='jpg', quality=raw_jpeg_quality, comment=comment)
      with open(tmp_name, 'rb') as f:
        raw_jpeg_bytes = f.read()
        if log_messages:
          message = f'Additional raw JPEG size = {len(raw_jpeg_bytes)} bytes.\n'
          self._log_message(message)
      os.remove(tmp_name)
      has_raw = 1
      if report_time:
        raw_encoding_time = time.perf_counter() - raw_encoding_start
      else:
        raw_encoding_time = 0
      if log_messages:
        message = 'Raw image was successfully encoded.\n'
        if report_time:
          message += f'Raw encoding time: {raw_encoding_time}\n'
        self._log_message(message)
    else:
      raw_jpeg_bytes = b''
      comment = 'none'
      has_raw = 0

    has_meta = 1 if metadata is not None else 0
    has_edit = 1 if editing_settings is not None else 0
    meta_bytes = b''
    edit_bytes = b''

    if has_meta:
        if log_messages:
            self._log_message('Encoding metadata...\n')
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_name = tmp.name
        write_json_file(metadata, tmp_name)
        with open(os.path.splitext(tmp_name)[0] + '.json', 'rb') as f:
            meta_bytes = f.read()
        os.remove(os.path.splitext(tmp_name)[0] + '.json')
        if log_messages:
            self._log_message('Metadata was successfully encoded.\n')

    if has_edit:
        if log_messages:
            self._log_message('Encoding editing settings...\n')
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_name = tmp.name
        write_json_file(editing_settings, tmp_name)
        with open(os.path.splitext(tmp_name)[0] + '.json', 'rb') as f:
            edit_bytes = f.read()
        os.remove(os.path.splitext(tmp_name)[0] + '.json')
        if log_messages:
            self._log_message('Editing settings were successfully encoded.\n')

    srgb_buf = imencode(srgb, quality=srgb_jpeg_quality)
    srgb_bytes = srgb_buf.tobytes()
    header = f'{has_raw}\n{has_meta}\n{comment}\n'.encode('utf-8')
    payload = header + raw_jpeg_bytes
    if has_meta:
      payload += b'\nMETA\n' + meta_bytes
    if has_edit:
        payload += b'\nEDIT\n' + edit_bytes
    if srgb_bytes[-2:] != b'\xFF\xD9':
        raise ValueError('Not a valid JPEG (no EOI marker found).')

    out_bytes = srgb_bytes[:-2] + b'\xFF\xD9' + payload

    with open(output_path, 'wb') as f:
        f.write(out_bytes)

    if log_messages:
        self._log_message(f'Image successfully saved with payload ({len(payload)} bytes).')

  def read_image(self, path: str, log_messages: Optional[bool]=False, report_time: Optional[bool]=False,
                  linearize: Optional[bool]=False) -> Dict[str, Any]:
    """Reads an sRGB JPEG, and if payload is appended after EOI, extract raw and metadata."""
    return self._read_image(path=path, log_messages=log_messages, report_time=report_time, linearize=linearize)

  def _read_image(self, path: str, log_messages: Optional[bool] = False, report_time: Optional[bool] = False,
                  linearize: Optional[bool] = False) -> Dict[str, Any]:
    """Reads an sRGB JPEG, and if payload is appended after EOI, extract raw, metadata, and editing settings."""
    srgb_img = imread(path)
    h, w = srgb_img.shape[:2]
    max_dim = max(h, w)
    if max_dim < MIN_DIM:
      scale = MIN_DIM / max_dim
      new_w = int(round(w * scale))
      new_h = int(round(h * scale))
      srgb_img = imresize(srgb_img, height=new_h, width=new_w)
    with open(path, 'rb') as f:
      data = f.read()

    idx = data.find(b'\xFF\xD9')
    if idx == -1:
      raise ValueError('Not a valid JPEG (no EOI marker found).')


    if idx + 2 == len(data) or linearize:
      if self._linearization_model is not None:
        if log_messages:
          self._log_message('\nLinearizing input sRGB...\n')
        with torch.no_grad():
          linearized_img = self._linearization_model(
            self._to_tensor(srgb_img * 0.85).to(device=self._device, dtype=torch.float32))
          linearized_img = self._to_np(linearized_img) * 0.7
          raw_img = ((linearized_img.reshape([-1, 3]) @ np.array(D50_INV_RAW_TO_XYZ).T) @ np.diag(
            np.array(D50_RAW_ILLUM))).reshape(linearized_img.shape).astype(np.float32)
          raw_img = np.clip(raw_img, 0.0, 1.0)
        metadata = METADATA
        editing_settings = None
      else:
        raw_img = None
        metadata = None
        editing_settings = None
      return {'srgb': srgb_img, 'raw': raw_img, 'metadata': metadata, 'editing_settings': editing_settings}


    payload = data[idx + 2:]

    newline_positions = [i for i, b in enumerate(payload) if b == 10]

    if len(newline_positions) < 3:
      if log_messages:
        self._log_message('\nMalformed header (less than 3 lines). Retrying with linearization.\n')
      return self._read_image(path=path, log_messages=log_messages, report_time=report_time, linearize=True)

    header_end = newline_positions[2] + 1

    header_bytes = payload[:header_end]
    try:
      header = header_bytes.decode('utf-8').strip().split('\n')
    except UnicodeDecodeError:
      if log_messages:
        self._log_message('\nHeader decoding failed (non-UTF8 bytes). Retrying with linearization.\n')
      return self._read_image(path=path, log_messages=log_messages, report_time=report_time, linearize=True)

    try:
      has_raw = bool(int(header[0])) if len(header) > 0 else False
      has_meta = bool(int(header[1])) if len(header) > 1 else False
      comment = header[2] if len(header) > 2 else 'none'
    except (ValueError, IndexError):
      if log_messages:
        self._log_message('\nHeader values malformed. Retrying with linearization.\n')
      return self._read_image(path=path, log_messages=log_messages, report_time=report_time, linearize=True)

    body = payload[header_end:]


    raw_img = None
    metadata = None
    editing_settings = None

    if has_raw:
      if self._raw_jpeg_adapter_model is None:
        raise ValueError('Raw JPEG Adapter model is not loaded.')

      if has_meta:
        if log_messages:
          self._log_message('\nDecoding metadata...\n')

        raw_jpeg_bytes, rest = body.split(b'\nMETA\n', 1)
        if b'\nEDIT\n' in rest:
          meta_part, edit_part = rest.split(b'\nEDIT\n', 1)
          metadata = json.loads(meta_part.decode('utf-8'))
          editing_settings = json.loads(edit_part.decode('utf-8'))
        else:
          metadata = json.loads(rest.decode('utf-8'))

        if log_messages:
          self._log_message('Metadata was successfully loaded.\n')
      else:
        raw_jpeg_bytes = body

      if log_messages:
        self._log_message('\nDecoding raw image...\n')
      raw_arr = np.frombuffer(raw_jpeg_bytes, np.uint8)
      proc_raw_img = imdecode(raw_arr)

      raw_decoding_start = time.perf_counter() if report_time else 0
      if log_messages:
        self._log_message('Post-processing with Raw-JPEG Adapter...\n')
      comment_params = self._raw_jpeg_adapter_model.decode_params(comment, device=self._device)
      rec_raw_img = self._raw_jpeg_adapter_model.reconstruct_raw(
        raw_jpeg=img_to_tensor(proc_raw_img).to(device=self._device, dtype=torch.float32).unsqueeze(0),
        gamma_map=comment_params['gamma_map'],
        scale_dct=comment_params['scale_dct'],
        lut=comment_params['lut'])
      raw_img = self._to_np(rec_raw_img)
      if log_messages:
        msg = 'Original raw image was successfully restored.\n'
        if report_time:
          msg += f'Raw restoration time: {time.perf_counter() - raw_decoding_start}\n'
        self._log_message(msg)

    elif has_meta:
      if body.startswith(b'META\n'):
        rest = body[5:]
        if b'\nEDIT\n' in rest:
          meta_part, edit_part = rest.split(b'\nEDIT\n', 1)
          metadata = json.loads(meta_part.decode('utf-8'))
          editing_settings = json.loads(edit_part.decode('utf-8'))
        else:
          metadata = json.loads(rest.decode('utf-8'))
    return {'srgb': srgb_img, 'raw': raw_img, 'metadata': metadata, 'editing_settings': editing_settings}

  @staticmethod
  def _rgb_to_luma(img):
    return 0.2126 * img[:, 0:1] + 0.7152 * img[:, 1:2] + 0.0722 * img[:, 2:3]

  @staticmethod
  def _box_blur(x: torch.Tensor, r: int) -> torch.Tensor:
    """Fast separable box mean filter."""
    if r < 1:
      return x
    k = 2 * r + 1
    ch = x.shape[1]
    w = torch.ones((ch, 1, k, 1), dtype=x.dtype, device=x.device) / k
    x = F.conv2d(x, w, padding=(r, 0), groups=ch)
    w = torch.ones((ch, 1, 1, k), dtype=x.dtype, device=x.device) / k
    return F.conv2d(x, w, padding=(0, r), groups=ch)

  @staticmethod
  def _guided_edge_preserve(y: torch.Tensor, radius: Optional[int] = 4, eps: Optional[float] = 1e-3) -> torch.Tensor:
    """Simplified guided-like filter for luminance."""
    if radius < 1:
      return y
    n = PipeLine._box_blur(torch.ones_like(y), radius) + EPS
    mean_y = PipeLine._box_blur(y, radius) / n
    mean_yy = PipeLine._box_blur(y * y, radius) / n
    var_y = mean_yy - mean_y * mean_y

    a = var_y / (var_y + eps)
    b = mean_y * (1 - a)

    mean_a = PipeLine._box_blur(a, radius) / n
    mean_b = PipeLine._box_blur(b, radius) / n
    return mean_a * y + mean_b

  @staticmethod
  def _chroma_luma_denoise(ycbcr: torch.Tensor, luma_strength: float, chroma_strength: float) -> torch.Tensor:
    """Denoise image with luma/chroma control.
    Args:
      ycbcr: (1, 3, H, W) YCbCr image.
      luma_strength: scalar in [0,1], controls luma denoising.
      chroma_strength: scalar in [0,1], controls chroma denoising.
    Returns:
      Denoised YCbCr tensor (1, 3, H, W).
    """
    assert ycbcr.ndim == 4 and ycbcr.shape[1] == 3
    y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
    # Luma denoising parameters
    r = int(2 + 10 * luma_strength)
    eps = (0.001 + 0.03 * luma_strength) ** 2

    # Chroma denoising parameters
    base_sigma = 3.0 + 12.0 * chroma_strength
    chroma_sigma = base_sigma * (ycbcr.shape[2] / 3000.0) ** 0.9
    kernel_size = int(2 * round(3 * chroma_sigma) + 1)

    # Luma denoising (edge-preserving)
    if luma_strength > 0:
      y_d = PipeLine._guided_edge_preserve(y, radius=r, eps=eps)
      y_out = (1 - luma_strength) * y + luma_strength * y_d
    else:
      y_out = y

    # Chroma denoising (Gaussian blur)
    if chroma_strength > 0:
      cb_d = PipeLine._gaussian_blur(cb, kernel_size=kernel_size, sigma=chroma_sigma)
      cr_d = PipeLine._gaussian_blur(cr, kernel_size=kernel_size, sigma=chroma_sigma)
      if chroma_strength > 0.4:
        cb_d = PipeLine._gaussian_blur(cb_d, kernel_size=kernel_size, sigma=chroma_sigma)
        cr_d = PipeLine._gaussian_blur(cr_d, kernel_size=kernel_size, sigma=chroma_sigma)
      mix = chroma_strength ** 1.5
      cb_out = (1 - mix) * cb + mix * cb_d
      cr_out = (1 - mix) * cr + mix * cr_d
    else:
      cb_out = cb
      cr_out = cr

    ycbcr_out = torch.cat([y_out, cb_out, cr_out], dim=1)
    return ycbcr_out


  @staticmethod
  def _gaussian_blur(img: torch.Tensor,
                     kernel_size: Optional[int] = 5,
                     sigma: Optional[float] = 1.0) -> torch.Tensor:
    """Fast separable Gaussian blur."""
    if sigma <= 0 or kernel_size < 3:
      return img

    kernel_size = int(kernel_size) | 1
    radius = kernel_size // 2

    # 1D gaussian kernel
    coords = torch.arange(-radius, radius + 1, device=img.device, dtype=img.dtype)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    # Reshaping for horizontal/vertical passes
    g_col = g.view(1, 1, -1, 1)
    g_row = g.view(1, 1, 1, -1)

    # Applies horizontal blur
    img = F.conv2d(img, g_row.expand(img.shape[1], 1, 1, -1),
                   padding=(0, radius), groups=img.shape[1])

    # Applies vertical blur
    img = F.conv2d(img, g_col.expand(img.shape[1], 1, -1, 1),
                   padding=(radius, 0), groups=img.shape[1])

    return img

  @staticmethod
  def _sharpen(img: torch.Tensor, amount, radius: Optional[int]=3, sigma: Optional[float]=1.0) -> torch.Tensor:
    """Edge-aware sharpening

    Args:
      img: (1,3,H,W), in [0,1]
      amount: strength of sharpening
      radius: blur kernel size
      sigma: blur sigma
    """
    c = img.shape[1]
    base = PipeLine._gaussian_blur(img, kernel_size=radius, sigma=sigma)
    detail = img - base
    kx = PipeLine._SOBEL_X.to(img.device, img.dtype).repeat(c, 1, 1, 1)
    ky = PipeLine._SOBEL_Y.to(img.device, img.dtype).repeat(c, 1, 1, 1)
    grad_x = F.conv2d(img, kx, padding=(0, 1), groups=c)
    grad_y = F.conv2d(img, ky, padding=(1, 0), groups=c)
    edge_mag = torch.sqrt(grad_x.mul(grad_x).add_(grad_y.mul(grad_y))).mean(1, keepdim=True)
    edge_mask = edge_mag / (edge_mag.amax(dim=(2, 3), keepdim=True) + EPS)
    edge_mask.clamp_(0, 1)
    out = img + amount * detail * edge_mask
    return out.clamp_(0, 1)


  @staticmethod
  def _auto_exposure(x: torch.Tensor,
                     target_gray: Optional[float] = AUTO_EXPOSURE_TARGET_GRAY,
                     target_sigma: Optional[float] = 0.05,
                     bins: Optional[int] = AUTO_EXPOSURE_HISTOGRAM_BINS,
                     max_ev: Optional[float] = AUTO_EXPOSURE_MAX_VALUE,
                     stats_size: Optional[int] = AUTO_EXPOSURE_INPUT_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Advanced auto exposure using histogram matching in linear sRGB.

    Args:
        x: input tensor, (B,3,H,W) or (3,H,W), values in [0,1].
        target_gray: center of target histogram.
        target_sigma: spread of target histogram (default 0.05).
        bins: number of histogram bins.
        max_ev: maximum absolute EV shift allowed.
        stats_size: spatial resolution for exposure stats.

    Returns:
        adjusted image (same shape as input), predicted EV per image (B,).
    """
    if x.dim() == 3:
      x = x.unsqueeze(0)
    b, _, h, w = x.shape
    device, dtype = x.device, x.dtype

    x_stats = F.interpolate(x, size=(stats_size, stats_size), mode="bilinear")

    y = PipeLine._rgb_to_luma(x_stats).clamp(EPS, 1.0).view(b, -1)

    bin_edges = torch.linspace(0, 1, bins, device=device, dtype=dtype)
    target_hist = torch.exp(-0.5 * ((bin_edges - target_gray) / target_sigma) ** 2)
    #print(target_hist)
    target_hist = target_hist / target_hist.sum()

    ev_out, out_imgs = [], []
    for i in range(b):
      yi = y[i]

      # Searches EV shifts
      ev_candidates = torch.linspace(-max_ev, max_ev, 49, device=device, dtype=dtype)
      scores = []
      for ev_c in ev_candidates:
        yi_scaled = (yi * (2.0 ** ev_c)).clamp(0, 1)
        hist = torch.histc(yi_scaled, bins=bins, min=0.0, max=1.0)
        hist = hist / (hist.sum() + EPS)
        scores.append(((hist - target_hist) ** 2).sum())
      scores = torch.stack(scores)

      # Picks EV with the best histogram match
      best_idx = torch.argmin(scores)
      ev_sel = ev_candidates[best_idx]
      ev_out.append(ev_sel)

      scale = 2.0 ** ev_sel
      out_imgs.append((x[i] * scale).clamp(0, 1))

    return torch.stack(out_imgs), torch.stack(ev_out)

  @staticmethod
  def to_np(x):
    """Converts tensor back to np."""
    return PipeLine._to_np(x)

  @staticmethod
  def _to_np(x):
    """Converts tensor back to np."""
    if len(x.shape) == 4:
      return tensor_to_img(x).astype(np.float32)
    else:
      return x.detach().cpu().numpy().squeeze().astype(np.float32)

  def _get_awb_stats(self, raw_img: np.ndarray, model: str, metadata:Optional[dict] = None
                     ) -> Dict[str, torch.Tensor]:
    """Generates AWB stats."""
    if model == 's24':
      # Creates input stats: histogram & capture feature
      img_stats = imresize(img=raw_img, height=self._s24_awb_config['target_size'][0],
                           width=self._s24_awb_config['target_size'][1])
      hist_stats = compute_2d_rgbg_histogram(img=img_stats, hist_bins=self._s24_awb_config['hist_bins'],
                                             hist_boundaries=self._s24_awb_config['hist_bounds'],
                                             edge_hist=True, uv_coord=True)
      capture_feature_stats = np.array([])
      for capture_data_feature in self._s24_awb_config['capture_data']:
        if capture_data_feature == 'iso' or capture_data_feature == 'shutter_speed':
          data_i = np.log(metadata[capture_data_feature])
        elif capture_data_feature == 'noise_stats':
          data_i = metadata[capture_data_feature]
        elif capture_data_feature == 'snr_stats':
          snr_img = compute_snr(raw_img)
          rgb_mean = np.mean(snr_img, axis=(0, 1))
          rgb_std = np.std(snr_img, axis=(0, 1))
          data_i = np.concatenate([rgb_mean.flatten(), rgb_std.flatten()], axis=0)
        else:
          raise NotImplementedError('Invalid capture feature.')
        capture_feature_stats = np.concatenate([capture_feature_stats, data_i.flatten()], axis=0)
      capture_feature_stats = min_max_normalization(capture_feature_stats, self._s24_awb_config['norm_min'],
                                                    self._s24_awb_config['norm_max'])
      return {'capture_data_stats': self._to_tensor(capture_feature_stats), 'hist_stats': self._to_tensor(hist_stats)}


    elif model == 'cc':
      # Creates histogram stats of: uv chroma, edges' uv chroma, and uv coords
      img_stats = imresize(img=raw_img, height=self._cc_awb_config['target_size'][0],
                           width=self._cc_awb_config['target_size'][1])
      hist_stats = np.zeros((self._cc_awb_config['hist_bins'], self._cc_awb_config['hist_bins'], 4))
      valid_chroma_rgb, valid_colors_rgb = self._cc_awb_model.get_hist_colors(img_stats)
      hist_stats[..., 0] = self._cc_awb_model.compute_histogram(valid_chroma_rgb, rgb=valid_colors_rgb)
      edge_img_stats = compute_edges(img_stats)
      valid_chroma_edges, valid_colors_edges = self._cc_awb_model.get_hist_colors(edge_img_stats)
      hist_stats[..., 1] = self._cc_awb_model.compute_histogram(valid_chroma_edges, rgb=valid_colors_edges)
      hist_stats[..., 2], hist_stats[:, :, 3] = self._cc_awb_model.get_uv_coords()
      return {'hist_stats': self._to_tensor(hist_stats)}
    else:
      raise ValueError(f'Unsupported model: {model}.')











