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

This script extracts denoised images from DNGs exported by Adobe Lightroom or Photoshop after applying the AI Denoiser.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

from tifffile import TiffFile
from utils.img_utils import im2double, imwrite

DENOISED_DNG_FOLDER = '/path/to/denoised/DNG/files'
out_folder = '/path/to/save/PNG-16/denoised/images'
os.makedirs(out_folder, exist_ok=True)
files = [f for f in os.listdir(DENOISED_DNG_FOLDER) if f.endswith('.dng')]

for i, f in enumerate(files):
  print(f'{i}/{len(files)}')
  dng = TiffFile(os.path.join(DENOISED_DNG_FOLDER, f))
  denoised = im2double(dng.series[2].keyframe.asarray())
  imwrite(denoised, os.path.join(out_folder, f), 'PNG-16')