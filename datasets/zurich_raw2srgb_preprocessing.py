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

This script applies a set of pre-processing to Zurich raw2sRGB dataset.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

import numpy as np
from utils import img_utils, file_utils
import cv2
from scipy.optimize import minimize
import shutil

# Path to dataset that includes: test and train folders.
dataset_folder = 'path/to/Zurich-RAW-to-DSLR-Dataset'

def remove_black_level(img, black_lv=63, white_lv=4 * 255):
  img = np.maximum(img.astype(np.float32) - black_lv, 0) / (white_lv - black_lv)
  return img

def fit_matrix(params, source, target):
  m = params.reshape(3, 3)
  residuals = target - source @ m.T
  return np.sum(residuals ** 2)


if __name__ == '__main__':
  set_names = ['train', 'test']
  for set_name in set_names:
    print(f'Processing {set_name} ...')
    os.makedirs(os.path.join(dataset_folder, set_name, 'raw_images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, set_name, 'srgb_images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, set_name, 'denoised_raw_images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, set_name, 'data'), exist_ok=True)

    files = [f for f in os.listdir(os.path.join(dataset_folder, set_name, 'huawei_raw')) if f.endswith('.png')]

    for idx, f in enumerate(files):
      print(f'Processing {set_name}: {idx}/{len(files)} ...')
      rgb_img = img_utils.imread(
        os.path.join(dataset_folder, set_name, 'huawei_visualized', f.replace('.png', '.jpg')))
      raw = img_utils.imread(os.path.join(dataset_folder, set_name, 'huawei_raw', f), single_channel=True,
                             normalize=False)
      raw = remove_black_level(raw)
      demosaiced_raw = img_utils.demosaice(raw, 'RGGB')

      # save raw image
      img_utils.imwrite(demosaiced_raw, os.path.join(dataset_folder, set_name, 'raw_images', f), 'PNG-16')

      # move srgb image
      shutil.move(os.path.join(dataset_folder, set_name, 'canon', f.replace('.png', '.jpg')),
                  os.path.join(dataset_folder, set_name, 'srgb_images', f.replace('.png', '.jpg')))
      # reads sRGB image
      srgb = img_utils.imread(os.path.join(dataset_folder, set_name, 'srgb_images',
                                           f.replace('.png', '.jpg')))
      # generates pseudo ground-truth denoised image
      srgb_degamma = srgb ** 2.2
      denoised_raw = img_utils.apply_mapping_func(srgb_degamma,
                                                  img_utils.get_mapping_func(srgb_degamma, demosaiced_raw))
      img_utils.imwrite(denoised_raw, os.path.join(dataset_folder, set_name, 'denoised_raw_images', f), 'PNG-16')

      illum = np.mean(demosaiced_raw.reshape([-1, 3]), axis=0)
      wb_raw = demosaiced_raw @ np.diag(illum[1]/illum)
      rgb_img = cv2.resize(rgb_img, (demosaiced_raw.shape[1], demosaiced_raw.shape[0]))
      source_colors = wb_raw.reshape([-1, 3])
      target_colors = rgb_img.reshape([-1, 3])
      constraints = []

      for j in range(3):
        constraints.append({'type': 'eq', 'fun': lambda params, j=j: np.sum(params[j::3]) - 1})

      bounds = [(0, None) for _ in range(9)]
      initial_guess = np.eye(3).flatten()
      result = minimize(fit_matrix, initial_guess, args=(source_colors, target_colors),
                        bounds=bounds, constraints=constraints, method='SLSQP')
      ccm = result.x.reshape(3, 3)
      data = {'cam_illum': illum,
              'cam_daylight_illum': None,
              'ccm': ccm,
              'orientation': 1}

      # save data
      file_utils.write_json_file(data,
                                 os.path.join(dataset_folder, set_name, 'data', f.replace('.png', '.json')))
