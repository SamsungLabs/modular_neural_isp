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

This script applies a set of pre-processing to Adobe 5K dataset. To download expert C images, use
"download_expertc_adobe5k.py". This script splits the dataset into training, testing, and validation sets and saves
demosaiced raw images and DNG metadata.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

import shutil
import numpy as np
from utils import file_utils, img_utils

# Path to the "raw_photos" of Adobe 5K dataset.
src_directory = 'path/to/dng_photos/in/Adobe5K'
# Path to jpg images of expert C.
expert_c_jpg_directory = 'path/to/expertC_jpg_photos/in/Adobe5K'
# Path to the directory where the processed data will be saved
dest_directory = 'path/to/dataset_after_preparation'


def organize_files_by_extension(src_dir, dest_dng):
  for root, _, files in os.walk(src_dir):
    for file in files:
      if file.lower().endswith('.dng'):
        src_file = os.path.join(root, file)
        dest_file = os.path.join(dest_dng, file)
        shutil.move(src_file, dest_file)
        print(f"Moved: {src_file} -> {dest_file}")


def undo_orientation(img, orientation):
  """Undo the orientation applied to an image."""
  if orientation == 2:
    img = np.flip(img, axis=1)  # Undo horizontal flip
  elif orientation == 3:
    img = np.rot90(img, 2)  # Undo 180-degree rotation
  elif orientation == 4:
    img = np.flip(img, axis=0)  # Undo vertical flip
  elif orientation == 5:
    img = np.flip(img, axis=1)  # Undo horizontal flip
    img = np.rot90(img, 1)  # Undo 90-degree clockwise rotation
  elif orientation == 6:
    img = np.rot90(img, 1)  # Undo 90-degree clockwise rotation
  elif orientation == 7:
    img = np.flip(img, axis=1)  # Undo horizontal flip
    img = np.rot90(img, -1)  # Undo 90-degree counterclockwise rotation
  elif orientation == 8:
    img = np.rot90(img, -1)  # Undo 90-degree counterclockwise rotation
  return img


if __name__ == '__main__':
  dest_jpg_directory = os.path.join(dest_directory, 'srgb')
  dest_dng_directory = os.path.join(dest_directory, 'dngs')
  os.makedirs(dest_jpg_directory, exist_ok=True)
  os.makedirs(dest_dng_directory, exist_ok=True)

  organize_files_by_extension(src_directory, dest_dng_directory)
  jpg_files = [f for f in os.listdir(expert_c_jpg_directory) if f.endswith('.jpg')]
  for f in jpg_files:
    shutil.move(os.path.join(expert_c_jpg_directory, f), os.path.join(dest_jpg_directory, f))

  set_names = ['testing', 'validation', 'training']

  for set_name in set_names:
     filenames = file_utils.read_json_file(os.path.join('adobe_5k_splits', set_name + '.json'))['filenames']
     if set_name == 'testing':
        set_name = 'test'
     elif set_name == 'training':
        set_name = 'train'
     elif set_name == 'validation':
        set_name = 'val'
     set_directory = os.path.join(dest_directory, set_name)
     os.makedirs(set_directory, exist_ok=True)
     os.makedirs(os.path.join(set_directory, 'srgb_images'), exist_ok=True)
     os.makedirs(os.path.join(set_directory, 'raw_images'), exist_ok=True)
     os.makedirs(os.path.join(set_directory, 'data'), exist_ok=True)
     os.makedirs(os.path.join(set_directory, 'dngs'), exist_ok=True)

     for i, f in enumerate(filenames):
        source_jpg = os.path.join(dest_jpg_directory, f + '.jpg')
        target_jpg = os.path.join(set_directory, 'srgb_images', f + '.jpg')
        source_dng = os.path.join(dest_dng_directory, f + '.dng')
        target_dng = os.path.join(set_directory, 'dngs', f + '.dng')
        shutil.move(source_jpg, target_jpg)
        shutil.move(source_dng, target_dng)

  for set_name in ['train', 'test', 'val']:
    print(f'Processing {set_name} ...')
    curr_folder = os.path.join(dest_directory, set_name)
    filenames = [f for f in os.listdir(os.path.join(curr_folder, 'dngs')) if f.endswith('.dng')]
    for i, f in enumerate(filenames):
      print(f'Processing {set_name}: {i}/{len(filenames)}')
      dng_filepath = os.path.join(curr_folder, 'dngs', f)
      dng_data = img_utils.extract_raw_metadata(dng_filepath)
      try:
        dng_data.update(img_utils.extract_additional_dng_metadata(dng_filepath))
        additional_metadata = True
      except:
        additional_metadata = False

      raw = img_utils.extract_image_from_dng(dng_filepath)
      raw = img_utils.normalize_raw(raw, dng_data['black_level'], dng_data['white_level'])
      try:
        demosacied_raw = img_utils.demosaice(raw, dng_data['pattern'])
      except:
        continue

      srgb_img = img_utils.imread(os.path.join(curr_folder, 'srgb_images', f.replace('.dng', '.jpg')))

      if dng_data['orientation'] != 1:
        srgb_img = undo_orientation(srgb_img, dng_data['orientation'])
        img_utils.imwrite(srgb_img,
                          os.path.join(curr_folder, 'srgb_images', f.replace('.dng', '')), 'JPG')

      raw_h, raw_w, _ = demosacied_raw.shape
      srgb_h, srgb_w, _ = srgb_img.shape
      if raw_h < srgb_h or raw_w < srgb_w:
        print(f'Error: RAW image dimensions smaller than sRGB for {f}. Skipping.')
        continue
      if raw_h != srgb_h or raw_w != srgb_w:
        start_h = (raw_h - srgb_h) // 2
        start_w = (raw_w - srgb_w) // 2
        demosacied_raw = demosacied_raw[start_h:start_h + srgb_h, start_w:start_w + srgb_w, :]

      img_utils.imwrite(demosacied_raw, os.path.join(curr_folder, 'raw_images',
                                                     f.replace('.dng', '')), 'PNG-16')

      data = {'cam_illum': dng_data['illum_color'],
              'cam_daylight_illum': dng_data['daylight_illum_color'],
              'ccm': dng_data['color_matrix'],
              'orientation': dng_data['orientation']}
      if additional_metadata:
        data.update({'additional_metadata': {
          'make': dng_data['make'], 'model': dng_data['model'], 'exposure_time': dng_data['exposure_time'],
          'f_number': dng_data['f_number'], 'iso': dng_data['iso'], 'focal_length': dng_data['focal_length'],
          'color_matrix1': dng_data['color_matrix1'], 'color_matrix2': dng_data['color_matrix2'],
          'camera_calibration1': dng_data['camera_calibration1'],
          'camera_calibration2': dng_data['camera_calibration2'],
          'calibration_illuminant1': dng_data['calibration_illuminant1'],
          'calibration_illuminant2': dng_data['calibration_illuminant2'],
          'forward_matrix1': dng_data['forward_matrix1'], 'forward_matrix2': dng_data['forward_matrix2'],
          'shutter_speed': dng_data['shutter_speed'], 'fov': dng_data['fov'], 'width': dng_data['width'],
          'height': dng_data['height']
        }})

      file_utils.write_json_file(data, os.path.join(curr_folder, 'data', f.replace('.dng', '.json')))


  # Removes sRGB images that do not have corresponding raw
  for set_name in ['train', 'test', 'val']:
    print(f'Finalizing {set_name} ...')
    curr_folder = os.path.join(dest_directory, set_name)
    srgb_filenames = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(curr_folder, 'srgb_images')) if f.endswith('.jpg')]
    raw_filenames = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(curr_folder, 'raw_images')) if f.endswith('.png')]
    missing_files = list(set(srgb_filenames) - set(raw_filenames))
    for f in missing_files:
      os.remove(os.path.join(curr_folder, 'srgb_images', f))
