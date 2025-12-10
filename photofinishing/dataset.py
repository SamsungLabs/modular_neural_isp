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

This file defines the data loading pipeline for training the photofinishing module.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from os.path import join, exists, dirname, basename
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from typing import Optional, Dict, List
import shutil
import collections
import h5py

from utils.constants import PHOTOFINISHING_TRAINING_INPUT_SIZE
from utils.img_utils import (imread, augment_img, img_to_tensor, raw_to_lsrgb, clip, imresize,
                             extract_non_overlapping_patches)
from utils.file_utils import read_json_file

class Data(Dataset):
  def __init__(self, in_img_dir: str, gt_img_dir: str, data_dir: Optional[str]=None,
               image_size: Optional[int]=PHOTOFINISHING_TRAINING_INPUT_SIZE, extract_patches: Optional[bool]=False,
               temp_folder: Optional[str]='ps_temp_h5', overwrite_temp_folder: Optional[bool]=False,
               geometric_aug: Optional[bool]=True, batch_size: Optional[int]=8, shuffle: Optional[bool]=False):
    self._in_img_dir = in_img_dir
    self._gt_img_dir = gt_img_dir
    self._data_dir = data_dir
    self._extract_patches = extract_patches
    if self._data_dir is None:
      self._data_dir = join(dirname(self._in_img_dir.rstrip("/\\")), 'data')
    self._image_size = image_size
    self._geo_aug = geometric_aug
    self._batch_size = batch_size
    self._shuffle = shuffle
    self._max_open_files = 64

    assert self._image_size > 0, 'Invalid image size.'

    if self._extract_patches:
      postfix = '_patches'
    else:
      postfix = ''
    self._temp_dir = join(dirname(gt_img_dir),
                          f'{temp_folder}_{basename(gt_img_dir)}_bs_{batch_size}_sz_{self._image_size}{postfix}')

    re_create = True
    if exists(self._temp_dir) and overwrite_temp_folder:
      logging.info('Temporary directory exists. Removing it and re-preprocessing images...')
      shutil.rmtree(self._temp_dir)
    elif exists(self._temp_dir):
      re_create = False
    else:
      os.makedirs(self._temp_dir)

    if re_create:
      logging.info(f'Preprocessing images with batch_size={batch_size}...')
      self._create_hdf5_files()
    else:
      logging.info(f'Found pre-extracted batches in {self._temp_dir}; skipping reprocessing.')

    self._h5_file_paths = sorted([join(self._temp_dir, f) for f in os.listdir(self._temp_dir) if f.endswith('.h5')])
    self._h5_cache: 'collections.OrderedDict[str, h5py.File]' = collections.OrderedDict()

  def __len__(self):
    return len(self._h5_file_paths)

  def _open_h5_file(self, h5_path: str) -> h5py.File:
    """Opens an HDF5 file with caching to avoid too many open files."""
    if h5_path in self._h5_cache:
      self._h5_cache.move_to_end(h5_path)
      return self._h5_cache[h5_path]

    if len(self._h5_cache) >= self._max_open_files:
      old_path, old_file = self._h5_cache.popitem(last=False)
      old_file.close()

    f = h5py.File(h5_path, 'r')
    self._h5_cache[h5_path] = f
    return f

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """Returns a full batch (of size batch_size) from one HDF5 file."""
    h5_path = self._h5_file_paths[idx]
    with self._open_h5_file(h5_path) as f:
      in_images = f['in_images'][()]
      gt_images = f['gt_images'][()]
    if self._shuffle:
      indices = np.random.permutation(len(in_images))
      in_images = [in_images[i] for i in indices]
      gt_images = [gt_images[i] for i in indices]
    in_batch = []
    gt_batch = []

    for in_patch, gt_patch in zip(in_images, gt_images):
      if self._geo_aug:
        in_patch, gt_patch = augment_img(in_patch, gt_patch)
      in_batch.append(img_to_tensor(in_patch))
      gt_batch.append(img_to_tensor(gt_patch))
    return {'in_images': torch.stack(in_batch).to(dtype=torch.float32),
            'gt_images': torch.stack(gt_batch).to(dtype=torch.float32)}

  def _create_hdf5_files(self):
    """Creates HDF5 files where each file contains 'batch_size' images."""
    in_images: List[np.ndarray] = []
    gt_images: List[np.ndarray] = []
    file_counter = 0
    in_img_files = sorted(os.listdir(self._in_img_dir))
    gt_img_files = sorted(os.listdir(self._gt_img_dir))
    for i, (in_img_file, gt_img_file) in enumerate(zip(in_img_files, gt_img_files)):
      if in_img_file.startswith('.'):
        continue
      print(f'Processing {i}/{len(in_img_files)} ...')
      in_img_path = join(self._in_img_dir, in_img_file)
      data_path = join(self._data_dir, in_img_file.replace('.png', '.json'))
      gt_img_path = join(self._gt_img_dir, gt_img_file)
      in_img = imread(in_img_path)
      gt_img = imread(gt_img_path)
      metadata = read_json_file(data_path)
      illum = np.array(metadata['cam_illum'], dtype=np.float32)
      ccm = np.array(metadata['ccm'], dtype=np.float32)
      in_img = clip(raw_to_lsrgb(in_img, illum_color=illum, ccm=ccm))
      if self._extract_patches:
        patches = extract_non_overlapping_patches(img=in_img, gt_img=gt_img, num_patches=0,
                                                  patch_size=self._image_size, allow_overlap=True,
                                                  add_resized_patch=True)
        for in_patch, gt_patch in zip(patches['img'], patches['gt']):
          in_images.append(in_patch.astype(np.float32))
          gt_images.append(gt_patch.astype(np.float32))
          if len(in_images) == self._batch_size:
            self._write_hdf5(file_counter, in_images, gt_images)
            in_images = []
            gt_images = []
            file_counter += 1
      else:
        in_images.append(imresize(in_img, height=self._image_size, width=self._image_size))
        gt_images.append(imresize(gt_img, height=self._image_size, width=self._image_size))
        if len(in_images) == self._batch_size:
          self._write_hdf5(file_counter, in_images, gt_images)
          in_images = []
          gt_images = []
          file_counter += 1

    if in_images:
      self._write_hdf5(file_counter, in_images, gt_images)

  def _write_hdf5(self, file_id: int, in_images: List[np.ndarray], gt_images: List[np.ndarray]):
    fname = join(self._temp_dir, f"batch_{file_id:05d}.h5")
    logging.info(f'Writing batch of {len(in_images)} images to {fname}...')
    with h5py.File(fname, 'w') as f:
      f.create_dataset('in_images', data=np.stack(in_images), compression='gzip')
      f.create_dataset('gt_images', data=np.stack(gt_images), compression='gzip')

