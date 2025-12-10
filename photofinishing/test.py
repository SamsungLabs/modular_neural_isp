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

This file contains the testing script for the photofinishing module.
"""

import argparse
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import time
import numpy as np

from tabulate import tabulate
from typing import List, Optional
from utils.file_utils import read_json_file

import torch
from photofinishing_model import PhotofinishingModule
from utils.img_utils import imread, img_to_tensor, tensor_to_img, get_psnr, get_ssim, raw_to_lsrgb, imresize


def print_line(end: Optional[bool]=False, length: Optional[int]=30):
  """Prints a separator line."""
  line = '-' * length
  if end:
    logging.info(f"{line}\n")
  else:
    logging.info(f"\n{line}")


def test_net(model: PhotofinishingModule, te_device: torch.device, in_te_dir: str,
             gt_te_dir: str, data_te_dir: str, post_process_ltm: bool, no_ds: bool) -> str:
  """Tests a given trained model."""

  if data_te_dir is None:
    data_te_dir = os.path.join(os.path.dirname(in_te_dir.rstrip("/\\")), 'data')

  in_filenames = [os.path.join(in_te_dir, fn) for fn in os.listdir(in_te_dir) if fn.endswith('.png')]
  gt_filenames = [os.path.join(gt_te_dir, fn) for fn in os.listdir(gt_te_dir) if fn.endswith('.jpg')]
  data_filenames = [os.path.join(data_te_dir, fn) for fn in os.listdir(data_te_dir) if fn.endswith('.json')]

  psnr = np.zeros((len(in_filenames), 1))
  ssim = np.zeros((len(in_filenames), 1))
  total_time = 0
  for idx, (in_file, gt_file, data_file) in enumerate(zip(in_filenames, gt_filenames, data_filenames)):
    print(f'Processing {idx+1}/{len(in_filenames)}...')
    raw_img = imread(in_file).astype(np.float32)
    shape = raw_img.shape
    gt_img = imread(gt_file).astype(np.float32)
    metadata = read_json_file(data_file)
    illum = np.array(metadata['cam_illum'], dtype=np.float32)
    ccm = np.array(metadata['ccm'], dtype=np.float32)
    lsrgb_img = raw_to_lsrgb(raw_img, illum_color=illum, ccm=ccm)
    if not no_ds:
      target_shape = [shape[0] // 4, shape[1] // 4]
      lsrgb_img = imresize(lsrgb_img, height=target_shape[0], width=target_shape[1])
      gt_img = imresize(gt_img, height=target_shape[0], width=target_shape[1])
    lsrgb_img_tensor = img_to_tensor(lsrgb_img).unsqueeze(0).to(device=te_device, dtype=torch.float32)
    start = time.time()
    with torch.no_grad():
      out_img_tensor = model(lsrgb_img_tensor, post_process_ltm=post_process_ltm)['output']
    end = time.time()
    total_time += (end - start)
    out_img = tensor_to_img(out_img_tensor)
    psnr[idx] = get_psnr(out_img, gt_img)
    ssim[idx] = get_ssim(out_img, gt_img)
  return f'PSNR = {psnr.mean()} - SSIM = {ssim.mean()} - Time = {total_time / len(in_filenames)}\n'

def get_args():
  parser = argparse.ArgumentParser(description='Test the photofinishing network.')
  parser.add_argument('--model-path', dest='model_path', required=True, help='Path to the trained model.')
  parser.add_argument(
    '--in-testing-dir', dest='in_te_dir', type=str, required=True, help='Testing input image directory.')
  parser.add_argument(
    '--gt-testing-dir', dest='gt_te_dir', type=str, required=True,
    help='Testing ground-truth image directory.')
  parser.add_argument(
    '--data-testing-dir', dest='data_te_dir', default=None, type=str, help='Testing input data directory.')
  parser.add_argument('--post-process-ltm', dest='post_process_ltm', action='store_true',
                      help='Enable multi-scale and refinement of the LTM coeffs to mitigate potential halo artifacts '
                           '(refer to Sec. B.1 of the supp materials).')
  parser.add_argument('--no-ds', dest='no_ds', action='store_true',
                      help='To disable downsampling before photofinishing.')
  parser.add_argument('--config-dir', dest='config_dir', default='config',
                      help='Directory containing config JSON files.')
  parser.add_argument('--result-dir', dest='result_dir', default='results',
                      help='Directory to save the results report (.txt).')
  return parser.parse_args()


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  args = get_args()
  os.makedirs(args.result_dir, exist_ok=True)
  print(tabulate([(key, value) for key, value in vars(args).items()], headers=['Argument', 'Value'], tablefmt='grid'))
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  assert os.path.exists(args.model_path), f"Model file not found at '{args.model_path}'"

  logging.info(f'Using device {device}')
  logging.info(f'Testing of photofinishing module -- model name: {os.path.basename(args.model_path)} ...')
  config_base = os.path.splitext(os.path.basename(args.model_path))[0]
  config = read_json_file(os.path.join(args.config_dir, f'{config_base}.json'))

  net = PhotofinishingModule(device=device, use_3d_lut=config['use_3d_lut'])
  net.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
  net.eval()
  logging.info(f'Model loaded from {args.model_path}')
  net.print_num_of_params()

  results = test_net(model=net, te_device=device, in_te_dir=args.in_te_dir, gt_te_dir=args.gt_te_dir,
                     data_te_dir=args.data_te_dir, post_process_ltm=args.post_process_ltm, no_ds=args.no_ds)
  print(results)
  with open(os.path.join(args.result_dir, os.path.splitext(os.path.basename(args.model_path))[0] + '.txt'), 'w') as f:
    f.write(results)

