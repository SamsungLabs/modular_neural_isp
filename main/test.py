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

This file contains the testing script for the entire pipeline.
"""

import torch
from typing import Optional
import numpy as np
import time
import argparse
import logging
import lpips
import sys
from tabulate import tabulate
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))


from pipeline import PipeLine
from utils.constants import *
from utils.img_utils import imread, img_to_tensor, tensor_to_img, get_psnr, get_ssim, get_lpips, get_delta_e
from utils.file_utils import read_json_file


def print_line(end: Optional[bool]=False, length: Optional[int]=30):
  """Prints a separator line."""
  line = '-' * length
  if end:
    logging.info(f"{line}\n")
  else:
    logging.info(f"\n{line}")


def test_net(model: PipeLine, te_device: torch.device, in_te_dir: str, gt_te_dir: str, data_te_dir: str,
             post_process_ltm: bool, report_lpips: bool, report_delta_e: bool, no_downsampling: bool) -> str:
  """Tests a given trained model."""

  if data_te_dir is None:
    data_te_dir = os.path.join(os.path.dirname(in_te_dir.rstrip("/\\")), 'data')
  in_filenames = [os.path.join(in_te_dir, fn) for fn in os.listdir(in_te_dir) if fn.endswith('.png')]
  gt_filenames = [os.path.join(gt_te_dir, fn) for fn in os.listdir(gt_te_dir) if fn.endswith('.jpg')]
  data_filenames = [os.path.join(data_te_dir, fn) for fn in os.listdir(data_te_dir) if fn.endswith('.json')]

  if report_lpips:
    lpips_model = lpips.LPIPS(net='vgg').to(device=te_device)
  else:
    lpips_model = None

  psnr = np.zeros((len(in_filenames), 1))
  ssim = np.zeros((len(in_filenames), 1))
  if report_lpips:
    lpips_ = np.zeros((len(in_filenames), 1))
  else:
    lpips_ = None
  if report_delta_e:
    delta_e = np.zeros((len(in_filenames), 1))
  else:
    delta_e = None
  total_time = 0

  for idx, (in_file, gt_file, data_file) in enumerate(zip(in_filenames, gt_filenames, data_filenames)):
    print(f'Processing {idx+1}/{len(in_filenames)}...', flush=True)
    raw_img = imread(in_file).astype(np.float32)
    gt_img = imread(gt_file).astype(np.float32)
    metadata = read_json_file(data_file)
    illum = np.array(metadata['cam_illum'], dtype=np.float32)
    ccm = np.array(metadata['ccm'], dtype=np.float32)
    raw_img = img_to_tensor(raw_img).unsqueeze(0).to(device=te_device)
    illum = torch.from_numpy(illum).unsqueeze(0).to(device=te_device)
    ccm = torch.from_numpy(ccm).unsqueeze(0).to(device=te_device)
    start = time.time()
    with torch.no_grad():
      out_img = model(raw=raw_img, illum=illum, ccm=ccm, post_process_ltm=post_process_ltm,
                      downscale_ps=not no_downsampling)['srgb']
    end = time.time()
    out_img = tensor_to_img(out_img)
    total_time += (end - start)
    psnr[idx] = get_psnr(out_img, gt_img)
    ssim[idx] = get_ssim(out_img, gt_img)
    if report_lpips:
      lpips_[idx] = get_lpips(out_img, gt_img, lpips_model)
    if report_delta_e:
      delta_e[idx] = get_delta_e(out_img, gt_img)
  message = f'PSNR = {psnr.mean()} - SSIM = {ssim.mean()} - '
  if report_lpips:
    message += f'LPIPS = {lpips_.mean()} - '
  if report_delta_e:
    message += f'Delta E 2000 = {delta_e.mean()} - '
  message += f'Time = {total_time / len(in_filenames)}\n'
  return message

def get_args():
  parser = argparse.ArgumentParser(description='Test the photofinishing network.')
  parser.add_argument(
    '--in-testing-dir', dest='in_te_dir', type=str, help='Testing input image directory.')
  parser.add_argument(
    '--gt-testing-dir', dest='gt_te_dir', type=str, help='Testing ground-truth image directory.')
  parser.add_argument(
    '--data-testing-dir', dest='data_te_dir', default=None, type=str,
    help='Testing input data directory.')
  parser.add_argument('--post-process-ltm', dest='post_process_ltm', action='store_true',
                      help='Enable multi-scale and refinement of the LTM coeffs to mitigate potential halo artifacts '
                           '(refer to Sec. B.1 of the supp materials).')
  parser.add_argument('--no-downsampling', dest='no_downsampling', action='store_true',
                      help='Disable image downsampling before photofinishing and skip guided upsampling after.')
  parser.add_argument('--report-lpips', dest='report_lpips', action='store_true',
                      help='To report LPIPS.')
  parser.add_argument('--report-delta-e', dest='report_delta_e', action='store_true',
                      help='To report Delta E 2000.')
  parser.add_argument('--result-dir', dest='result_dir', default='results',
                      help='Directory to save the results report (.txt).')
  parser.add_argument('--result-file-postfix', dest='result_postfix', default=None,
                      help='Optional postfix to append to the result file name.')
  parser.add_argument('--photofinishing-model-path', dest='photofinishing_model_path',
                      help='Path to the trained photofinishing model.')
  parser.add_argument('--denoising-model-path', dest='denoising_model_path',
                      help='Path to the trained denoising model.')
  parser.add_argument('--enhancement-model-path', dest='enhancement_model_path',
                      help='Path to the trained enhancement model.')
  return parser.parse_args()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  args = get_args()
  os.makedirs(args.result_dir, exist_ok=True)
  print(tabulate([(key, value) for key, value in vars(args).items()], headers=['Argument', 'Value'], tablefmt='grid'))
  if torch.cuda.is_available():
    device = torch.device('cuda')
  elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
  else:
    device = torch.device('cpu')

  logging.info(f'Using device {device}')
  enhancement_model_path = (
    os.path.basename(args.enhancement_model_path) if args.enhancement_model_path is not None else None)
  logging.info(f'Testing of modular pipeline:\n'
               f'Denoising model name: {os.path.basename(args.denoising_model_path)}\n'
               f'Enhancement model name: {enhancement_model_path}\n'
               f'Photofinishing model name: {os.path.basename(args.photofinishing_model_path)}...\n')

  net = PipeLine(running_device=device, photofinishing_model_path=args.photofinishing_model_path,
                 generic_denoising_model_path=args.denoising_model_path,
                 enhancement_model_path=args.enhancement_model_path)
  net.eval()

  results = test_net(model=net, te_device=device, in_te_dir=args.in_te_dir, gt_te_dir=args.gt_te_dir,
                     data_te_dir=args.data_te_dir, post_process_ltm=args.post_process_ltm,
                     report_lpips=args.report_lpips, report_delta_e=args.report_delta_e,
                     no_downsampling=args.no_downsampling)
  print(results)
  postfix = '' if enhancement_model_path is None else '_' + os.path.splitext(enhancement_model_path)[0]
  if args.result_postfix is not None:
    postfix += f'_{args.result_postfix}'
  with open(os.path.join(
     args.result_dir, os.path.splitext(
       os.path.basename(args.photofinishing_model_path))[0] + '_' + os.path.splitext(
       os.path.basename(args.denoising_model_path))[0] + postfix +'.txt'), 'w') as f:
    f.write(results)


