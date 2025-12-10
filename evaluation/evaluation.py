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

This script calculates and reports PSNR, SSIM, delta E 2000, and LPIPS metrics between images in the GT and result dirs.
"""

import os
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
import numpy as np
import argparse
from tqdm import tqdm
import lpips
from utils.img_utils import imread, get_psnr, get_ssim, get_lpips, get_delta_e
from typing import Optional
import torch

def calculate_metrics(result_dir, gt_dir, device: Optional[str]='gpu'):
    """Calculates metrics of results in result_dr compared to gt_dir."""
    lpips_model = lpips.LPIPS(net='vgg').to(device=torch.device(
        'cuda' if device.lower()=='gpu' else 'cpu'))
    psnr_list = []
    ssim_list = []
    lpips_list = []
    delta_e_list = []

    result_files = sorted(os.listdir(result_dir))
    gt_files = sorted(os.listdir(gt_dir))

    if len(result_files) != len(gt_files):
        raise ValueError(f'Mismatch in the number of images: {len(result_files)} in result directory, '
                         f'{len(gt_files)} in ground truth directory.')

    for result_file, gt_file in tqdm(zip(result_files, gt_files), total=len(result_files)):
        result_path = os.path.join(result_dir, result_file)
        gt_path = os.path.join(gt_dir, gt_file)

        result_img = imread(result_path)[:, :, :3]
        gt_img = imread(gt_path)[:, :, :3]

        psnr_list.append(get_psnr(gt_img, result_img))

        ssim_list.append(get_ssim(gt_img, result_img))

        lpips_list.append(get_lpips(gt_img, result_img, lpips_model, device=device))

        delta_e_list.append(get_delta_e(gt_img, result_img))

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_lpips = np.mean(lpips_list)
    avg_delta_e = np.mean(delta_e_list)

    return avg_psnr, avg_ssim, avg_lpips, avg_delta_e


def main():
    parser = argparse.ArgumentParser(description='Calculate PSNR, SSIM, LPIPS, and delta E between result & gt images.')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Directory containing result images.')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Directory containing ground truth images.')
    parser.add_argument('--device', type=str, required=True,
                        choices=['gpu', 'cpu'], help='Options: "gpu" or "cpu".')
    args = parser.parse_args()

    if not os.path.exists(args.result_dir) or not os.path.exists(args.gt_dir):
        print('Error: One or both of the directories do not exist.')
        return

    avg_psnr, avg_ssim, avg_lpips, avg_delta_e = calculate_metrics(args.result_dir, args.gt_dir, args.device)

    print(f'Average PSNR: {avg_psnr:.2f} dB')
    print(f'Average SSIM: {avg_ssim:.4f}')
    print(f'Average LPIPS: {avg_lpips:.4f}')
    print(f'Average DeltaE2000: {avg_delta_e:.4f}')

if __name__ == "__main__":
    main()
