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

This demo demonstrates the console-based usage of our pipeline and its photo-editing tool features.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import argparse
from typing import Union
from utils.img_utils import (extract_image_from_dng, extract_raw_metadata, normalize_raw, demosaice, imwrite, imread,
                             extract_additional_dng_metadata, tensor_to_img)
import numpy as np
from utils.file_utils import read_json_file
from pipeline import PipeLine
from utils.constants import *
import torch


def find_json_for_png(input_file: str) -> Union[str, None]:
  """Finds the associated JSON metadata file for a PNG-16 input image."""
  if not input_file.lower().endswith('.png'):
    return None

  base_name = os.path.splitext(os.path.basename(input_file))[0]
  current_dir = os.path.dirname(input_file)
  parent_dir = os.path.dirname(current_dir)

  candidate_paths = [
      os.path.join(current_dir, f'{base_name}.json'),
      os.path.join(current_dir, 'data', f'{base_name}.json'),
      os.path.join(parent_dir, 'data', f'{base_name}.json'),
  ]

  for path in candidate_paths:
    if os.path.isfile(path):
      return path

  raise FileNotFoundError(
    f'Metadata JSON file for "{input_file}" not found. Checked:\n' +
    '\n'.join(candidate_paths))


def get_args():
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(description='Photofinishing Module Demo')

  parser.add_argument(
    '--input-file', dest='input_file', type=str, required=True,
    help=(
      'Path to a DNG, PNG-16 raw image, or a JPEG saved with embedded raw using our framework. '
      'If a PNG-16 is used, its data JSON file is assumed to be in the same directory, or in a "data" '
      'subdirectory in the same or parent directory. '
      'If a PNG-8 or a JPEG without embedded raw is provided, the image will first be linearized.'
    )
  )
  parser.add_argument('--output-dir', type=str, default='.',
                      help='Directory to save the output image.', dest='output_dir')

  parser.add_argument('--device', type=str, default='gpu', choices=DEVICES,
                      help=f'Device to run the model on: {DEVICES}.')

  parser.add_argument('--save-intermediate', action='store_true',
                      help='Save the output of intermediate stages as PNG-16 files.')

  parser.add_argument('--save-input-raw', action='store_true',
                      help='Save the input raw image as a 16-bit PNG file for reference purposes.')

  parser.add_argument('--re-compute-awb', action='store_true',
                      help='Recompute AWB gains using time-aware/C5 illuminant estimation methods.')
  parser.add_argument('--pref-awb', action='store_true',
                      help='Use user-preference AWB (only effective if --re-compute-awb is enabled).')
  parser.add_argument('--auto-exposure', action='store_true',
                      help='Applies post-capture auto exposure adjustment.')
  parser.add_argument('--ev-value', dest='ev_value', type=float, default=0.0,
                      help=f'Exposure value (EV) applied to the raw image. Range [{EV_MIN}, {EV_MAX}].')
  parser.add_argument('--denoising-strength', dest='denoising_strength', type=float, default=1.0,
                      help='Strength of denoising applied to the raw image. Range [0.0, 1.0].')
  parser.add_argument('--luma-denoising-strength', dest='luma_denoising_strength',
                      type=float, default=0.0,
                      help='Strength of luma denoising. Higher values apply stronger edge-preserving smoothing. '
                           'Range [0.0, 1.0].')
  parser.add_argument('--chroma-denoising-strength', dest='chroma_denoising_strength', type=float,
                      default=0.0, help='Strength of chroma denoising. Higher values apply stronger color noise '
                                        'suppression. Range [0.0, 1.0].')
  parser.add_argument('--enhancement-strength', dest='enhancement_strength', type=float, default=1.0,
                      help='Strength of enhancement applied to the image. Range [0.0, 1.0].')
  parser.add_argument('--post-process-ltm', dest='post_process_ltm',
                      action='store_true',
                      help='Enable multi-scale and refinement of the LTM coeffs to mitigate potential halo artifacts '
                           '(refer to Sec. B.1 of the supp materials).')
  parser.add_argument('--solver-iterations', dest='solver_iterations', type=int,
                      default=BILATERAL_SOLVER_ITERS,
                      help='Iterations for the bilateral solver (only active with --post-process-ltm).')
  parser.add_argument('--photofinishing-model-path', dest='photofinishing_model_path', default=None,
                      help='Path to the trained photofinishing model.')
  parser.add_argument('--multi-style-photofinishing-model-paths', type=str, nargs='+', default=None,
                      help=('List of paths to photofinishing models, one per picture style. Length must equal the '
                            'number of styles being mixed, and must match the lengths of the multi-style weights lists.'
                            ' If not provided, a single model path must be provided via --photofinishing-model-path.'))
  parser.add_argument('--multi-style-gain-weights', type=float, nargs='+', default=None,
                      help=('List of gain weights, one per picture style. Length must equal the number of styles being '
                            'mixed. Each value is a scalar controlling the contribution of the corresponding '
                            'style\'s gain.'))
  parser.add_argument('--multi-style-gamma-weights', type=float, nargs='+', default=None,
                      help=('List of gamma weights, one per picture style. Length must equal the number of styles '
                            'being mixed. Each value is a scalar controlling the contribution of the corresponding '
                            'style\'s gamma correction.'))
  parser.add_argument('--multi-style-ltm-weights', type=float, nargs='+', default=None,
                      help=('List of local tone mapping (LTM) weights, one per picture style. Length must equal the '
                            'number of styles being mixed. Each value is a scalar controlling the contribution of the '
                            'corresponding style\'s LTM.'))
  parser.add_argument('--multi-style-chroma-weights', type=float, nargs='+', default=None,
                      help=('List of chroma mapping weights, one per picture style. Length must equal the number of '
                            'styles being mixed. Each value is a scalar controlling the contribution of the '
                            'corresponding style\'s chroma mapping.'))
  parser.add_argument('--multi-style-gtm-weights', type=float, nargs='+', default=None,
                      help=('List of global tone mapping (GTM) weights, one per picture style. Length must equal the '
                            'number of styles being mixed. Each value is a scalar controlling the contribution of the '
                            'corresponding style\'s GTM.'))
  parser.add_argument('--multi-style-weights', type=float, nargs='+', default=None,
                      help=('Generic list of weights, one per picture style, applied uniformly to all '
                            'photofinishing components (gain, gamma, LTM, chroma, GTM). Length must equal the '
                            'number of styles being mixed. If provided, this overrides the need to specify the '
                            'individual component weights.'))
  parser.add_argument('--denoising-model-path', dest='denoising_model_path',
                      help='Path to the trained denoising model.')
  parser.add_argument('--enhancement-model-path', dest='enhancement_model_path',
                      help='Path to the trained enhancement model.')
  parser.add_argument('--contrast-amount', dest='contrast_amount', type=float, default=0.0,
                      help='Contrast amount in range [-1.0, 1.0]. Positive increases contrast, negative decreases '
                           'contrast.')
  parser.add_argument('--vibrance-amount', dest='vibrance_amount', type=float, default=0.0,
                      help='Vibrance amount in range [-1.0, 1.0]. Positive increases vibrance, negative decreases '
                           'vibrance.')
  parser.add_argument('--saturation-amount', dest='saturation_amount', type=float, default=0.0,
                      help='Saturation amount in range [-1.0, 1.0]. Positive increases saturation, negative decreases '
                           'saturation.')
  parser.add_argument('--sharpening-amount', dest='sharpening_amount', type=float, default=0.0,
                      help='Sharpening strength. Range [0.0, 50.0], with higher values increasing edge contrast. '
                           'Negative values apply softening instead of sharpening.')

  parser.add_argument('--highlight-amount', dest='highlight_amount', type=float, default=0.0,
                      help='Highlight adjustment amount in range [-1.0, 1.0]. Positive values boost highlights, '
                           'while negative values compress highlights to recover detail.')
  parser.add_argument('--shadow-amount', dest='shadow_amount', type=float, default=0.0,
                      help='Shadow adjustment amount in range [-1.0, 1.0]. Positive values lift shadows to brighten '
                           'dark regions, negative values deepen shadows to increase contrast in dark areas.')
  parser.add_argument('--target-cct', dest='target_cct', type=float, default=None,
                      help='Target correlated color temperature (CCT) in Kelvin. If None, this option is ignored and '
                           'no CCT adjustment is applied. Range: 1800.0–10000.0.')
  parser.add_argument('--target-tint', dest='target_tint', type=float, default=None,
                      help='Target tint (offset in uv space). If None, this option is ignored and no tint adjustment '
                           'is applied. Range: -100.0 to +100.0.')
  parser.add_argument('--store-raw', action='store_true',
                      help='Store the raw image and metadata inside the output JPEG.')
  parser.add_argument('--apply-orientation', action='store_true',
                      help='Apply the EXIF orientation (rotation/flip) to the output image.')
  return parser.parse_args()

if __name__ == '__main__':
  args = get_args()
  os.makedirs(args.output_dir, exist_ok=True)
  assert os.path.exists(args.input_file), 'File does not exist.'
  assert (
     args.input_file.lower().endswith('.dng')
     or args.input_file.lower().endswith('.png')
     or args.input_file.lower().endswith('.jpg')
     or args.input_file.lower().endswith('.jpeg')
  ), 'Invalid file type. Supported formats: DNG, PNG, JPEG.'

  basename = os.path.splitext(os.path.basename(args.input_file))[0]
  dir_path = args.output_dir

  assert args.device in DEVICES, 'Invalid device.'

  if args.device == 'gpu':
    if torch.cuda.is_available():
      device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
      device = torch.device('mps')
    else:
      device = torch.device('cpu')
  else:
    device = torch.device('cpu')

  if args.re_compute_awb:
    s24_awb_model_path = PATH_TO_S24_AWB_MODEL
    cc_awb_model_path = PATH_TO_GENERIC_AWB_MODEL
    post_awb_model_path = PATH_TO_POST_AWB_MODEL
    pref_awb = args.pref_awb
  else:
    s24_awb_model_path = None
    cc_awb_model_path = None
    post_awb_model_path = None
    pref_awb = False

  if args.store_raw or args.input_file.lower().endswith('.jpg'):
    raw_jpg_model_path = PATH_TO_RAW_JPEG_MODEL
  else:
    raw_jpg_model_path = None

  single_ps_path = args.photofinishing_model_path
  multi_ps_paths = args.multi_style_photofinishing_model_paths
  assert not (single_ps_path and multi_ps_paths), (
    f'Conflict: both --photofinishing-model-path and --multi-style-photofinishing-model-paths were given. '
    f'Expected exactly one of them.'
  )
  assert single_ps_path or multi_ps_paths, (
    'No photofinishing model paths provided. '
    'Expected either --photofinishing-model-path (single mode) '
    'or --multi-style-photofinishing-model-paths (multi-style mode).'
  )

  if multi_ps_paths:
    n_styles = len(multi_ps_paths)
    generic_w = args.multi_style_weights
    detailed = [
      args.multi_style_gain_weights,
      args.multi_style_gtm_weights,
      args.multi_style_ltm_weights,
      args.multi_style_chroma_weights,
      args.multi_style_gamma_weights,
    ]
    if generic_w is None and all(w is None for w in detailed):
      raise AssertionError('No multi-style weights provided. Expected either --multi-style-weights or all '
                           'component-specific weight lists.')

    if generic_w is not None:
      assert len(generic_w) == n_styles, (f'Length mismatch for --multi-style-weights. Expected {n_styles} values, '
                                          f'got {len(generic_w)}.')
    else:
      for name, w in zip(['--multi-style-gain-weights', '--multi-style-gamma-weights', '--multi-style-ltm-weights',
                          '--multi-style-chroma-weights', '--multi-style-gtm-weights',], detailed):
        assert w is not None, (f'{name} missing. Expected all component-specific weight lists when '
                               '--multi-style-weights is not provided.')
        assert len(w) == n_styles, f'Length mismatch for {name}. Expected {n_styles} values, got {len(w)}.'

    if generic_w is None:
      ps_weights = {'gain': np.array(detailed[0]) / (np.array(detailed[0]).sum() + EPS),
                 'gtm': np.array(detailed[1]) / (np.array(detailed[1]).sum() + EPS),
                 'ltm': np.array(detailed[2]) / (np.array(detailed[2]).sum() + EPS),
                 'chroma': np.array(detailed[3]) / (np.array(detailed[3]).sum() + EPS),
                 'gamma': np.array(detailed[4]) / (np.array(detailed[4]).sum() + EPS),}
    else:
      ps_weights = {'generic': np.array(generic_w) / (np.array(generic_w).sum() + EPS)}
  else:
    ps_weights = None

  net = PipeLine(running_device=device, photofinishing_model_path=single_ps_path,
                 photofinishing_style_model_paths=multi_ps_paths,
                 generic_denoising_model_path=args.denoising_model_path, denoising_model_path=args.denoising_model_path,
                 enhancement_model_path=args.enhancement_model_path, s24_awb_model_path=s24_awb_model_path,
                 cc_awb_model_path=cc_awb_model_path,  post_awb_model_path=post_awb_model_path,
                 raw_jpeg_adapter_model_path=raw_jpg_model_path)
  net.eval()

  if args.input_file.lower().endswith('.png'):
    _, dtype = imread(args.input_file, return_dtype=True)
  else:
    dtype = None

  if args.input_file.lower().endswith('.dng'):
    metadata = extract_raw_metadata(args.input_file)
    metadata.update(extract_additional_dng_metadata(args.input_file))
    raw_img = extract_image_from_dng(args.input_file)
    raw_img = normalize_raw(img=raw_img, black_level=metadata['black_level'],
                            white_level=metadata['white_level']).astype(np.float32)
    if not (raw_img.shape[-1] == 4 and len(raw_img.shape) == 3):
      try:
        print('demosaicing...\n')
        raw_img = demosaice(raw_img, metadata['pattern'], tile_mode=True)
      except:
        raise NotImplementedError('Unsupported bayer pattern.')
    else:
      raw_img = raw_img[..., :3]
  elif args.input_file.lower().endswith('.png') and dtype == np.uint16:
    raw_img = imread(args.input_file).astype(np.float32)
    metadata_path = find_json_for_png(args.input_file)
    if metadata_path is None:
      raise FileNotFoundError
    metadata = read_json_file(metadata_path)
  elif (
     args.input_file.lower().endswith('.png') or args.input_file.lower().endswith('.jpg') or
     args.input_file.lower().endswith('.jpeg')):
    outputs = net.read_image(args.input_file, log_messages=True, report_time=True)
    raw_img = outputs['raw']
    metadata = outputs['metadata']
    if raw_img is None or metadata is None:
      linearization_model_path = PATH_TO_LINEARIZATION_MODEL
      net.update_model(linearization_model_path=linearization_model_path)
      outputs = net.read_image(args.input_file, log_messages=True, report_time=True)
      raw_img = outputs['raw']
      metadata = outputs['metadata']
      if raw_img is None or metadata is None:
        raise ValueError('Input image is missing embedded raw data and/or metadata; linearization failed.')

  else:
    raise ValueError('Unexpected input file format. Supported formats: DNG, PNG-16, or JPEG with embedded raw.')
  if not args.re_compute_awb:
    if 'cam_illum' in metadata:
      key = 'cam_illum'
    else:
      key = 'illum_color'
    illum = np.array(metadata[key], dtype=np.float32)
    if 'color_matrix' in metadata:
      key = 'color_matrix'
    else:
      key = 'ccm'
    ccm = np.array(metadata[key], dtype=np.float32)
    img_metadata = None
  else:
    illum = None
    ccm = None

  # Save raw image
  if args.save_input_raw:
    raw_file_name = os.path.join(args.output_dir, f'{basename}-0-in-raw.png')
    imwrite(raw_img, raw_file_name, format='png-16')

  with torch.no_grad():
    if multi_ps_paths:
      style_gain = []
      style_gtm = []
      style_ltm = []
      style_chroma = []
      style_gamma = []

      if 'generic' in ps_weights:
        gain_w = gtm_w = ltm_w = chroma_w = gamma_w = ps_weights['generic'][0]
      else:
        gain_w = ps_weights['gain'][0]
        gtm_w = ps_weights['gtm'][0]
        ltm_w = ps_weights['ltm'][0]
        chroma_w = ps_weights['chroma'][0]
        gamma_w = ps_weights['gamma'][0]



      # get params of the first style
      outputs = net(raw_img, post_process_ltm=args.post_process_ltm, illum=illum, ccm=ccm,
                    solver_iter=args.solver_iterations, awb_user_pref=pref_awb, img_metadata=metadata,
                    auto_exposure=args.auto_exposure, log_messages=True, contrast_amount=args.contrast_amount,
                    vibrance_amount=args.vibrance_amount, saturation_amount=args.saturation_amount,
                    shadow_amount=args.shadow_amount, highlight_amount=args.highlight_amount,
                    sharpening_amount=args.sharpening_amount, target_cct=args.target_cct, target_tint=args.target_tint,
                    denoising_strength=args.denoising_strength, enhancement_strength=args.enhancement_strength,
                    chroma_denoising_strength=args.chroma_denoising_strength,
                    luma_denoising_strength=args.luma_denoising_strength,
                    gain_param=None if gain_w else 0, gtm_param=None if gtm_w else 0,
                    ltm_param=None if ltm_w else 0, chroma_lut_param=None if chroma_w else 0,
                    gamma_param=None if gamma_w else 0,
                    ev_scale=args.ev_value, report_time=True, style_id=1)
      lsrgb = outputs['lsrgb']
      denoised_raw = outputs['denoised_raw']
      style_gain.append(outputs['gain_param'])
      style_gtm.append(outputs['gtm_param'])
      style_ltm.append(outputs['ltm_param'])
      style_chroma.append(outputs['chroma_lut_param'])
      style_gamma.append(outputs['gamma_param'])

      for style_idx in range(1, len(multi_ps_paths)):
        if 'generic' in ps_weights:
          gain_w = gtm_w = ltm_w = chroma_w = gamma_w = ps_weights['generic'][style_idx]
        else:
          gain_w = ps_weights['gain'][style_idx]
          gtm_w = ps_weights['gtm'][style_idx]
          ltm_w = ps_weights['ltm'][style_idx]
          chroma_w = ps_weights['chroma'][style_idx]
          gamma_w = ps_weights['gamma'][style_idx]
        outputs = net.get_ps_params(lsrgb=lsrgb, style_id=style_idx + 1, post_process_ltm=args.post_process_ltm,
                                    solver_iter=args.solver_iterations, contrast_amount=args.contrast_amount,
                                    vibrance_amount=args.vibrance_amount, saturation_amount=args.saturation_amount,
                                    highlight_amount=args.highlight_amount, shadow_amount=args.shadow_amount,
                                    gain_param=None if gain_w else 0, gtm_param=None if gtm_w else 0,
                                    ltm_param=None if ltm_w else 0, chroma_lut_param=None if chroma_w else 0,
                                    gamma_param=None if gamma_w else 0)
        style_gain.append(outputs['gain_param'])
        style_gtm.append(outputs['gtm_param'])
        style_ltm.append(outputs['ltm_param'])
        style_chroma.append(outputs['chroma_lut_param'])
        style_gamma.append(outputs['gamma_param'])

      gain_params, gtm_params, ltm_params, chroma_params, gamma_params = 0, 0, 0, 0, 0
      for style_idx in range(len(multi_ps_paths)):
        if 'generic' in ps_weights:
          gain_weight = ps_weights['generic'][style_idx]
          gtm_weight = ps_weights['generic'][style_idx]
          ltm_weight = ps_weights['generic'][style_idx]
          chroma_weight = ps_weights['generic'][style_idx]
          gamma_weight = ps_weights['generic'][style_idx]
        else:
          gain_weight = ps_weights['gain'][style_idx]
          gtm_weight = ps_weights['gtm'][style_idx]
          ltm_weight = ps_weights['ltm'][style_idx]
          chroma_weight = ps_weights['chroma'][style_idx]
          gamma_weight = ps_weights['gamma'][style_idx]
        gain_params += gain_weight * style_gain[style_idx] if gain_weight else 0
        gtm_params += gtm_weight * style_gtm[style_idx] if gtm_weight else 0
        ltm_params += ltm_weight * style_ltm[style_idx] if ltm_weight else 0
        chroma_params += chroma_weight * style_chroma[style_idx] if chroma_weight else 0
        gamma_params += gamma_weight * style_gamma[style_idx] if gamma_weight else 0

      # final rendering:
      outputs = net(raw_img, denoised_raw=denoised_raw, illum=illum, ccm=ccm, awb_user_pref=pref_awb,
                    img_metadata=metadata, log_messages=True, sharpening_amount=args.sharpening_amount,
                    target_cct=args.target_cct, target_tint=args.target_tint,
                    contrast_amount=args.contrast_amount,
                    vibrance_amount=args.vibrance_amount, saturation_amount=args.saturation_amount,
                    shadow_amount=args.shadow_amount, highlight_amount=args.highlight_amount,
                    denoising_strength=args.denoising_strength, enhancement_strength=args.enhancement_strength,
                    chroma_denoising_strength=args.chroma_denoising_strength,
                    luma_denoising_strength=args.luma_denoising_strength, gain_param=gain_params, gtm_param=gtm_params,
                    ltm_param=ltm_params, chroma_lut_param=chroma_params, gamma_param=gamma_params, report_time=True,
                    style_id=1, return_intermediate=args.save_intermediate, apply_orientation=args.apply_orientation)

    else:
      outputs = net(raw_img, post_process_ltm=args.post_process_ltm, illum=illum, ccm=ccm,
                    solver_iter=args.solver_iterations, awb_user_pref=pref_awb, img_metadata=metadata,
                    auto_exposure=args.auto_exposure, log_messages=True, contrast_amount=args.contrast_amount,
                    vibrance_amount=args.vibrance_amount, saturation_amount=args.saturation_amount,
                    shadow_amount=args.shadow_amount, highlight_amount=args.highlight_amount,
                    sharpening_amount=args.sharpening_amount, target_cct=args.target_cct, target_tint=args.target_tint,
                    denoising_strength=args.denoising_strength, enhancement_strength=args.enhancement_strength,
                    chroma_denoising_strength=args.chroma_denoising_strength,
                    luma_denoising_strength=args.luma_denoising_strength, ev_scale=args.ev_value, report_time=True,
                    return_intermediate=args.save_intermediate, apply_orientation=args.apply_orientation)

  if args.save_intermediate:
    denoised_raw_file_name = os.path.join(args.output_dir, f'{basename}-1-denoised.png')
    imwrite(tensor_to_img(outputs['denoised_raw']), denoised_raw_file_name, format='png-16')
    lsrgb_file_name = os.path.join(args.output_dir, f'{basename}-2-lsrgb.png')
    imwrite(tensor_to_img(outputs['lsrgb']), lsrgb_file_name, format='png-16')
    imwrite(tensor_to_img(outputs['lsrgb_gain']), os.path.join(dir_path, f'{basename}-3-gain.png'), 'PNG-16')
    imwrite(tensor_to_img(outputs['lsrgb_gtm']), os.path.join(dir_path, f'{basename}-4-gtm.png'), 'PNG-16')
    imwrite(tensor_to_img(outputs['lsrgb_ltm']), os.path.join(dir_path, f'{basename}-5-ltm.png'), 'PNG-16')
    imwrite(tensor_to_img(outputs['processed_lsrgb']), os.path.join(dir_path, f'{basename}-6-cbcr-lut.png'), 'PNG-16')
    imwrite(tensor_to_img(outputs['gamma']), os.path.join(dir_path, f'{basename}-7-gamma.png'), 'PNG-16')

  if args.store_raw:
    print('Saving (with raw embedded)...')
    net.save_image(srgb=outputs['srgb'], raw=raw_img, metadata=outputs['metadata'],
                   output_path=os.path.join(dir_path, f'{basename}-output.jpg'), log_messages=True, report_time=True)
  else:
    print('Saving...')
    imwrite(outputs['srgb'], os.path.join(dir_path, f'{basename}-output.jpg'), 'JPEG',

            quality=DEFAULT_SRGB_JPEG_QUALITY)


