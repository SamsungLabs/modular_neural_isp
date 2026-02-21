"""
Copyright (c) 2025-present Samsung Electronics Co., Ltd.

Author(s):
Mahmoud Afifi (m.afifi1@samsung.com, m.3afifi@gmail.com)
Luxi Zhao (lucy.zhao@samsung.com, lucyzhao.zlx@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

This file contains all constant values used in this project.
"""


# To install Exiftool (optional):
# 1) pip install PyExifTool
# 2) sudo apt install libimage-exiftool-perl (for Debian/Ubuntu) Or
#    sudo pacman -S perl-image-exiftool (for Arch Linux) Or
#    manual install from exiftool.org for Windows or MacOS -- rename exiftool(-k).exe to exiftool.ext
# 3) which exiftool (that will print the location to update EXIFTOOL_PATH or paste location of exiftool.exe if manually
# installed)

import platform
import os

os_name = platform.system()

if os_name == 'Windows':
  EXIFTOOL_PATH = r'C:\exiftool\exiftool.exe'  # Update accordingly
elif os_name == 'Darwin':
  EXIFTOOL_PATH = '/usr/local/bin/exiftool' # Update accordingly
else:
  EXIFTOOL_PATH = '/usr/bin/exiftool'  # Update accordingly

VERSION = 'v0.1.5-beta'

EPS = 0.00000001

DEVICES = ['cpu', 'gpu']

MIN_DIM = 2000

LOG_MAX_LENGTH = 400

# AWB
AWB_S24_MODELS = ['awb_iso_shutter_speed', 'awb_iso_shutter_speed_noise', 'awb_iso_shutter_speed_snr',
                  'awb_iso_shutter_speed_noise_snr']
AWB_CC_MODELS = ['c5_single_encoder_neutral']
AWB_UP_MODELS = ['c5_single_encoder_neutral_to_s24_user_pref']
AWB_CAPTURE_DATA_SIZE = [2, 8, 8, 14]
AWB_MODELS_DIR = 'awb_ccm/models'
PI = 22.0 / 7
SHRINK_FACTOR = 0.9999999
XYZ_TO_SRGB_D50 = [[3.1338561, -1.6168667, -0.4906146], [-0.9787684, 1.9161415, 0.0334540],
                   [0.0719453, -0.2289914, 1.4052427]]
CALIB_ILLUM1 = 21
CALIB_ILLUM2 = 17
TINT_SCALE = 0.0005

# Denoising
DENOISING_MODELS_DIR = 'denoising/models'

# Saving
JPEG_ADAPT_LUT_BINS = 128
JPEG_ADAPT_LUT_CHS = 3
RAW_JPEG_QUALITY_OPTIONS = [25, 50, 75, 95]
JPEG_RAW_MODELS = {25: 'raw_jpeg_adapter_q_25.pth', 50: 'raw_jpeg_adapter_q_50.pth',
                   75: 'raw_jpeg_adapter_q_75.pth', 95: 'raw_jpeg_adapter_q_95.pth'}
DEFAULT_SRGB_JPEG_QUALITY = 95
DEFAULT_RAW_JPEG_QUALITY = 95

# Photofinishing
RGB_TO_YCBCR = [[0.2126, 0.7152, 0.0722], [-0.114572, -0.385428, 0.5], [0.5, -0.454153, -0.045847]] #BT.709
YCBCR_TO_RGB = [[1.0, 0.0, 1.5748], [1.0, -0.1873, -0.4681], [1.0, 1.8556, 0.0]]


BILATERAL_SOLVER_ITERS = 80

CBCR_LUT_SIZE = 24

RGB_LUT_SIZE = 11

GAIN_MIN = 0.25   # EV -2
GAIN_MAX = 4.0    # EV +2

GAMMA_MIN = 1.2
GAMMA_MAX = 3.0

LTM_GRID_SIZE = 64
LTM_GRID_DEPTH = 18

LUT_NET_INPUT_SIZE = 128
GAIN_NET_INPUT_SIZE = 128
LTM_NET_INPUT_SIZE = 384
GAMMA_NET_INPUT_SIZE = 128
GTM_NET_INPUT_SIZE = 128

PHOTOFINISHING_TRAINING_INPUT_SIZE = 512

LTM_NET_BASE_CHANNELS = 48

LUT_NET_BOTTLENECK_CHANNELS = 28


GAIN_NET_BASE_CHANNELS = 20
GTM_NET_BASE_CHANNELS = 20


COLOR_NET_BASE_CHANNELS = 24

GAMMA_NET_BASE_CHANNELS = 20

# ENHANCEMENT
ENHANCEMENT_TRAINING_INPUT_SIZE = 512
ENHANCEMENT_MODEL_WIDTH = 8
ENHANCEMENT_MODEL_MIDDLE = 4
ENHANCEMENT_MODEL_ENCODER = [2, 2]
ENHANCEMENT_MODEL_DECODER = [2, 2]


# GUI
BACKGROUND_COLOR = '#1e1e1e'
SECOND_BACKGROUND_COLOR = '#323232'
FOREGROUND_COLOR = '#535353'
TEXT_COLOR = '#dddddd'
WARNING_COLOR = '#fde58f'
ERROR_COLOR = '#fd8f8f'
STYLE_NAMES = ['Default', 'Warm', 'Moody', 'Cinematic', 'Greenish', 'Retro']
PAUSE_TIME = 1000
AUTO_EXPOSURE_MAX_VALUE = 1.8
AUTO_EXPOSURE_HISTOGRAM_BINS = 96
AUTO_EXPOSURE_TARGET_GRAY = 0.08
AUTO_EXPOSURE_INPUT_SIZE = 128
EV_MIN = -4
EV_MAX = 4
BILATERAL_SOLVER_ITERS_OPTIONS = [20, 40, 80, 160]
DEFAULT_BILATERAL_SOLVER_ITERS_CPU_IDX = 0
DEFAULT_BILATERAL_SOLVER_ITERS_GPU_IDX = 2
PHOTOFINISHING_MODELS_FOLDER = os.path.join('..', 'photofinishing', 'models')
AWB_MODELS_FOLDER = os.path.join('..', 'awb_ccm', 'models')
IO_MODELS_FOLDER = os.path.join('..', 'io_', 'models')
DENOISING_MODELS_FOLDER = os.path.join('..', 'denoising', 'models')
PATH_TO_PHOTOFINISHING_DEFAULT_MODEL = os.path.join(PHOTOFINISHING_MODELS_FOLDER, 'photofinishing_s24-style-0.pth')
PATHS_TO_PHOTOFINISHING_MODELS = [
                                  os.path.join(PHOTOFINISHING_MODELS_FOLDER, 'photofinishing_s24-style-1.pth'),
                                  os.path.join(PHOTOFINISHING_MODELS_FOLDER, 'photofinishing_s24-style-2.pth'),
                                  os.path.join(PHOTOFINISHING_MODELS_FOLDER, 'photofinishing_s24-style-3.pth'),
                                  os.path.join(PHOTOFINISHING_MODELS_FOLDER, 'photofinishing_s24-style-4.pth'),
                                  os.path.join(PHOTOFINISHING_MODELS_FOLDER, 'photofinishing_s24-style-5.pth')
                                  ]
NUMBER_OF_STYLES = 1 + len(PATHS_TO_PHOTOFINISHING_MODELS)
PATH_TO_GENERIC_DENOISER_MODEL = os.path.join(DENOISING_MODELS_FOLDER, 'generic_base.pth')
PATH_TO_S24_DENOISER_MODEL = os.path.join(DENOISING_MODELS_FOLDER, 's24_base.pth')
PATH_TO_ENHANCEMENT_MODEL = os.path.join('..', 'enhancement', 'models', 'enhancement_s24-style-0.pth')
PATH_TO_S24_AWB_MODEL = os.path.join(AWB_MODELS_FOLDER, 'model-awb_iso_shutter_speed_noise.pt')
PATH_TO_GENERIC_AWB_MODEL = os.path.join(AWB_MODELS_FOLDER, 'model-c5_single_encoder_neutral.pth')
PATH_TO_POST_AWB_MODEL = os.path.join(AWB_MODELS_FOLDER, 'model-c5_single_encoder_neutral_to_s24_user_pref.pt')
PATH_TO_LINEARIZATION_MODEL = os.path.join(IO_MODELS_FOLDER, 'cie_xyz_net.pth')
PATH_TO_RAW_JPEG_MODEL = os.path.join(IO_MODELS_FOLDER, JPEG_RAW_MODELS[DEFAULT_RAW_JPEG_QUALITY])

DEFAULT_CCT = 5500
DEFAULT_TINT = 0
CCT_MIN = 1800
CCT_MAX = 10000
TINT_MIN = -100
TINT_MAX = 100

DEFAULT_DENOISING = 100
DEFAULT_CHROMA_DENOISING = 0
DEFAULT_LUMA_DENOISING = 0
DEFAULT_ENHANCEMENT = 50


DENOISING_MIN = 0
DENOISING_MAX = 100
CHROMA_DENOISING_MIN = 0
CHROMA_DENOISING_MAX = 100
LUMA_DENOISING_MIN = 0
LUMA_DENOISING_MAX = 100
ENHANCEMENT_MIN = 0
ENHANCEMENT_MAX = 100


DEFAULT_HIGHLIGHTS = 0
DEFAULT_SHADOWS = 0
DEFAULT_SATURATION = 0
DEFAULT_VIBRANCE = 0
DEFAULT_CONTRAST = 0
DEFAULT_SHARPENING = 0

HIGHLIGHTS_MIN = -100
HIGHLIGHTS_MAX = 100
SHADOWS_MIN = -100
SHADOWS_MAX = 100
SATURATION_MIN = -100
SATURATION_MAX = 100
VIBRANCE_MIN = -100
VIBRANCE_MAX = 100
CONTRAST_MIN = -100
CONTRAST_MAX = 100
SHARPENING_MIN = 0
SHARPENING_MAX = 100
TEMPORAL_WINDOW_SIZE = 9

PREVIEW_IMAGE_MAX_SIZE = 1500

D50_RAW_ILLUM = [ 0.47183424, 1.0, 0.6412845 ]
D50_INV_RAW_TO_XYZ = [[ 1.81713277, -0.60861565, -0.1730732],
                      [-0.26556286, 1.1600217, 0.11623782],
                      [-0.07264907, 0.42998402, 0.77540641]]
D50_CCM = [[ 1.65023296, -0.45531686, -0.19612742],
           [-0.31783843, 1.58334158, -0.26515062],
           [-0.01829498, -0.92885035, 1.94742248]]
METADATA = {
    "make": "None",
    "model": "Synthetic",
    "exposure_time": 0.03333333333,
    "f_number": 1.7,
    "iso": 250,
    "focal_length": 6.3,
    "color_matrix1": [
        0.697265625, -0.1767578125, -0.0751953125,
        -0.353515625, 1.202148438, 0.123046875,
        -0.1064453125, 0.2998046875, 0.494140625
    ],
    "color_matrix2": [
        1.611328125, -1.025390625, -0.02734375,
        0.0166015625, 0.9423828125, 0.1103515625,
        0.0791015625, 0.126953125, 0.52734375
    ],
    "camera_calibration1": [
        0.9990234375, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.025390625
    ],
    "camera_calibration2": [
        0.9990234375, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.025390625
    ],
    "calibration_illuminant1": 21,
    "calibration_illuminant2": 17,
    "forward_matrix1": [
        0.66015625, 0.2373046875, 0.06640625,
        0.1982421875, 0.9443359375, -0.142578125,
        0.015625, -0.38671875, 1.196289062
    ],
    "forward_matrix2": [
        0.3818359375, 0.4111328125, 0.1708984375,
        -0.056640625, 1.084960938, -0.0283203125,
        -0.1396484375, -0.935546875, 1.901367188
    ],
    "shutter_speed": 0.03333333333,
    "fov": 76.0941493376305,
    "width": 4000,
    "height": 3000,
    "illum_color": D50_RAW_ILLUM,
    "ccm": D50_CCM,
    "orientation": 1
}


ABOUT_MESSAGE = (
    f'This is the user interface for the modular neural ISP ({VERSION})\n'
    'presented in the paper:\n\n'
    'Modular Neural Image Signal Processing.\n\n'
    'Developed by:\n'
    '• Denoisers and upsampler – Zhongling Wang\n'
    '• Evaluation – Ran Zhang\n'
    '• Photofinishing, detail enhancement, pipeline integration,\n'
    '  editing operators, and photo-editing tool – Mahmoud Afifi\n\n'
    'Auto White Balance based on:\n'
    '“Time-Aware Auto White Balance in Mobile Photography,” ICCV 2025\n'
    '“Learning Camera-Agnostic White Balance Preferences,” ICCVW 2025\n'
    '“Cross-Camera Convolutional Color Constancy,” ICCV 2021\n\n'
    'Embedding raw data uses:\n'
    '“Raw-JPEG Adapter: Efficient Raw Image Compression with JPEG,” arXiv 2025\n\n'
    'Image linearization uses:\n'
    '“CIE XYZ Net: Unprocessing Images for Low-Level Computer Vision Tasks,” TMAPI 2021\n\n'
    'Acknowledgments:\n'
    '• Luxi Zhao – generic denoiser development\n'
    '• Raghav Goyal – denoiser experiments\n\n'
    'Inquiries: Mahmoud Afifi (m.3afifi@gmail.com)'
)





