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

This script downloads the Expert C .tif images from the Adobe 5K dataset. After downloading, use Adobe
Lightroom/Photoshop to export them as full-resolution JPG images with 90% quality (make sure to export using
"sRGB IEC 61966-2-1" color space). Finally, run adobe_5k_preprocessing.py to split the dataset into training, testing,
and validation sets and extract raw images and DNG data.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# Path to filesAdobeMIT.txt
filename = 'Path/to/filesAdobeMIT.txt'
# Path to save expert c's TIF image files
output_path = 'Output/path/to/save/expert_c/TIF/images'
MAX_WORKERS = 10  # Number of concurrent threads for downloading
RETRY_COUNT = 3  # Number of retries for each file
BATCH_SIZE = 500  # Number of files per batch (for better progress tracking)


def download_file(url, filepath, retry=RETRY_COUNT):
  """Download a file synchronously with retries."""
  for attempt in range(retry):
    try:
      response = requests.get(url, timeout=60)
      if response.status_code == 200:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as file:
          file.write(response.content)
        print(f"Downloaded: {filepath}")
        return True
      else:
        print(f"Failed to download {url}. Status code: {response.status}")
    except requests.exceptions.RequestException as e:
      print(f"Error downloading {url} (attempt {attempt + 1}/{retry}): {e}")
    time.sleep(2)
  return False


def process_batch(url_base, filenames, output_path):
  """Process a batch of files using a thread pool."""
  with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    tasks = []
    for filename in filenames:
      file_url = f"{url_base}{filename}.tif"
      file_path = os.path.join(output_path, f"{filename}.tif")
      if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        continue
      tasks.append(executor.submit(download_file, file_url, file_path))

    for task in tasks:
      task.result()


def download_in_batches(url_base, filenames, output_path):
  """Download files in batches."""
  for i in range(0, len(filenames), BATCH_SIZE):
    batch = filenames[i:i + BATCH_SIZE]
    print(f"Processing batch {i // BATCH_SIZE + 1}...")
    process_batch(url_base, batch, output_path)


if __name__ == "__main__":
  with open(filename, "r") as file:
    filenames = [line.strip() for line in file]
  url_base = "https://data.csail.mit.edu/graphics/fivek/img/tiff16_c/"
  download_in_batches(url_base, filenames, output_path)
