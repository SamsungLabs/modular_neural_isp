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

This file contains file utility functions.
"""

from typing import List, Union, Dict
import json
import numpy as np
import os


def convert_np_arrays(data: np.ndarray) -> List:
  """ Recursively converts any NumPy arrays in the data to lists."""
  if isinstance(data, np.ndarray):
    return data.tolist()

  elif isinstance(data, dict):
    return {key: convert_np_arrays(value) for key, value in data.items()}
  elif isinstance(data, list):
    return [convert_np_arrays(item) for item in data]
  return data

def write_json_file(data: Union[Dict, List], file_path: str):
  """Writes data to a JSON file."""
  with open(os.path.splitext(file_path)[0] + '.json', 'w') as f:
    json.dump(convert_np_arrays(data), f, indent=4)

def read_json_file(file_path: str):
  """Reads a JSON file and returns its contents as a dictionary."""
  with open(file_path, 'r') as file:
    data = json.load(file)
  return data