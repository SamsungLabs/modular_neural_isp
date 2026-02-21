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

This file contains the front-end functionality of the photo-editing tool.
"""


import sys
import os
import glob
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

import tkinter as tk
import numpy as np
import PIL.Image, PIL.ImageTk
from tkinter import filedialog
from tkinter.font import Font
from typing import Optional, Union, Dict, List, Tuple
from utils.constants import *
from utils.file_utils import write_json_file, read_json_file
from utils.img_utils import (extract_image_from_dng, extract_raw_metadata, normalize_raw, demosaice,
                             extract_additional_dng_metadata, apply_exif_orientation, imresize,
                             undo_exif_orientation)
from main.pipeline import PipeLine


class PictureStyleWidget(tk.Frame):
  def __init__(
     self, parent: tk.Widget, title: str, font: tk.font.Font, colors: Dict[str, str], size: Optional[int] = 60,
     radius: Optional[int] = 4, font_sz: Optional[int] = 9) -> None:
    super().__init__(parent, bg=colors['BG'])

    self._colors = colors
    self._font_sz = font_sz
    self._font = font
    self._size = size
    self._radius = radius
    self._suspend_sync = False
    self.on_toggle = None

    # Header row: checkbox + title
    top_row = tk.Frame(self, bg=colors['BG'])
    top_row.pack(anchor='center', pady=(0, 2))

    self._enabled_var = tk.BooleanVar(value=True)
    tk.Checkbutton(top_row, variable=self._enabled_var,
                   bg=self._colors['BG'], fg=self._colors['TEXT'], selectcolor=self._colors['FG'],
                   activebackground=self._colors['BG'], activeforeground=self._colors['TEXT'],
                   command=self._toggle).pack(side='left', padx=(0, 3))

    tk.Label(top_row, text=title,
             bg=self._colors['BG'], fg=self._colors['TEXT'],
             font=(font.actual("family"), self._font_sz, 'bold')).pack(side='left')

    # Top slider (style strength)
    self._style_strength = tk.Scale(self, from_=0, to=100,
                                    orient='horizontal', showvalue=False,
                                    bg=self._colors['BG'], fg=self._colors['TEXT'],
                                    troughcolor=self._colors['FG'], highlightthickness=0,
                                    width=6, sliderlength=10, font=self._font, length=self._size - 20,
                                    command=self._on_style_strength_changed)
    self._style_strength.pack(anchor='center')

    # Mid row: left slider, canvas, right slider
    mid = tk.Frame(self, bg=self._colors['BG'])
    mid.pack()

    # Left slider (digital gain)
    self._digital_gain_strength = tk.Scale(mid, from_=100, to=0,
                                           orient='vertical', showvalue=False,
                                           bg=self._colors['BG'], fg=self._colors['TEXT'],
                                           troughcolor=self._colors['FG'], highlightthickness=0,
                                           width=6, sliderlength=10, font=self._font, length=self._size + 10)
    self._digital_gain_strength.pack(side='left', padx=1)

    self._canvas = tk.Canvas(mid, width=self._size, height=self._size, bg=self._colors['BG'],
                             highlightbackground=self._colors['TEXT'], highlightthickness=1)
    self._canvas.pack(side='left', padx=1)
    self._canvas.bind('<B1-Motion>', self._drag)

    # Right slider (gamma)
    self._gamma_strength = tk.Scale(mid, from_=100, to=0,
                                    orient='vertical', showvalue=False,
                                    bg=self._colors['BG'], fg=self._colors['TEXT'],
                                    troughcolor=self._colors['FG'], highlightthickness=0,
                                    width=6, sliderlength=10, font=self._font, length=self._size + 10)
    self._gamma_strength.pack(side='left', padx=1)

    # Bottom slider (GTM)
    self._gtm_strength = tk.Scale(self, from_=0, to=100,
                                  orient='horizontal', showvalue=False,
                                  bg=self._colors['BG'], fg=self._colors['TEXT'],
                                  troughcolor=self._colors['FG'], highlightthickness=0,
                                  width=6, sliderlength=10, font=self._font, length=self._size - 20)
    self._gtm_strength.pack(anchor='center')

    # This is to track any pending deferred callbacks (for safe cancellation)
    self._pending_after = []

    # Circle (vertical: LTM, horizontal: chroma)
    cx, cy = self._size // 2, self._size // 2
    self._circle = self._canvas.create_oval(cx - self._radius, cy - self._radius,
                                            cx + self._radius, cy + self._radius,
                                            outline=self._colors['TEXT'], width=1)
    self._apply_enabled_state(self._enabled_var.get())

  def _move_circle(self, cx: int, cy: int) -> None:
    """Moves the draggable control-circle to a new position within the canvas."""
    self._canvas.coords(self._circle, cx - self._radius, cy - self._radius,
                        cx + self._radius, cy + self._radius)

  def _set_all_sliders_immediate(self, top: int, left: int, right: int, bottom: int) -> None:
    """Sets all sliders immediately to the given values."""
    self._style_strength.set(top)
    self._digital_gain_strength.set(left)
    self._gamma_strength.set(right)
    self._gtm_strength.set(bottom)


  def _set_all_sliders_deferred(self, top: int, left: int, right: int, bottom: int) -> None:
    """Sets all sliders later (UI update deferred, cancelable)."""
    self._pending_after.append(self.after_idle(lambda: self._style_strength.set(top)))
    self._pending_after.append(self.after_idle(lambda: self._digital_gain_strength.set(left)))
    self._pending_after.append(self.after_idle(lambda: self._gamma_strength.set(right)))
    self._pending_after.append(self.after_idle(lambda: self._gtm_strength.set(bottom)))

  def set_values(self, style_strength, digital_gain, gamma, gtm, dot_x, dot_y):
    """Sets all sliders and circle to given values (used when reloading stored settings)."""
    # enables temporarily if disabled
    was_disabled = not self._enabled_var.get()
    if was_disabled:
      self._apply_enabled_state(True)

    self._style_strength.set(style_strength)
    self._digital_gain_strength.set(digital_gain)
    self._gamma_strength.set(gamma)
    self._gtm_strength.set(gtm)

    # moves circle to stored normalized location
    cx = int(dot_x / 100 * (self._size - 2 * self._radius)) + self._radius
    cy = int(dot_y / 100 * (self._size - 2 * self._radius)) + self._radius
    self._move_circle(cx, cy)

    if was_disabled:
      self._apply_enabled_state(False)


  def _cancel_pending(self) -> None:
    """Cancels any pending deferred slider updates."""
    for aid in self._pending_after:
      try:
        self.after_cancel(aid)
      except Exception:
        pass
    self._pending_after.clear()

  def _apply_enabled_state(self, enabled: bool) -> None:
    """Enables or disables the widget UI safely (no race with deferred setters)."""
    # Cancels any pending deferred updates first
    self._cancel_pending()

    if enabled:
      for s in (self._style_strength, self._gtm_strength, self._digital_gain_strength, self._gamma_strength):
        s.configure(state='normal')
      self._canvas.bind('<B1-Motion>', self._drag)
      self._canvas.configure(highlightbackground=self._colors['TEXT'])
      self._canvas.itemconfig(self._circle, outline=self._colors['TEXT'])
      self._move_circle(self._size - self._radius, self._radius)
      self._set_all_sliders_immediate(100, 100, 100, 100)

    else:
      self._set_all_sliders_immediate(0, 0, 0, 0)
      self._move_circle(self._radius, self._size - self._radius)
      self._canvas.unbind('<B1-Motion>')
      self._canvas.configure(highlightbackground=self._colors['FG'])
      self._canvas.itemconfig(self._circle, outline=self._colors['FG'])
      for s in (self._style_strength, self._gtm_strength, self._digital_gain_strength, self._gamma_strength):
        s.configure(state='disabled')

    self.update_idletasks()

  def _toggle(self):
    """Applies UI state based on checkbox toggle."""
    self._apply_enabled_state(self._enabled_var.get())
    if callable(self.on_toggle):
      self.after_idle(lambda: self.on_toggle(self))

  def _drag(self, event):
    """Drags circle inside the canvas bounds."""
    if not self._enabled_var.get():
      return
    x = min(max(event.x, self._radius), self._size - self._radius)
    y = min(max(event.y, self._radius), self._size - self._radius)
    self._move_circle(x, y)

  def _get_dot_position(self) -> Tuple[int, int]:
    """Returns normalized (x,y) dot offset in range [0,100]."""
    x1, y1, x2, y2 = self._canvas.coords(self._circle)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    norm_x = int(100 * (cx - self._radius) / (self._size - 2 * self._radius))
    norm_y = int(100 * (cy - self._radius) / (self._size - 2 * self._radius))
    return norm_x, norm_y

  def get_values(self):
    """Returns slider values + (normalized x,y)."""
    dot_x, dot_y = self._get_dot_position()
    return (
      self._style_strength.get(),
      self._digital_gain_strength.get(),
      self._gamma_strength.get(),
      self._gtm_strength.get(),
      dot_x,
      dot_y
    )

  def get_position(self) -> Tuple[int, int]:
    x1, y1, x2, y2 = self._canvas.coords(self._circle)
    cx = int((x1 + x2) // 2)
    cy = int((y1 + y2) // 2)
    return cx, cy

  def move_position(self, cx: int, cy: int):
    cx = min(max(cx, self._radius), self._size - self._radius)
    cy = min(max(cy, self._radius), self._size - self._radius)
    self._move_circle(cx, cy)


  def _on_style_strength_changed(self, value: Union[str, float]):
    """Synchronizes all style controls with the top slider (global strength)."""
    if self._suspend_sync:
      return
    try:
      val = float(value)
    except ValueError:
      return

    self._digital_gain_strength.set(val)
    self._gamma_strength.set(val)
    self._gtm_strength.set(val)
    x = int(val / 100 * (self._size - 2 * self._radius)) + self._radius
    y = int((100 - val) / 100 * (self._size - 2 * self._radius)) + self._radius
    self._move_circle(x, y)

class PhotoEditorUI:
  def __init__(self, root: tk.Tk, full_screen: Optional[bool] = False) -> None:
    """Initializes the main photo editor interface."""
    self._root = root
    self._pipeline = None
    self._reset_img_vars()
    self._as_shot_disabled = False
    self._gui_enabled = True
    self._log_message = ''
    self._is_video_sequence = False
    self._auto_exp_locked = False
    self._reset_temporal_data()
    self._root.title('Photo Editor UI')
    self._root.configure(bg=BACKGROUND_COLOR)
    self._gpu_available = (
       torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()))
    self._force_rendering = False
    self._current_images = []
    self._is_disabled = True
    self._editing_settings = None

    if full_screen:
      self._root.attributes('-fullscreen', True)  # true fullscreen
      self._root.bind('<Escape>', lambda e: self._root.attributes('-fullscreen', False))
    else:
      self._root.geometry('1200x700')
    available_fonts = list(tk.font.families())
    if 'Segoe UI Light' in available_fonts:
      self._font_family = 'Segoe UI Light'
    else:
      self._font_family = 'Arial'  # safe fallback

    self._text_font = Font(family=self._font_family, size=10)

    self._theme = {
      'BG': BACKGROUND_COLOR,
      'TEXT': TEXT_COLOR,
      'FG': FOREGROUND_COLOR
    }

    self._zero_state = [False] * NUMBER_OF_STYLES
    self._create_menu()
    self._create_layout()
    self._create_status_bar()
    self._set_ui_enabled(enabled=False, exclude_file_ops=True)

    if self._device_setting.get().lower() == 'gpu':
      if torch.cuda.is_available():
        self._device = torch.device('cuda')
      elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        self._device = torch.device('mps')
      else:
        self._device = torch.device('cpu')
    else:
      self._device = torch.device('cpu')

    self._update_status('Loading AI models ...\n', erase_after=False)

    if int(self._raw_jpeg_quality.get()) == DEFAULT_RAW_JPEG_QUALITY:
      raw_jpeg_model = PATH_TO_RAW_JPEG_MODEL
    else:
      raw_jpeg_model = os.path.join(IO_MODELS_FOLDER, JPEG_RAW_MODELS[int(self._raw_jpeg_quality.get())])


    self._pipeline = PipeLine(running_device=self._device,
                              photofinishing_model_path=PATH_TO_PHOTOFINISHING_DEFAULT_MODEL,
                              photofinishing_style_model_paths=PATHS_TO_PHOTOFINISHING_MODELS,
                              generic_denoising_model_path=PATH_TO_GENERIC_DENOISER_MODEL,
                              denoising_model_path=PATH_TO_S24_DENOISER_MODEL,
                              enhancement_model_path=PATH_TO_ENHANCEMENT_MODEL,
                              s24_awb_model_path=PATH_TO_S24_AWB_MODEL,
                              cc_awb_model_path=PATH_TO_GENERIC_AWB_MODEL,
                              post_awb_model_path=PATH_TO_POST_AWB_MODEL,
                              raw_jpeg_adapter_model_path= raw_jpeg_model,
                              linearization_model_path=PATH_TO_LINEARIZATION_MODEL,
                              log=self._log_message)
    self._synch_log(get=True)

    self._root.after(PAUSE_TIME, lambda: self._status_var.set(''))


  def _synch_log(self, get: Optional[bool]=False) -> None:
    if self._pipeline is not None:
      if get:
        self._log_message = self._pipeline.get_log()
      else:
        self._pipeline.set_log(self._log_message)

  def _create_status_bar(self) -> None:
    """Creates bottom status bar for brief processing messages."""
    self._status_var = tk.StringVar(value='')
    status_bar = tk.Label(self._root, textvariable=self._status_var, bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
                          font=self._text_font, anchor='w')
    status_bar.pack(side='bottom', fill='x')

    # Image index status
    self._img_idx_var = tk.StringVar(value='')
    img_idx_label = tk.Label(self._root, textvariable=self._img_idx_var,
                             bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
                             font=self._text_font, anchor='e')
    img_idx_label.place(relx=0.98, rely=1.0, anchor='se')

  def _create_menu(self) -> None:
    """Creates the top menu bar and all sub-menus."""
    menu_bar = tk.Menu(self._root, bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                       tearoff=False, activebackground=FOREGROUND_COLOR, activeforeground=TEXT_COLOR)
    self._root.config(menu=menu_bar)

    # File Menu
    file_menu = tk.Menu(menu_bar, tearoff=0, bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                        activebackground=FOREGROUND_COLOR, activeforeground=TEXT_COLOR)
    file_menu.add_command(label='Open file', command=self._open_file)
    file_menu.add_command(label='Open folder', command=self._open_folder)
    file_menu.add_command(label='Export picture style', command=self._export_style)
    file_menu.add_command(label='Import picture style', command=self._import_style)
    file_menu.add_command(label='Save', command=self._save_file)
    file_menu.add_command(label='Save all', command=self._save_all_files)
    file_menu.add_separator()
    file_menu.add_command(label='Exit', command=self._root.quit)

    # Settings Menu
    settings_menu = tk.Menu(menu_bar, tearoff=0, bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                            activebackground=FOREGROUND_COLOR, activeforeground=TEXT_COLOR)

    # Auto White Balance submenu
    self._awb_setting = tk.StringVar(value='camera')  # default
    awb_menu = tk.Menu(settings_menu, tearoff=0, bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                       activebackground=FOREGROUND_COLOR, activeforeground=TEXT_COLOR)
    awb_menu.add_radiobutton(label='Camera-specific (if available)', variable=self._awb_setting, value='camera')
    awb_menu.add_radiobutton(label='Always use cross-camera', variable=self._awb_setting, value='generic')
    settings_menu.add_cascade(label='Auto White Balance', menu=awb_menu)

    # Denoising submenu
    self._denoise_setting = tk.StringVar(value='camera')  # default
    denoise_menu = tk.Menu(settings_menu, tearoff=0, bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                           activebackground=FOREGROUND_COLOR, activeforeground=TEXT_COLOR)
    denoise_menu.add_radiobutton(label='Camera-specific (if available)', variable=self._denoise_setting, value='camera')
    denoise_menu.add_radiobutton(label='Always use generic denoiser', variable=self._denoise_setting, value='generic')
    settings_menu.add_cascade(label='Denoising', menu=denoise_menu)

    # Raw-JPEG quality submenu
    self._raw_jpeg_quality = tk.StringVar(value=f'{DEFAULT_RAW_JPEG_QUALITY}')  # default
    raw_jpeg_menu = tk.Menu(settings_menu, tearoff=0, bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                            activebackground=FOREGROUND_COLOR, activeforeground=TEXT_COLOR)
    raw_jpeg_menu.add_radiobutton(label='Quality 75', variable=self._raw_jpeg_quality, value='75')
    raw_jpeg_menu.add_radiobutton(label='Quality 95', variable=self._raw_jpeg_quality, value='95')
    settings_menu.add_cascade(label='Raw-JPEG Quality', menu=raw_jpeg_menu)

    # sRGB-JPEG quality submenu
    self._srgb_jpeg_quality = tk.StringVar(value=f'{DEFAULT_SRGB_JPEG_QUALITY}')  # default
    srgb_jpeg_menu = tk.Menu(settings_menu, tearoff=0, bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                             activebackground=FOREGROUND_COLOR, activeforeground=TEXT_COLOR)
    srgb_jpeg_menu.add_radiobutton(label='Quality 75', variable=self._srgb_jpeg_quality, value='75')
    srgb_jpeg_menu.add_radiobutton(label='Quality 85', variable=self._srgb_jpeg_quality, value='85')
    srgb_jpeg_menu.add_radiobutton(label='Quality 95', variable=self._srgb_jpeg_quality, value='95')
    srgb_jpeg_menu.add_radiobutton(label='Quality 100', variable=self._srgb_jpeg_quality, value='100')
    settings_menu.add_cascade(label='sRGB-JPEG Quality', menu=srgb_jpeg_menu)

    # Local Tone Mapping submenu
    self._ltm_mode = tk.StringVar(value='enhanced')  # default

    # Choose default iterations index based on device availability
    default_iter_idx = (
      DEFAULT_BILATERAL_SOLVER_ITERS_GPU_IDX
      if self._gpu_available and torch.cuda.is_available()
      else DEFAULT_BILATERAL_SOLVER_ITERS_CPU_IDX
    )
    default_iters = BILATERAL_SOLVER_ITERS_OPTIONS[default_iter_idx]

    self._ltm_iters = tk.IntVar(value=default_iters)

    ltm_menu = tk.Menu(settings_menu, tearoff=0,
                       bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                       activebackground=FOREGROUND_COLOR, activeforeground=TEXT_COLOR)

    # Mode radio buttons
    ltm_menu.add_radiobutton(
      label='Enhanced', variable=self._ltm_mode, value='enhanced',
      command=self._on_ltm_settings_changed
    )
    ltm_menu.add_radiobutton(
      label='Standard', variable=self._ltm_mode, value='standard',
      command=self._on_ltm_settings_changed
    )

    ltm_menu.add_separator()

    # Iterations submenu
    iters_menu = tk.Menu(ltm_menu, tearoff=0,
                         bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                         activebackground=FOREGROUND_COLOR, activeforeground=TEXT_COLOR)

    for it in BILATERAL_SOLVER_ITERS_OPTIONS:
      iters_menu.add_radiobutton(
        label=f'{it} iterations',
        variable=self._ltm_iters,
        value=it,
        command=self._on_ltm_settings_changed
      )

    ltm_menu.add_cascade(label='Iterations', menu=iters_menu)

    settings_menu.add_cascade(label='Local Tone Mapping', menu=ltm_menu)

    # Disable iteration options if mode is standard
    def _update_ltm_iters_state(*_):
      state = 'normal' if self._ltm_mode.get() == 'enhanced' else 'disabled'
      end = iters_menu.index('end')
      if end is not None:
        for i in range(end + 1):
          iters_menu.entryconfig(i, state=state)

    self._ltm_mode.trace_add('write', _update_ltm_iters_state)
    _update_ltm_iters_state()


    # Device submenu
    self._device_setting = tk.StringVar(value='gpu' if self._gpu_available else 'cpu')
    device_menu = tk.Menu(settings_menu, tearoff=0, bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                          activebackground=FOREGROUND_COLOR, activeforeground=TEXT_COLOR)
    device_menu.add_radiobutton(label='GPU', variable=self._device_setting, value='gpu',
                                state=('normal' if self._gpu_available else 'disabled'))
    device_menu.add_radiobutton(label='CPU', variable=self._device_setting, value='cpu')
    settings_menu.add_cascade(label='Device', menu=device_menu)

    self._device_menu = device_menu

    # Auto Orientation submenu
    self._auto_orientation = tk.BooleanVar(value=True)  # default: ON
    auto_orient_menu = tk.Menu(settings_menu, tearoff=0,
                               bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                               activebackground=FOREGROUND_COLOR, activeforeground=TEXT_COLOR)
    auto_orient_menu.add_checkbutton(label='Enable Auto Orientation', variable=self._auto_orientation,
                                     onvalue=True, offvalue=False, command=self._on_auto_orientation_changed)
    settings_menu.add_cascade(label='Auto Orientation', menu=auto_orient_menu)

    # Preview Resolution submenu
    self._preview_mode = tk.StringVar(value='draft')  # default: draft/fast
    preview_menu = tk.Menu(settings_menu, tearoff=0,
                           bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                           activebackground=FOREGROUND_COLOR, activeforeground=TEXT_COLOR)

    preview_menu.add_radiobutton(
      label='Draft Preview (Fast)',
      variable=self._preview_mode,
      value='draft',
      command=self._on_preview_mode_changed
    )
    preview_menu.add_radiobutton(
      label='Full-Resolution Preview (Accurate)',
      variable=self._preview_mode,
      value='full',
      command=self._on_preview_mode_changed
    )
    settings_menu.add_cascade(label='Preview Resolution', menu=preview_menu)

    # Info Menu
    info_menu = tk.Menu(menu_bar, tearoff=0, bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                        activebackground=FOREGROUND_COLOR, activeforeground=TEXT_COLOR)
    info_menu.add_command(label='About', command=self._show_about)

    # Attach menus
    menu_bar.add_cascade(label='File', menu=file_menu)
    menu_bar.add_cascade(label='Settings', menu=settings_menu)
    menu_bar.add_command(label='Log', command=self._show_log)
    menu_bar.add_cascade(label='Info', menu=info_menu)

    # Attach callbacks for menu setting changes
    self._awb_setting.trace_add('write', lambda *args: self._on_awb_setting_changed())
    self._denoise_setting.trace_add('write', lambda *args: self._on_denoise_setting_changed())
    self._device_setting.trace_add('write', lambda *args: self._on_device_setting_changed())
    self._raw_jpeg_quality.trace_add('write', lambda *args: self._on_raw_jpeg_quality_changed())
    self._srgb_jpeg_quality.trace_add('write', lambda *args: self._on_jpeg_srgb_quality_changed())

  def _show_log(self) -> None:
    """Shows a pop-up window containing console log messages."""
    log_win = tk.Toplevel(self._root)
    log_win.title('Log Messages')
    log_win.configure(bg=BACKGROUND_COLOR)
    width, height = 600, 400
    log_win.geometry(f'{width}x{height}')

    log_win.update_idletasks()
    screen_width = log_win.winfo_screenwidth()
    screen_height = log_win.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    log_win.geometry(f'{width}x{height}+{x}+{y}')

    text_area = tk.Text(log_win, bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR, font=('Courier New', 10), wrap='word')
    text_area.pack(expand=True, fill='both', padx=10, pady=10)
    text_area.insert('end', self._log_message)
    text_area.config(state='disabled')


  def _show_about(self) -> None:
    """Displays the application information dialog."""
    about_win = tk.Toplevel(self._root)
    about_win.title('About')
    about_win.configure(bg=BACKGROUND_COLOR)
    about_win.resizable(False, False)

    win_w, win_h = 450, 650
    x = self._root.winfo_x() + (self._root.winfo_width() - win_w) // 2
    y = self._root.winfo_y() + (self._root.winfo_height() - win_h) // 2
    about_win.geometry(f'{win_w}x{win_h}+{x}+{y}')

    frame = tk.Frame(about_win, bg=SECOND_BACKGROUND_COLOR, bd=2, relief='groove')
    frame.pack(expand=True, fill='both', padx=10, pady=10)

    label = tk.Label(frame,
                     text=ABOUT_MESSAGE,
                     bg=SECOND_BACKGROUND_COLOR,
                     fg=TEXT_COLOR, font=self._text_font, justify='left', wraplength=360)
    label.pack(expand=True, fill='both', padx=10, pady=10)

    ok_btn = tk.Button(frame, text='OK', bg=FOREGROUND_COLOR, fg=TEXT_COLOR, font=self._text_font, relief='flat',
                       command=about_win.destroy)
    ok_btn.pack(pady=6)

  def _create_layout(self):
    """Builds the left control panel and main image preview area."""
    self._root.bind("<Configure>", self._on_window_resize)

    # Left panel
    left_panel = tk.Frame(self._root, bg=BACKGROUND_COLOR)
    left_panel.pack(side='left', fill='y', padx=8, pady=8)

    top_row = tk.Frame(left_panel, bg=BACKGROUND_COLOR)
    top_row.pack(pady=8, anchor='center')

    # Browse group
    browse_group = tk.LabelFrame(top_row, text='Browse', font=self._text_font,
                                 fg=TEXT_COLOR, bg=BACKGROUND_COLOR,
                                 bd=2, relief='groove', labelanchor='n')
    browse_group.pack(side='left', padx=6)

    browse_inner = tk.Frame(browse_group, bg=BACKGROUND_COLOR)
    browse_inner.pack(padx=6, pady=7)

    btn_prev = tk.Button(browse_inner, text='<<', bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                         font=self._text_font, relief='flat', width=2)
    btn_prev.pack(side='left', padx=4)

    self._entry_index = tk.Entry(browse_inner, width=4, font=self._text_font,
                                 bg=FOREGROUND_COLOR, fg=TEXT_COLOR, justify='center')
    self._entry_index.pack(side='left', padx=4)

    btn_next = tk.Button(browse_inner, text='>>', bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                         font=self._text_font, relief='flat', width=2)
    btn_next.pack(side='left', padx=4)

    self._btn_prev = btn_prev
    self._btn_next = btn_next

    btn_prev.configure(command=self._on_prev_image)
    btn_next.configure(command=self._on_next_image)
    self._entry_index.bind('<Return>', self._on_goto_image)

    # Exposure group (narrower)
    exposure_group = tk.LabelFrame(top_row, text='Exposure', font=self._text_font,
                                   fg=TEXT_COLOR, bg=BACKGROUND_COLOR,
                                   bd=2, relief='groove', labelanchor='n')
    exposure_group.pack(side='left', padx=4)

    exposure_inner = tk.Frame(exposure_group, bg=BACKGROUND_COLOR)
    exposure_inner.pack(padx=6, pady=4)

    # Auto button
    self._auto_exp_enabled = False
    btn_auto_exp = tk.Button(exposure_inner, text='Auto',
                             bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                             font=self._text_font, relief='sunken' if self._auto_exp_enabled else 'flat',
                             width=4,
                             command=lambda: self._toggle_auto_exposure(btn_auto_exp))
    btn_auto_exp.pack(side='left', padx=4)
    self._btn_auto_exp = btn_auto_exp

    # EV label + slider
    tk.Label(exposure_inner, text='EV', bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
             font=self._text_font).pack(side='left', padx=2)

    self._ev_slider = tk.Scale(exposure_inner, from_=EV_MIN, to=EV_MAX,
                               orient='horizontal', resolution=0.1,
                               bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
                               troughcolor=FOREGROUND_COLOR, highlightthickness=0,
                               font=self._text_font, width=12, sliderlength=12, length=60)
    self._ev_slider.set(0)
    self._ev_slider.pack(side='left', padx=4)
    self._ev_slider.bind("<ButtonRelease-1>", lambda e: self._on_exposure_change())

    # White Balance group
    wb_group = tk.LabelFrame(left_panel, text='White Balance', font=self._text_font,
                             fg=TEXT_COLOR, bg=BACKGROUND_COLOR, bd=2, relief='groove', labelanchor='n')
    wb_group.pack(pady=4, anchor='center')

    self._wb_var = tk.StringVar(value='as_shot')

    wb_radio_frame = tk.Frame(wb_group, bg=BACKGROUND_COLOR)
    wb_radio_frame.pack(pady=2)

    self._wb_radio_buttons = []

    for text, value in [
      ('As Shot', 'as_shot'),
      ('Auto', 'auto'),
      ('Auto (Neutral)', 'auto_neutral'),
      ('Custom', 'custom')
    ]:
      rb = tk.Radiobutton(wb_radio_frame, text=text, variable=self._wb_var, value=value,
                          bg=BACKGROUND_COLOR, fg=TEXT_COLOR, selectcolor=FOREGROUND_COLOR,
                          font=self._text_font, command=self._on_white_balance_change)
      rb.pack(side='left', padx=1.5)
      self._wb_radio_buttons.append(rb)

    # Sliders
    wb_slider_frame = tk.Frame(wb_group, bg=BACKGROUND_COLOR)
    wb_slider_frame.pack(pady=2)

    tk.Label(wb_slider_frame, text='CCT', bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
             font=self._text_font).pack(side='left', padx=2)
    self._cct_slider = tk.Scale(wb_slider_frame, from_=2000, to=10000, orient='horizontal',
                                bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
                                troughcolor=FOREGROUND_COLOR, highlightthickness=0,
                                font=self._text_font, width=12, sliderlength=12, length=60)
    self._cct_slider.pack(side='left', padx=2)
    self._cct_slider.set(DEFAULT_CCT)
    self._cct_slider.bind("<ButtonRelease-1>", self._set_custom_wb)

    tk.Label(wb_slider_frame, text='Tint', bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
             font=self._text_font).pack(side='left', padx=2)
    self._tint_slider = tk.Scale(wb_slider_frame, from_=TINT_MIN, to=TINT_MAX, orient='horizontal',
                                 bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
                                 troughcolor=FOREGROUND_COLOR, highlightthickness=0,
                                 font=self._text_font, width=12, sliderlength=12, length=60)
    self._tint_slider.pack(side='left', padx=2)
    self._tint_slider.set(DEFAULT_TINT)
    self._tint_slider.bind("<ButtonRelease-1>", self._set_custom_wb)

    # Details group (denoising + enhancement controls)
    details_group = tk.LabelFrame(left_panel, text='Details', font=self._text_font,
                                  fg=TEXT_COLOR, bg=BACKGROUND_COLOR,
                                  bd=2, relief='groove', labelanchor='n')
    details_group.pack(pady=2, anchor='center')

    details_frame = tk.Frame(details_group, bg=BACKGROUND_COLOR)
    details_frame.pack(padx=8, pady=4)

    # Four horizontal sliders in one row (compact)
    detail_sliders = [
      ('Denoising', DEFAULT_DENOISING, DENOISING_MIN, DENOISING_MAX),
      ('Chroma', DEFAULT_CHROMA_DENOISING, CHROMA_DENOISING_MIN, CHROMA_DENOISING_MAX),
      ('Luma', DEFAULT_LUMA_DENOISING, LUMA_DENOISING_MIN, LUMA_DENOISING_MAX),
      ('Enhance', DEFAULT_ENHANCEMENT, ENHANCEMENT_MIN, ENHANCEMENT_MAX),
    ]

    self._details_sliders = {}

    for i, (label, default_value, min_v, max_v) in enumerate(detail_sliders):
      col = i
      frame = tk.Frame(details_frame, bg=BACKGROUND_COLOR)
      frame.grid(row=0, column=col, padx=6)

      tk.Label(frame, text=label,
               bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
               font=(self._font_family, 8, 'bold')).pack(anchor='center')

      slider = tk.Scale(frame, from_=min_v, to=max_v,
                        orient='horizontal', showvalue=False,
                        bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
                        troughcolor=FOREGROUND_COLOR, highlightthickness=0,
                        width=10, sliderlength=10,
                        font=self._text_font, length=60)
      slider.set(default_value)
      slider.bind("<ButtonRelease-1>", lambda e, name=label: self._on_details_changed(name))
      slider.pack(anchor='center')

      self._details_sliders[label] = slider

    # Picture Styles group
    styles_group = tk.LabelFrame(left_panel, font=self._text_font,
                                 fg=TEXT_COLOR, bg=BACKGROUND_COLOR,
                                 bd=2, relief='groove', labelanchor='n')
    styles_group.pack(pady=2, anchor='center')

    # Title + help icon row
    title_row = tk.Frame(styles_group, bg=BACKGROUND_COLOR)
    title_row.pack(anchor='center', pady=(2, 0))

    tk.Label(title_row, text='Picture Styles',
             bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
             font=self._text_font).pack(side='left')

    help_btn = tk.Button(title_row, text='\u2753', relief='flat',
                         bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                         font=(self._font_family, 8, 'bold'),
                         command=self._show_style_help,
                         width=2)
    help_btn.pack(side='left', padx=(4, 0))

    styles_frame = tk.Frame(styles_group, bg=BACKGROUND_COLOR)
    styles_frame.pack(padx=6, pady=4)

    self._style_widgets = []
    for i in range(NUMBER_OF_STYLES):
      w = PictureStyleWidget(styles_frame, STYLE_NAMES[i], self._text_font, self._theme, size=60)
      w.grid(row=i // 3, column=i % 3, padx=5, pady=6)
      self._wire_style_callbacks(w, i)
      self._style_widgets.append(w)
    # Only enables the first (default) style; disable the rest
    for i, w in enumerate(self._style_widgets):
      if i == 0:
        w._enabled_var.set(True)
        w._apply_enabled_state(True)
      else:
        w._enabled_var.set(False)
        w._apply_enabled_state(False)

    self._zero_state = [False] * len(self._style_widgets)

    # Global Operations and Editing
    adjust_edit_row = tk.Frame(left_panel, bg=BACKGROUND_COLOR)
    adjust_edit_row.pack(pady=2, anchor='center')

    # Global operations group
    actions_group = tk.LabelFrame(adjust_edit_row, text='Global Ops', font=self._text_font, fg=TEXT_COLOR,
                                  bg=BACKGROUND_COLOR, bd=2, relief='groove', labelanchor='n')
    actions_group.pack(side='left', padx=2)

    actions_frame = tk.Frame(actions_group, bg=BACKGROUND_COLOR)
    actions_frame.pack(padx=4, pady=10)

    row1 = tk.Frame(actions_frame, bg=BACKGROUND_COLOR)
    row1.pack(anchor='center', pady=(0, 4))

    btn_copy_all = tk.Button(row1, text='Copy', bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                             font=self._text_font, relief='flat', width=4, command=self._copy_all_settings)
    btn_copy_all.pack(side='left', padx=2)

    btn_paste_all = tk.Button(row1, text='Paste', bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR, font=self._text_font,
                              relief='flat', width=4, command=self._paste_all_settings, state='disabled')
    btn_paste_all.pack(side='left', padx=2)
    self._paste_all_btn = btn_paste_all
    self._copied_settings = None

    btn_reset_all = tk.Button(actions_frame, text='Reset',
                              bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
                              font=self._text_font, relief='flat', width=10, command=self._reset_all_settings)

    btn_reset_all.pack(anchor='center', pady=(0, 2))

    # Editing group
    editing_group = tk.LabelFrame(adjust_edit_row, text='Editing', font=self._text_font,
                                  fg=TEXT_COLOR, bg=BACKGROUND_COLOR,
                                  bd=2, relief='groove', labelanchor='n')
    editing_group.pack(side='left', padx=4)

    editing_frame = tk.Frame(editing_group, bg=BACKGROUND_COLOR)
    editing_frame.pack(padx=3, pady=4)

    # Editing sliders
    controls = [
      ('Highlights', DEFAULT_HIGHLIGHTS, HIGHLIGHTS_MIN, HIGHLIGHTS_MAX),
      ('Shadows', DEFAULT_SHADOWS, SHADOWS_MIN, SHADOWS_MAX),
      ('Sharpening', DEFAULT_SHARPENING, SHARPENING_MIN, SHARPENING_MAX),
      ('Saturation', DEFAULT_SATURATION, SATURATION_MIN, SATURATION_MAX),
      ('Vibrance', DEFAULT_VIBRANCE, VIBRANCE_MIN, VIBRANCE_MAX),
      ('Contrast', DEFAULT_CONTRAST, CONTRAST_MIN, CONTRAST_MAX),
    ]

    self._editing_sliders = {}

    for i, (label, default_value, mn, mx) in enumerate(controls):
      r, c = divmod(i, 3)
      frame = tk.Frame(editing_frame, bg=BACKGROUND_COLOR)
      frame.grid(row=r, column=c, padx=2, pady=3)

      tk.Label(frame, text=label,
               bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
               font=(self._font_family, 8, 'bold')).pack(anchor='center')

      slider = tk.Scale(frame, from_=mn, to=mx,
                        orient='horizontal', showvalue=False,
                        bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
                        troughcolor=FOREGROUND_COLOR, highlightthickness=0,
                        width=10, sliderlength=10, font=self._text_font,
                        length=60)

      slider.set(default_value)
      slider.bind("<ButtonRelease-1>", lambda e, name=label: self._on_editing_changed(name))

      slider.pack(anchor='center')
      self._editing_sliders[label] = slider

    # Image display area
    image_outer = tk.Frame(self._root, bg=BACKGROUND_COLOR)
    image_outer.pack(expand=True, fill='both', padx=10, pady=10)

    border_frame = tk.Frame(image_outer, bg=SECOND_BACKGROUND_COLOR, bd=1, relief='sunken', highlightthickness=0)
    border_frame.pack(expand=True, fill='both')

    self._canvas = tk.Canvas(border_frame, bg=BACKGROUND_COLOR, highlightthickness=0, bd=0)
    self._canvas.pack(expand=True, fill='both', padx=1, pady=1)


  def _open_file(self) -> None:
    """Opens a single image file."""
    file_path = filedialog.askopenfilename(
      title='Open Image',
      filetypes=[('Image files', '*.jpg *.jpeg *.png *.dng')]
    )
    if not file_path:
      return

    self._update_status(f'Loading {os.path.basename(file_path)}')
    self._is_video_sequence = False
    self._auto_exp_locked = False
    self._current_images = [file_path]
    self._saved_settings = [None]
    self._current_index = 0

    self._update_image_index_status()
    self._global_style_applied = False
    self._global_style_settings = None
    self._on_browse()
    self._set_ui_enabled(True)
    self._is_disabled = False
    self._update_image_index_status()
    self._update_browse_buttons_state()

  def _open_folder(self) -> None:
    """Opens an image folder."""
    folder_path = filedialog.askdirectory(title='Select Folder')
    if not folder_path:
      return

    self._is_video_sequence = False
    self._auto_exp_locked = False
    patterns = ('*.jpg', '*.jpeg', '*.png', '*.dng')
    files = []
    for p in patterns:
      files.extend(glob.glob(os.path.join(folder_path, p)))
    files.sort()

    if not files:
      self._show_message_dialog(
        'No Images Found',
        'No supported images were found in the selected folder.\n\n'
        'Supported formats: JPG, PNG, and DNG.',
        level='warning', long_message=True
      )
      self._update_status('No Images Found', show_message=False)
      return

    self._current_images = files
    self._saved_settings = [None] * len(files)
    self._current_index = 0

    self._update_status(f'Selected: {os.path.basename(folder_path)} ({len(files)} images)')

    self._update_browse_entry(str(self._current_index + 1))
    self._update_image_index_status()
    self._global_style_applied = False
    self._global_style_settings = None
    self._on_browse()
    self._set_ui_enabled(True)
    self._is_disabled = False
    self._update_image_index_status()
    self._update_browse_buttons_state()

  def _reset_temporal_data(self):
    """Resets all temporal data."""
    if self._is_video_sequence:
      default_value = [None] * TEMPORAL_WINDOW_SIZE
    else:
      default_value = None
    self._pre_illum = default_value
    self._pre_ccm = default_value
    self._pre_gain = default_value
    self._pre_gtm = default_value
    self._pre_chroma_lut = default_value
    self._pre_gamma = default_value

  def _save_file(self) -> None:
    """Saves the currently processed image."""
    try:
      if not hasattr(self, '_current_images') or not self._current_images:
        self._show_message_dialog('Save Failed', 'No image loaded to save.', level='error')
        return

      self._save_current_settings()

      save_path = filedialog.asksaveasfilename(title='Save Image', defaultextension='.jpg',
                                               filetypes=[('JPEG', '*.jpg'), ('All Files', '*.*')]
                                               )
      if not save_path:
        return

      self._update_status('Saving current image...', erase_after=False)
      self._set_ui_enabled(False)
      self._is_disabled = True

      self._force_rendering = False
      self._load_image(self._current_images[self._current_index])
      self._render_image(preview=False)

      raw_img = getattr(self, '_raw_img', None)
      srgb_img = getattr(self, '_srgb_img', None)
      metadata = getattr(self, '_metadata', None)

      # Safety check
      if raw_img is None or srgb_img is None:
        self._show_message_dialog('Save Failed', 'No processed image found.', level='error')
        return

      # Save via pipeline
      self._synch_log()
      editing_settings = {
        'ev': self._ev_slider.get(),
        'auto_exp': self._auto_exp_enabled,
        'wb_mode': self._wb_var.get(),
        'cct': self._cct_slider.get(),
        'tint': self._tint_slider.get(),
        'details': {k: s.get() for k, s in self._details_sliders.items()},
        'editing': {k: s.get() for k, s in self._editing_sliders.items()},
        #'styles': [w.get_values() for w in self._style_widgets],
        'styles': [
          {
            'enabled': w._enabled_var.get(),
            'values': w.get_values(),
          }
          for w in self._style_widgets
        ],
      }
      save_path = PhotoEditorUI._get_unique_save_path(save_path)
      self._pipeline.save_image(srgb=srgb_img, raw=raw_img, metadata=metadata, editing_settings=editing_settings,
                                output_path=save_path, log_messages=True,
                                report_time=True, srgb_jpeg_quality=int(self._srgb_jpeg_quality.get()),
                                raw_jpeg_quality=int(self._raw_jpeg_quality.get()))
      self._synch_log(get=True)
      self._update_status(f'Saved: {os.path.basename(save_path)}')

    except Exception as e:
      self._show_message_dialog('Save Failed', f'Error while saving image:\n{e}', level='error',
                                long_message=True)
    finally:
      self._set_ui_enabled(True)
      self._is_disabled = False
      self._reset_img_vars(exclude_raw=True, exclude_metadata=True)
      if self._raw_img is not None:
        self._raw_img = self._pipeline.to_np(self._raw_img)
      self._render_image(preview=True)
      self._force_rendering = True

  def _save_all_files(self) -> None:
    """Saves all images in the currently loaded folder."""
    current_idx = self._current_index
    img_path = self._current_images[self._current_index]
    filename = os.path.basename(img_path)
    self._force_rendering = False
    self._save_current_settings()
    try:
      if not hasattr(self, '_current_images') or not self._current_images:
        self._show_message_dialog('Save Failed', 'No images loaded to save.', level='error')
        return

      save_folder = filedialog.askdirectory(title='Select Output Folder')
      if not save_folder:
        self._force_rendering = True
        return

      self._reset_temporal_data()
      os.makedirs(save_folder, exist_ok=True)
      total = len(self._current_images)
      self._update_status(f'Saving all {total} images...', erase_after=False)
      self._set_ui_enabled(False)
      self._is_disabled = True

      for idx in range(total):
        self._current_index = idx
        img_path = self._current_images[self._current_index]
        filename = os.path.basename(img_path)
        output_path = os.path.join(save_folder, os.path.splitext(filename)[0] + '.jpg')
        self._update_status(f'Saving {self._current_index + 1}/{total}: {os.path.splitext(filename)[0]}',
                            erase_after=False)
        style = self._saved_settings[self._current_index]
        self._reset_img_vars()
        self._load_image(img_path)
        if style is None:
          style = self._use_default_or_imported_style()
        self._render_image(preview=False, style=style)

        raw_img = getattr(self, '_raw_img', None)
        srgb_img = getattr(self, '_srgb_img', None)
        metadata = getattr(self, '_metadata', None)

        if raw_img is None or srgb_img is None:
          self._show_message_dialog('Skipped', f'No processed data for {filename}.', level='warning')
          continue
        self._synch_log()
        output_path = PhotoEditorUI._get_unique_save_path(output_path)
        self._pipeline.save_image(srgb=srgb_img, raw=raw_img, metadata=metadata, output_path=output_path,
                                  log_messages=True, report_time=True,
                                  editing_settings=style,
                                  srgb_jpeg_quality=int(self._srgb_jpeg_quality.get()),
                                  raw_jpeg_quality=int(self._raw_jpeg_quality.get()))

      self._synch_log(get=True)
      self._update_status(f'All {total} images saved to {save_folder}.')
      self._show_message_dialog('Save Complete', f'Successfully saved {total} images to: {save_folder}',
                                level='info', long_message=True)

    except Exception as e:
      self._update_status(f'Skipped {filename} (no processed data)', erase_after=False)
    finally:
      self._current_index = current_idx
      self._set_ui_enabled(True)
      self._is_disabled = False
      self._load_image(self._current_images[self._current_index])
      style = self._saved_settings[self._current_index]
      if style is None:
        style = self._use_default_or_imported_style()
      self._render_image(preview=True, style=style)
      self._force_rendering = True


  def _toggle_auto_exposure(self, btn: tk.Button) -> None:
    """Toggles auto exposure button state."""

    if getattr(self, '_auto_exp_locked', False):
      return

    self._auto_exp_enabled = not self._auto_exp_enabled

    if self._auto_exp_enabled:
      btn.config(relief='sunken')
      self._update_status('Auto exposure ON', show_message=False)
    else:
      btn.config(relief='flat')
      self._update_status('Auto exposure OFF', show_message=False)

    if self._force_rendering:
      self._render_image(preview=True)

  def _copy_all_settings(self):
    """Copies all editing parameters from current UI controls."""
    self._copied_settings = self._copy_current_style_settings()
    self._paste_all_btn.config(state='normal')
    self._update_status('Settings copied')

  def _paste_all_settings(self):
    """Pastes previously copied editing parameters to current UI controls."""
    if not self._copied_settings:
      return

    self._force_rendering = False
    self._reset_img_vars(exclude_raw=True, exclude_metadata=True)
    self._ev_slider.set(self._copied_settings['ev'])
    self._wb_var.set(self._copied_settings['wb_mode'])
    self._cct_slider.set(self._copied_settings['cct'])
    self._tint_slider.set(self._copied_settings['tint'])

    if self._copied_settings.get('auto_exp', False) != self._auto_exp_enabled:
      self._toggle_auto_exposure(self._btn_auto_exp)

    for k, v in self._copied_settings['details'].items():
      if k in self._details_sliders:
        self._details_sliders[k].set(v)

    for k, v in self._copied_settings['editing'].items():
      if k in self._editing_sliders:
        self._editing_sliders[k].set(v)

    for i, w in enumerate(self._style_widgets):
      if self._copied_settings['styles'][i]['enabled']:
        if not w._enabled_var.get():
          w._enabled_var.set(True)
          w._apply_enabled_state(True)
        t, l, r, b, dx, dy = self._copied_settings['styles'][i]['values']
        w._set_all_sliders_immediate(t, l, r, b)
        dx = max(0, min(100, dx))
        dy = max(0, min(100, dy))
        cx = int(dx / 100 * (w._size - 2 * w._radius)) + w._radius
        cy = int(dy / 100 * (w._size - 2 * w._radius)) + w._radius
        w._move_circle(cx, cy)
      else:
        if w._enabled_var.get():
          w._enabled_var.set(False)
          w._apply_enabled_state(False)
    self._update_status('Settings pasted')


    self._render_image(preview=True)
    self._force_rendering = True

  def _reset_all_settings(self, with_rendering: Optional[bool]=True):
    """Resets all editing settings to defaults and enables only Style 0 (default picture style)."""
    try:
      self._force_rendering = False

      self._saved_settings[self._current_index] = self._copy_default_style_settings()
      self._apply_style(self._saved_settings[self._current_index])

      if with_rendering:
        self._reset_img_vars(exclude_raw=True, exclude_metadata=True)
        self._render_image(preview=True)
        self._update_status('All settings reset to defaults')

    except Exception as e:
      self._show_message_dialog('Reset Failed', f'Error while resetting: {e}', level='error')
    finally:
      self._force_rendering = True


  def _reset_img_vars(self, exclude_raw: Optional[bool]=False,
                      exclude_denoised_raw: Optional[bool] = False,
                      exclude_lsrgb: Optional[bool]=False,
                      exclude_srgb: Optional[bool] = False,
                      exclude_gain_params: Optional[bool] = False,
                      exclude_gtm_params: Optional[bool] = False,
                      exclude_ltm_params: Optional[bool] = False,
                      exclude_chroma_params: Optional[bool] = False,
                      exclude_gamma_params: Optional[bool] = False,
                      exclude_metadata: Optional[bool]=False):
    if not exclude_raw:
      self._raw_img = None
    if not exclude_denoised_raw:
      self._denoised_raw_img = None
    if not exclude_lsrgb:
      self._lsrgb_img = None
    if not exclude_srgb:
      self._srgb_img = None
    if not exclude_gain_params:
      self._gain_params = [None] * NUMBER_OF_STYLES
    if not exclude_gtm_params:
      self._gtm_params = [None] * NUMBER_OF_STYLES
    if not exclude_ltm_params:
      self._ltm_params = [None] * NUMBER_OF_STYLES
    if not exclude_chroma_params:
      self._chroma_params = [None] * NUMBER_OF_STYLES
    if not exclude_gamma_params:
      self._gamma_params = [None] * NUMBER_OF_STYLES
    if not exclude_metadata:
      self._metadata = None
      self._current_image_is_taken_by_the_specific_camera = False

  def _show_style_help(self) -> None:
    """Shows guidance on using picture style controls."""
    help_win = tk.Toplevel(self._root)
    help_win.title('Picture Style Controls')
    help_win.configure(bg=BACKGROUND_COLOR)
    help_win.resizable(False, False)

    width, height = 380, 230
    help_win.geometry(f'{width}x{height}')

    help_win.update_idletasks()
    screen_width = help_win.winfo_screenwidth()
    screen_height = help_win.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    help_win.geometry(f'{width}x{height}+{x}+{y}')

    tk.Label(help_win, text='Adjusting Picture Style:',
             bg=BACKGROUND_COLOR, fg=TEXT_COLOR,
             font=(self._font_family, 10, 'bold')).pack(pady=(10, 2))

    tk.Label(
      help_win,
      text=(
        '- Top slider: Style global strength\n'
        '- Left slider: Digital gain (brightness boost)\n'
        '- Right slider: Gamma adjustment\n'
        '- Bottom slider: Global tone mapping\n'
        '- Drag the dot inside the box:\n'
        '    * Up-Down: Local tone mapping\n'
        '    * Right-left: Chroma adjustment\n'
        'Enable checkbox allows each picture style to take effect.'
      ), bg=BACKGROUND_COLOR, fg=TEXT_COLOR, justify='left', anchor='w', font=(self._font_family, 9), wraplength=320
    ).pack(padx=12, pady=6)

    tk.Button(help_win, text="Close",
              bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
              font=self._text_font, relief='flat',
              command=help_win.destroy).pack(pady=6)

  def _set_custom_wb(self, event=None) -> None:
    """Switches WB mode to 'custom' when CCT or Tint sliders are changed."""
    if self._wb_var.get() != 'custom':
      self._wb_var.set('custom')
      self._update_status('White Balance: Custom', show_message=False)
    self._on_white_balance_change()

  def _update_image_index_status(self) -> None:
    """Updates bottom-right image index display."""
    if hasattr(self, '_current_images'):
      total = len(self._current_images)
      idx = self._current_index + 1 if hasattr(self, '_current_index') else 1
      self._update_browse_entry(str(self._current_index + 1))
      if total > 1:
        self._img_idx_var.set(f"{idx}/{total}")
      else:
        self._img_idx_var.set('')
    else:
      self._img_idx_var.set('')

  def _set_ui_enabled(self, enabled: bool, exclude_file_ops: Optional[bool] = False) -> None:
    """Globally enables/disables UI controls."""
    self._gui_enabled = 'normal' if enabled else 'disabled'

    # Recursively toggle widgets (non-menu)
    def toggle(widget: tk.Widget):
      # Skip PictureStyleWidget children—they manage enable/disable internally
      if isinstance(widget, PictureStyleWidget):
        return

      try:
        widget.configure(state=self._gui_enabled)
      except tk.TclError:
        pass

      for child in widget.winfo_children():
        toggle(child)

    toggle(self._root)

    try:
      self._device_menu.entryconfig('GPU', state='normal' if self._gpu_available else 'disabled')
    except:
      pass

    try:
      menu_bar = self._root.nametowidget(self._root['menu'])
      # fetch File submenu
      file_menu = menu_bar.nametowidget(menu_bar.entrycget('File', 'menu'))
      if enabled:
        # enables menubar items
        end = menu_bar.index('end')
        if end is not None:
          for i in range(end + 1):
            try:
              menu_bar.entryconfig(i, state='normal')
            except:
              pass

        # enables all inside File menu
        end = file_menu.index('end')
        if end is not None:
          for i in range(end + 1):
            file_menu.entryconfig(i, state='normal')

      # disable mode
      else:
        # Menubar remains clickable
        menu_bar.entryconfig('File', state='normal')
        menu_bar.entryconfig('Log', state='normal')
        menu_bar.entryconfig('Info', state='normal')

        # Disables all menu items first
        end = menu_bar.index('end')
        if end is not None:
          for i in range(end + 1):
            try:
              menu_bar.entryconfig(i, state='disabled')
            except:
              pass

        # Re-enables File + Log + Info cascades
        menu_bar.entryconfig('File', state='normal')
        menu_bar.entryconfig('Log', state='normal')
        menu_bar.entryconfig('Info', state='normal')

        # chooses allowed inside File menu
        if exclude_file_ops:
          allowed = {'Open file', 'Open folder', 'Exit'}
        else:
          allowed = {'Exit'}

        end_file = file_menu.index('end')
        if end_file is not None:
          for i in range(end_file + 1):
            label = file_menu.entrycget(i, 'label')
            # enables only allowed
            file_menu.entryconfig(i, state='normal' if label in allowed else 'disabled')

    except Exception:
      pass

    # Keeps 'As Shot' disabled if the current image forbids it
    if self._as_shot_disabled and hasattr(self, '_wb_radio_buttons'):
      try:
        self._wb_radio_buttons[0].config(state='disabled')
      except Exception:
        pass

    # Keeps entry index textbox disabled if it is a video sequence.
    if self._is_video_sequence and hasattr(self, '_entry_index'):
      try:
        self._entry_index.config(state='disabled')
      except Exception:
        pass

    # Keeps auto exposure disabled if is locked.
    if self._auto_exp_locked:
      try:
        self._btn_auto_exp.config(state='disabled')
      except Exception:
        pass

    self._root.update_idletasks()

  def _on_style_toggled(self, idx: int):
    try:
      self._root.update_idletasks()
    except Exception:
      pass
    self._on_style_changed(idx)

  def _wire_style_callbacks(self, w: PictureStyleWidget, idx: int) -> None:
    w.on_toggle = lambda _w, k=idx: self._on_style_toggled(k)
    for s in (w._style_strength, w._digital_gain_strength, w._gamma_strength, w._gtm_strength):
      s.bind("<ButtonRelease-1>", lambda _e, k=idx: self._on_style_changed(k))
    try:
      w._canvas.bind('<ButtonRelease-1>', lambda e, k=idx: self._on_style_changed(k), add='+')

    except Exception:
      pass

  def _attach_menu_callbacks(self):
    self._awb_setting.trace_add('write', lambda *args: self._on_awb_setting_changed())
    self._denoise_setting.trace_add('write', lambda *args: self._on_denoise_setting_changed())
    self._device_setting.trace_add('write', lambda *args: self._on_device_setting_changed())
    self._raw_jpeg_quality.trace_add('write', lambda *args: self._on_raw_jpeg_quality_changed())
    self._srgb_jpeg_quality.trace_add('write', lambda *args: self._on_jpeg_srgb_quality_changed())

  def _update_browse_entry(self, text: str):
    """Updates browse entry textbox."""
    self._entry_index.delete(0, 'end')
    self._entry_index.insert(0, text)

  def _export_style(self) -> None:
    """Exports the current UI state (exposure, WB, details, editing, styles) to a JSON file."""
    file_path = filedialog.asksaveasfilename(
      title='Export Picture Style',
      defaultextension='.json',
      filetypes=[('JSON', '*.json')]
    )
    if not file_path:
      return

    try:
      data = {
        # Exposure
        'ev': self._ev_slider.get(),
        'auto_exposure': self._auto_exp_enabled,

        # White balance
        'wb_mode': self._wb_var.get(),
        'cct': self._cct_slider.get(),
        'tint': self._tint_slider.get(),

        # Details sliders
        'details': {k: s.get() for k, s in self._details_sliders.items()},

        # Editing sliders
        'editing': {k: s.get() for k, s in self._editing_sliders.items()},

        # Picture styles
        'styles': [
          {
            'enabled': w._enabled_var.get(),
            'values': w.get_values(),  # (top, left, right, bottom, x, y)
          }
          for w in self._style_widgets
        ],
      }

      write_json_file(data, file_path)
      self._update_status('Picture style exported successfully')

    except Exception as e:
      self._show_message_dialog('Export Failed', f'Error: {e}', level='error')
      self._update_status(f'Export Failed - error: {e}', show_message=False)

  def _import_style(self) -> None:
    """Imports a saved picture style JSON and applies all settings to the current UI."""
    file_path = filedialog.askopenfilename(
      title='Import Picture Style',
      filetypes=[('JSON', '*.json')]
    )
    if not file_path:
      return

    self._reset_img_vars(exclude_raw=True, exclude_metadata=True)
    self._force_rendering = False
    try:
      data = read_json_file(file_path)
    except Exception as e:
      self._show_message_dialog('Import Failed', f'Error: {e}', level='error', long_message=True)
      self._update_status(f'Import Failed - error: {e}', show_message=False)
      self._force_rendering = True
      self._render_image(preview=True)
      return

    try:
      self._update_editing_settings(data)

      self._update_status('Picture style imported successfully')

      self._render_image(preview=True)
      self._force_rendering = True

      # Asks to apply style to all images (if multiple):
      apply_all = False
      if hasattr(self, '_current_images') and len(self._current_images) > 1:
        apply_all, is_video_seq = self._ask_yes_no_dialog(
          'Apply Style to All Images',
          'Multiple images are currently loaded.\n\n'
          'Do you want to apply this imported style to all images in the set?',
          checkbox_label='This is a video sequence (apply temporal consistency)'
        )
        self._is_video_sequence = is_video_seq
        if apply_all:
          self._saved_settings = [None] * len(self._current_images)
          if self._is_video_sequence:
            self._auto_exp_locked = True
            if self._auto_exp_enabled:
              self._set_auto_exposure_btn(value=False)
              self._show_message_dialog(
                'Auto Exposure Disabled',
                'Auto Exposure has been disabled for video sequence processing. '
                'Style applied to all frames.', level='info', long_message=True)
            else:
              self._show_message_dialog('Global Style Applied', 'Style applied to all frames.', level='info')
            self._btn_auto_exp.config(state='disabled')
            if 'auto_exposure' in data:
              data['auto_exposure'] = False
            if self._current_index:
              self._current_index = 0
              self._update_image_index_status()
              self._update_browse_buttons_state()
              self._load_image(self._current_images[self._current_index])
              self._render_image(preview=True)
            self._reset_temporal_data()
          else:
            self._auto_exp_locked = False
            self._btn_auto_exp.config(state='normal')
            self._set_auto_exposure_btn(data['auto_exposure'])
            self._update_browse_buttons_state()
            self._update_browse_entry(str(self._current_index + 1))
            self._show_message_dialog(
              'Global Style Applied',
              'Style applied to all images in the set.',
              level='info'
            )
        else:
          self._auto_exp_locked = False
          self._btn_auto_exp.config(state='normal')
          self._set_auto_exposure_btn(data['auto_exposure'])
          self._is_video_sequence = False
          self._update_browse_buttons_state()
          self._update_browse_entry(str(self._current_index + 1))
      # Stores imported style info
      self._global_style_applied = apply_all
      self._global_style_settings = self._copy_current_style_settings() if apply_all else None

    except Exception as e:
      self._show_message_dialog('Import Failed', f'Error: {e}', level='error')
      self._update_status(f'Import Failed - error: {e}', show_message=False)
      self._force_rendering = True
      self._render_image(preview=True)

  def _update_editing_settings(self, data: Dict, rendering: Optional[bool]=True):
    """Updates editing settings."""
    self._force_rendering = False
    # Auto exposure
    if 'auto_exposure' in data:
      desired = bool(data['auto_exposure'])
      # ensures we have the button handle
      if hasattr(self, '_btn_auto_exp'):
        if desired != self._auto_exp_enabled:
          # uses your existing toggle to keep UI (relief/status) consistent
          self._toggle_auto_exposure(self._btn_auto_exp)
      else:
        # fallback: just sets the flag if button isn't created for some reason
        self._auto_exp_enabled = desired
    # Exposure
    if 'ev' in data:
      self._ev_slider.set(data['ev'])
    if 'auto_exposure' in data:
      self._auto_exp_enabled = bool(data['auto_exposure'])
      self._update_status('Auto exposure ON' if self._auto_exp_enabled else 'Auto exposure OFF', show_message=False)

    # White balance
    if 'wb_mode' in data:
      if getattr(self, '_as_shot_disabled', False) and data['wb_mode'] == 'as_shot':
        wb_mode = 'auto'  # fallback if As Shot is disabled
      else:
        wb_mode = data['wb_mode']
      self._wb_var.set(wb_mode)
    if 'cct' in data:
      self._cct_slider.set(data['cct'])
    if 'tint' in data:
      self._tint_slider.set(data['tint'])
    self._on_white_balance_change()

    # Details
    if 'details' in data:
      for k, v in data['details'].items():
        if k in self._details_sliders:
          self._details_sliders[k].set(v)

    # Editing
    if 'editing' in data:
      for k, v in data['editing'].items():
        if k in self._editing_sliders:
          self._editing_sliders[k].set(v)

    # Styles
    if 'styles' in data:
      for i, style in enumerate(data['styles']):
        if i >= len(self._style_widgets):
          break
        w = self._style_widgets[i]

        enabled = bool(style.get('enabled', False))
        w._enabled_var.set(enabled)
        w._apply_enabled_state(enabled)

        vals = style.get('values')
        if isinstance(vals, (list, tuple)):
          if len(vals) >= 6:
            t, l, r, b, cx, cy = vals[:6]
          elif len(vals) == 4:
            t, l, r, b = vals
            cx, cy = (w._size - w._radius, w._radius) if enabled else (w._radius, w._size - w._radius)
          else:
            t = l = r = b = None
            cx = cy = None
        else:
          t = l = r = b = None
          cx = cy = None

        # applies defaults if values missing, based on enable state
        if t is None:
          if enabled:
            t = l = r = b = 100
            cx, cy = (w._size - w._radius, w._radius)
          else:
            t = l = r = b = 0
            cx, cy = (w._radius, w._size - w._radius)

        w._set_all_sliders_immediate(t, l, r, b)
        if cx is not None and cy is not None:
          cx = int(cx / 100 * (w._size - 2 * w._radius)) + w._radius
          cy = int(cy / 100 * (w._size - 2 * w._radius)) + w._radius
          w._move_circle(cx, cy)
    if rendering:
      self._render_image(preview=True)
    self._force_rendering = True

  def _copy_current_style_settings(self):
    """Captures current style parameters for global reuse."""
    return {
      'ev': self._ev_slider.get(),
      'wb_mode': self._wb_var.get(),
      'cct': self._cct_slider.get(),
      'tint': self._tint_slider.get(),
      'auto_exp': self._auto_exp_enabled,
      'details': {k: s.get() for k, s in self._details_sliders.items()},
      'editing': {k: s.get() for k, s in self._editing_sliders.items()},
      'styles': [{'enabled': w._enabled_var.get(), 'values': w.get_values(),} for w in self._style_widgets],
    }

  def _on_awb_setting_changed(self):
    self._update_status(f'White-balance mode: {self._awb_setting.get()}', show_message=False)
    if self._force_rendering and hasattr(self, '_current_image_is_taken_by_the_specific_camera'):
      if self._current_image_is_taken_by_the_specific_camera and (
         self._wb_var.get() == 'auto' or self._wb_var.get() == 'auto_neutral'):
        self._reset_img_vars(exclude_raw=True, exclude_metadata=True, exclude_denoised_raw=True)
        self._render_image(preview=True)

  def _on_denoise_setting_changed(self):
    self._update_status(f'Denoiser: {self._denoise_setting.get()}', show_message=False)
    if self._force_rendering and hasattr(self, '_current_image_is_taken_by_the_specific_camera'):
      if self._current_image_is_taken_by_the_specific_camera:
        self._reset_img_vars(exclude_raw=True, exclude_metadata=True)
        self._render_image(preview=True)

  def _on_window_resize(self, event):
    """Re-renders image when window or display area is resized."""
    if hasattr(self, '_srgb_img') and isinstance(self._srgb_img, np.ndarray):
      if hasattr(self, '_resize_after_id'):
        self._root.after_cancel(self._resize_after_id)
      self._resize_after_id = self._root.after(200, lambda: self._show_image())

  def _on_device_setting_changed(self):
    new_device = self._device_setting.get()
    self._update_status(f'Updating device to {new_device}...', erase_after=False)
    if new_device.lower() == 'gpu':
      if torch.cuda.is_available():
        self._device = torch.device('cuda')
      elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        self._device = torch.device('mps')
      else:
        self._device = torch.device('cpu')
    else:
      self._device = torch.device('cpu')

    # Chooses correct index based on new setting
    default_iter_idx = (
        DEFAULT_BILATERAL_SOLVER_ITERS_GPU_IDX
        if new_device == 'gpu' and torch.cuda.is_available()
        else DEFAULT_BILATERAL_SOLVER_ITERS_CPU_IDX
    )
    new_default_iters = BILATERAL_SOLVER_ITERS_OPTIONS[default_iter_idx]

    if self._ltm_iters.get() != new_default_iters:
        self._ltm_iters.set(new_default_iters)

    # Updates device of pipeline networks:
    self._pipeline.update_device(self._device)
    self._root.after(PAUSE_TIME, lambda: self._status_var.set(''))

    if self._raw_img is not None:
      self._raw_img = self._raw_img.to(device=self._device)

    if self._denoised_raw_img is not None:
      self._denoised_raw_img = self._denoised_raw_img.to(device=self._device)

    if self._lsrgb_img is not None:
      self._lsrgb_img = self._lsrgb_img.to(device=self._device)

    if self._pre_illum is not None:
      for i in range(TEMPORAL_WINDOW_SIZE):
        if self._pre_illum[i] is not None and isinstance(self._pre_illum[i], torch.Tensor):
          self._pre_illum[i] = self._pre_illum[i].to(device=self._device)

    if self._pre_ccm is not None:
      for i in range(TEMPORAL_WINDOW_SIZE):
        if self._pre_ccm[i] is not None and isinstance(self._pre_ccm[i], torch.Tensor):
          self._pre_ccm[i] = self._pre_ccm[i].to(device=self._device)

    if self._pre_gain is not None:
      for i in range(TEMPORAL_WINDOW_SIZE):
        if self._pre_gain[i] is not None and isinstance(self._pre_gain[i], torch.Tensor):
          self._pre_gain[i] = self._pre_gain[i].to(device=self._device)

    if self._pre_gtm is not None:
      for i in range(TEMPORAL_WINDOW_SIZE):
        if self._pre_gtm[i] is not None and isinstance(self._pre_gtm[i], torch.Tensor):
          self._pre_gtm[i] = self._pre_gtm[i].to(device=self._device)

    if self._pre_chroma_lut is not None:
      for i in range(TEMPORAL_WINDOW_SIZE):
        if self._pre_chroma_lut[i] is not None and isinstance(self._pre_chroma_lut[i], torch.Tensor):
          self._pre_chroma_lut[i] = self._pre_chroma_lut[i].to(device=self._device)

    if self._pre_gamma is not None:
      for i in range(TEMPORAL_WINDOW_SIZE):
        if self._pre_gamma[i] is not None and isinstance(self._pre_gamma[i], torch.Tensor):
          self._pre_gamma[i] = self._pre_gamma[i].to(device=self._device)


    for i in range(NUMBER_OF_STYLES):
      if self._gain_params[i] is not None and isinstance(self._gain_params[i], torch.Tensor):
        self._gain_params[i] = self._gain_params[i].to(device=self._device)

      if self._gtm_params[i] is not None and isinstance(self._gtm_params[i], torch.Tensor):
        self._gtm_params[i] = self._gtm_params[i].to(device=self._device)

      if self._ltm_params[i] is not None and isinstance(self._ltm_params[i], torch.Tensor):
        self._ltm_params[i] = self._ltm_params[i].to(device=self._device)

      if self._chroma_params[i] is not None and isinstance(self._chroma_params[i], torch.Tensor):
        self._chroma_params[i] = self._chroma_params[i].to(device=self._device)

      if self._gamma_params[i] is not None and isinstance(self._gamma_params[i], torch.Tensor):
        self._gamma_params[i] = self._gamma_params[i].to(device=self._device)

  def _on_preview_mode_changed(self):
    """Handles change between draft and full-resolution preview."""
    mode = self._preview_mode.get()
    if mode == 'draft':
      self._update_status('Preview mode: Draft (fast low-resolution preview)', erase_after=False)
      if isinstance(self._raw_img, torch.Tensor):
        self._raw_img = self._pipeline.to_np(self._raw_img)
        self._reset_img_vars(exclude_raw=True, exclude_metadata=True)
    else:
      self._update_status('Preview mode: Full-resolution (accurate)', erase_after=False)
      self._load_image(self._current_file_name)

    self._render_image(preview=True)

  def _on_raw_jpeg_quality_changed(self):
    self._update_status(f'Raw-JPEG quality: {self._raw_jpeg_quality.get()}', show_message=False)
    if self._pipeline is not None:
      self._update_status('Updating Raw-JPEG model ...', erase_after=False)
      self._pipeline.update_model(
        raw_jpeg_model_path=os.path.join(IO_MODELS_FOLDER, JPEG_RAW_MODELS[int(self._raw_jpeg_quality.get())]))
      self._root.after(PAUSE_TIME, lambda: self._status_var.set(''))

  def _on_jpeg_srgb_quality_changed(self):
    self._update_status(f'sRGB-JPEG quality: {self._srgb_jpeg_quality.get()}', show_message=False)

  def _on_browse(self, rendering: Optional[bool]=True):
    file_name = self._current_images[self._current_index]
    self._current_file_name = file_name
    self._update_status(f'Browse updated -- current file name: {file_name}', show_message=False)
    if hasattr(self, '_entry_index') and hasattr(self, '_current_index'):
      self._update_browse_entry(str(self._current_index + 1))
      self._update_image_index_status()
    self._update_browse_buttons_state()
    self._load_image(file_name)
    saved = self._saved_settings[self._current_index]
    if self._editing_settings is not None and saved is None:
      if self._is_disabled:
        self._set_ui_enabled(True)
      self._update_editing_settings(self._editing_settings)
      if self._is_disabled:
        self._set_ui_enabled(False)
    else:
      self._apply_style(saved)
    if rendering:
      self._render_image(preview=True)
    self._force_rendering = True


  def _is_camera_specific(self) -> None:
    """Checks if current image is taken by the specific camera (in our tool, it is S24 main camera)."""
    self._current_image_is_taken_by_the_specific_camera = PipeLine.is_s24_camera(self._metadata)

  def _use_default_or_imported_style(self):
    """Determines which style to apply when no explicit settings are saved."""
    # Case 1: imported style applied to all
    if getattr(self, '_global_style_applied', False) and getattr(self, '_global_style_settings', None):
      return self._global_style_settings
    # Case 2: Image has editing settings saved.
    if self._editing_settings is not None:
      return self._editing_settings
    # Case 3: fallback to default settings
    return self._copy_default_style_settings()



  def _apply_style(self, style_data: Optional[Dict]=None, rendering: Optional[bool]=True):
    """Applies a given style dict (saved, imported, or default) to the current image."""
    try:
      self._force_rendering = False
      if style_data is None:
        # Choose between imported or default
        style_data = self._use_default_or_imported_style()


      # Exposure
      if 'ev' in style_data:
        self._ev_slider.set(style_data['ev'])


      # White Balance
      if 'wb_mode' in style_data:
        self._wb_var.set(style_data['wb_mode'])
      if 'cct' in style_data:
        self._cct_slider.set(style_data['cct'])
      if 'tint' in style_data:
        self._tint_slider.set(style_data['tint'])
      self._on_white_balance_change()

      # Auto Exposure
      desired_auto_exp = bool(style_data.get('auto_exp', False))
      self._set_auto_exposure_btn(desired_auto_exp)

      # Details
      if 'details' in style_data:
        for k, v in style_data['details'].items():
          if k in self._details_sliders:
            self._details_sliders[k].set(v)

      # Editing
      if 'editing' in style_data:
        for k, v in style_data['editing'].items():
          if k in self._editing_sliders:
            self._editing_sliders[k].set(v)

      # Picture Styles
      if 'styles' in style_data:
        for i, w in enumerate(self._style_widgets):
          if i >= len(style_data['styles']):
            break

          vals = style_data['styles'][i]['values']
          enabled = style_data['styles'][i]['enabled'] if i < len(style_data['styles']) else False

          if len(vals) >= 6:
            t, l, r, b, dx, dy = vals[:6]
          else:
            t = l = r = b = dx = dy = 0

          if w._enabled_var.get() != enabled:
            w._enabled_var.set(enabled)
            w._apply_enabled_state(enabled)
          else:
            w._apply_enabled_state(enabled)

          if enabled:
            w._suspend_sync = True
            try:
              w._digital_gain_strength.set(l)
              w._gamma_strength.set(r)
              w._gtm_strength.set(b)
              w._style_strength.set(t)
              dx = max(0, min(100, dx))
              dy = max(0, min(100, dy))
              cx = int(dx / 100 * (w._size - 2 * w._radius)) + w._radius
              cy = int(dy / 100 * (w._size - 2 * w._radius)) + w._radius
              w._move_circle(cx, cy)
            finally:
              w._suspend_sync = False
      if rendering:
        self._update_status('Style applied', show_message=True)
        self._render_image(preview=True)
      self._force_rendering = True

    except Exception as e:
      self._show_message_dialog('Apply Style Failed', f'Error: {e}', level='error')


  def _load_image(self, input_file: str) -> None:
    """Loads image."""
    self._reset_img_vars()
    self._as_shot_disabled = False
    if input_file.lower().endswith('.dng'):
      self._metadata = extract_raw_metadata(input_file)
      self._metadata.update(extract_additional_dng_metadata(input_file))
      self._is_camera_specific()
      self._raw_img = extract_image_from_dng(input_file)
      self._raw_img = normalize_raw(img=self._raw_img, black_level=self._metadata['black_level'],
                              white_level=self._metadata['white_level']).astype(np.float32)
      self._original_sz = max(self._raw_img.shape)
      self._editing_settings = None
      if not (self._raw_img.shape[-1] == 4 and len(self._raw_img.shape) == 3):
        try:
          self._update_status('Demosaicing...', show_message=True, erase_after=False)
          self._raw_img = demosaice(self._raw_img, self._metadata['pattern'], tile_mode=True)
        except Exception:
          self._update_status('Error: Unsupported bayer pattern.', show_message=True, erase_after=False)
          self._reset_img_vars()
          self._show_image()
      else:
        self._raw_img = self._raw_img[..., :3]
      self._enable_as_shot_option()
    elif (
       input_file.lower().endswith('.png') or input_file.lower().endswith('.jpg') or
       input_file.lower().endswith('.jpeg')):
      self._synch_log()
      outputs = self._pipeline.read_image(input_file, log_messages=True, report_time=True)
      self._synch_log(get=True)
      self._raw_img = outputs['raw']
      self._original_sz = max(self._raw_img.shape)
      self._metadata = outputs['metadata']
      self._editing_settings = outputs['editing_settings']
      if self._metadata.get('model') == 'Synthetic' and self._metadata.get('make') == 'None':
        self._disable_as_shot_option()
        self._current_image_is_taken_by_the_specific_camera = False
      else:
        self._is_camera_specific()
        self._enable_as_shot_option()
    else:
      self._update_status('Error: Unsupported file format.', show_message=True, erase_after=False)
      self._reset_img_vars()


  def _set_auto_exposure_btn(self, value: bool):
    """Sets auto exposure btn value manually."""
    if value != self._auto_exp_enabled:
      self._auto_exp_enabled = value
      self._btn_auto_exp.config(relief='sunken' if value else 'flat')
      self._update_status(f'Auto exposure {"ON" if value else "OFF"}', show_message=False)

  def _on_exposure_change(self, *args):
    self._update_status('Exposure value changed', show_message=False)
    self._reset_img_vars(exclude_raw=True, exclude_metadata=True)
    if self._force_rendering:
      self._render_image(preview=True)

  def _on_white_balance_change(self, *args):
    self._update_status('White balance changed', show_message=False)
    self._reset_img_vars(exclude_raw=True, exclude_metadata=True, exclude_denoised_raw=True)
    if self._force_rendering:
      self._render_image(preview=True)

  def _disable_as_shot_option(self):
    """Disables 'As Shot' white balance option and updates selection if needed."""
    self._as_shot_disabled = True
    try:
      # Disables As Shot
      self._wb_radio_buttons[0].config(state='disabled')
      # If it was selected, switch to Auto
      if self._wb_var.get() == 'as_shot':
        self._wb_var.set('auto')
        pre_force_rendering = self._force_rendering
        self._force_rendering = False
        self._on_white_balance_change()
        self._force_rendering = pre_force_rendering
    except Exception:
      pass

  def _enable_as_shot_option(self):
    """Re-enables 'As Shot' white balance option."""
    self._as_shot_disabled = False
    try:
      self._wb_radio_buttons[0].config(state='normal')
    except Exception:
      pass

  def _on_style_changed(self, style_idx: int):
    """Handles style editing."""
    w = self._style_widgets[style_idx]
    t, l, r, b, dx, dy = w.get_values()
    circle_zero = (dx <= 5.0 or dy >= 95.0)
    is_zero_now = any(v <= 5.0 for v in (t, l, r, b)) or circle_zero
    was_zero = self._zero_state[style_idx]
    if is_zero_now != was_zero:
      self._zero_state[style_idx] = is_zero_now
      self._on_from_or_to_zero(style_idx)

    self._update_status(
      f'Style {style_idx} changed: {(t, l, r, b, dx, dy)}', show_message=False)
    if self._force_rendering:
      self._render_image(preview=True)

  def _reset_saved_style_params(self, style_idx: int):
    """Resets pre-computed style parameters."""
    #print(f'Resetting style values of {style_idx}')
    self._gain_params[style_idx] = None
    self._gtm_params[style_idx] = None
    self._ltm_params[style_idx] = None
    self._chroma_params[style_idx] = None
    self._gamma_params[style_idx] = None

  def _on_from_or_to_zero(self, style_idx: int):
    """Handles transitions to/from zero state for a style."""
    self._reset_saved_style_params(style_idx)

  def _on_details_changed(self, label: str):
    self._update_status(f'Details changed: {label} - {self._details_sliders[label].get()}', show_message=False)
    self._reset_img_vars(exclude_raw=True, exclude_metadata=True, exclude_denoised_raw=True)
    if self._force_rendering:
      self._render_image(preview=True)

  def _on_editing_changed(self, name: str):
    self._update_status(f'Editing changed: {name} - {self._editing_sliders[name].get()}', show_message=False)
    self._reset_img_vars(exclude_raw=True, exclude_denoised_raw=True, exclude_metadata=True, exclude_lsrgb=True)
    if self._force_rendering:
      self._render_image(preview=True)

  def _on_ltm_settings_changed(self, *args):
    self._update_status(f'LTM Mode: {self._ltm_mode.get()} - Iterations: {self._ltm_iters.get()}',
                        show_message=False)
    self._reset_img_vars(exclude_raw=True, exclude_denoised_raw=True, exclude_metadata=True, exclude_lsrgb=True,
                         exclude_gain_params=True, exclude_gtm_params=True)
    if self._force_rendering:
      self._render_image(preview=True)

  def _update_status(self, message: str, show_message: Optional[bool]=True, erase_after: Optional[bool]=True):
    self._log_message += f'\n{message}'
    self._synch_log()
    if show_message:
      self._status_var.set(message)
      if erase_after:
        self._root.after(PAUSE_TIME, lambda: self._status_var.set(''))

  def _show_message_dialog(self, title: str, message: str, level: Optional[str] = 'info',
                           long_message: Optional[bool]=False) -> None:
    """Shows a themed dialog. level: 'info', 'warning', or 'error'"""
    win = tk.Toplevel(self._root)
    win.title(title)
    win.configure(bg=BACKGROUND_COLOR)
    win.resizable(False, False)

    accent_color = {'info': TEXT_COLOR, 'warning': WARNING_COLOR, 'error': ERROR_COLOR}.get(level, TEXT_COLOR)

    frame = tk.Frame(win, bg=SECOND_BACKGROUND_COLOR, bd=2, relief='groove')
    frame.pack(expand=True, fill='both', padx=10, pady=10)

    tk.Label(frame, text=message, bg=SECOND_BACKGROUND_COLOR, fg=accent_color, font=self._text_font, justify='left',
             wraplength=360, anchor='w').pack(padx=15, pady=(10, 12))

    tk.Button(frame, text='OK', bg=FOREGROUND_COLOR, fg=TEXT_COLOR, font=self._text_font, relief='flat', width=10,
              command=win.destroy).pack(pady=(0, 8))
    h = 150 if long_message else 100
    self._center_popup(win, 450, h)

  def _ask_yes_no_dialog(self, title: str, message: str, checkbox_label: Optional[str] = None) -> Tuple[bool, bool]:
    """Shows a themed Yes/No dialog (optionally with a checkbox)."""
    win = tk.Toplevel(self._root)
    win.title(title)
    win.configure(bg=BACKGROUND_COLOR)
    win.resizable(False, False)

    frame = tk.Frame(win, bg=SECOND_BACKGROUND_COLOR, bd=2, relief='groove')
    frame.pack(expand=True, fill='both', padx=10, pady=10)

    tk.Label(
      frame, text=message, bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
      font=self._text_font, justify='left', wraplength=360, anchor='w').pack(padx=15, pady=(10, 12))

    # Optional checkbox
    checkbox_var = tk.BooleanVar(value=False)
    if checkbox_label:
      cb = tk.Checkbutton(
        frame, text=checkbox_label, variable=checkbox_var,
        bg=SECOND_BACKGROUND_COLOR, fg=TEXT_COLOR,
        selectcolor=FOREGROUND_COLOR, font=self._text_font,
        activebackground=SECOND_BACKGROUND_COLOR, activeforeground=TEXT_COLOR
      )
      cb.pack(pady=(0, 10))

    choice = {'yes': False}

    def on_yes():
      choice['yes'] = True
      win.destroy()

    def on_no():
      choice['yes'] = False
      win.destroy()

    btn_frame = tk.Frame(frame, bg=SECOND_BACKGROUND_COLOR)
    btn_frame.pack(pady=(0, 8))

    tk.Button(btn_frame, text='Yes', bg=FOREGROUND_COLOR, fg=TEXT_COLOR,
              font=self._text_font, relief='flat', width=8, command=on_yes).pack(side='left', padx=5)
    tk.Button(btn_frame, text='No', bg=FOREGROUND_COLOR, fg=TEXT_COLOR,
              font=self._text_font, relief='flat', width=8, command=on_no).pack(side='left', padx=5)

    self._center_popup(win, 500, 200 if checkbox_label else 160)
    win.wait_window()

    return choice['yes'], checkbox_var.get()

  def _center_popup(self, win: tk.Toplevel, w: int, h: int):
    """Centers a popup window relative to the main app."""
    x = self._root.winfo_x() + (self._root.winfo_width() - w) // 2
    y = self._root.winfo_y() + (self._root.winfo_height() - h) // 2
    win.geometry(f'{w}x{h}+{x}+{y}')


  def _on_auto_orientation_changed(self):
    """Handles toggling of automatic EXIF orientation correction."""
    state = 'ON' if self._auto_orientation.get() else 'OFF'
    self._update_status(f'Auto Orientation {state}', show_message=False)
    if self._srgb_img is not None and self._metadata is not None:
      if self._auto_orientation.get():
        self._srgb_img = apply_exif_orientation(self._srgb_img, self._metadata['orientation'])
      else:
        self._srgb_img = undo_exif_orientation(self._srgb_img, self._metadata['orientation'])
    self._show_image()

  @staticmethod
  def _are_settings_equal(a, b):
    """Checks if two style dictionaries are equivalent (within numeric tolerance)."""
    if not a or not b:
      return False
    try:
      def equal_dict(d1, d2):
        if d1.keys() != d2.keys():
          return False
        for k in d1:
          if k.lower() in ('cct', 'tint'):
            continue
          v1, v2 = d1[k], d2[k]
          if isinstance(v1, dict) and isinstance(v2, dict):
            if not equal_dict(v1, v2):
              return False
          elif isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
            if len(v1) != len(v2):
              return False
            for x, y in zip(v1, v2):
              if isinstance(x, (float, int)) and isinstance(y, (float, int)):
                if abs(x - y) > 1e-5:
                  return False
              elif x != y:
                return False
          elif isinstance(v1, (float, int)) and isinstance(v2, (float, int)):
            if abs(v1 - v2) > 1e-5:
              return False
          elif v1 != v2:
            return False
        return True
      return equal_dict(a, b)
    except Exception:
      return False

  @staticmethod
  def _get_unique_save_path(save_path: str) -> str:
    """Returns a unique path by appending (1), (2), ... if the file already exists."""
    base, ext = os.path.splitext(save_path)
    counter = 1
    new_path = save_path
    while os.path.exists(new_path):
      new_path = f'{base}({counter}){ext}'
      counter += 1
    return new_path

  def _copy_default_style_settings(self):
    """Returns a baseline default settings dict for comparison."""
    tmp = {}
    # mimic defaults without reading UI
    tmp['ev'] = 0
    if getattr(self, '_as_shot_disabled', False):
      tmp['wb_mode'] = 'auto'  # fallback if As Shot is disabled
    else:
      tmp['wb_mode'] = 'as_shot'  # normal default
    tmp['cct'] = DEFAULT_CCT
    tmp['tint'] = DEFAULT_TINT
    tmp['auto_exp'] = False
    tmp['details'] = {
      'Denoising': DEFAULT_DENOISING,
      'Chroma': DEFAULT_CHROMA_DENOISING,
      'Luma': DEFAULT_LUMA_DENOISING,
      'Enhance': DEFAULT_ENHANCEMENT,
    }
    tmp['editing'] = {
      'Highlights': DEFAULT_HIGHLIGHTS,
      'Shadows': DEFAULT_SHADOWS,
      'Sharpening': DEFAULT_SHARPENING,
      'Saturation': DEFAULT_SATURATION,
      'Vibrance': DEFAULT_VIBRANCE,
      'Contrast': DEFAULT_CONTRAST,
    }
    tmp['styles'] = []
    tmp['styles'] = []
    for i in range(NUMBER_OF_STYLES):
      if i == 0:
        tmp['styles'].append({'enabled': True, 'values': (100, 100, 100, 100, 100, 0)})
      else:
        tmp['styles'].append({'enabled': False, 'values': (0, 0, 0, 0, 0, 100)})
    return tmp


  def _save_current_settings(self):
    """Stores settings for the current image if modified."""
    if not hasattr(self, '_current_images') or not self._current_images:
      return
    current = self._copy_current_style_settings()
    self._saved_settings[self._current_index] = current

  def _on_next_image(self):
    """Moves to the next image if available."""
    if not hasattr(self, '_current_images') or not self._current_images:
      return
    self._save_current_settings()
    if self._current_index < len(self._current_images) - 1:
      self._current_index += 1
      self._on_browse()
    self._update_browse_buttons_state()

  def _on_prev_image(self):
    """Moves to the previous image if available."""
    if not hasattr(self, '_current_images') or not self._current_images:
      return
    self._save_current_settings()
    if self._current_index > 0:
      self._current_index -= 1
      self._on_browse()
    self._update_browse_buttons_state()

  def _on_goto_image(self, event=None):
    """Jump to a specific image index via entry box."""
    if not hasattr(self, '_current_images') or not self._current_images:
      return
    self._save_current_settings()
    try:
      idx = int(self._entry_index.get()) - 1
      if 0 <= idx < len(self._current_images):
        self._current_index = idx
        self._on_browse()
    except ValueError:
      pass
    self._update_browse_buttons_state()

  def _update_browse_buttons_state(self):
    """Enables/disables Next/Previous buttons and entry box depending on image list size."""
    # If no images loaded
    if not hasattr(self, '_current_images') or not self._current_images:
      self._btn_prev.config(state='disabled')
      self._btn_next.config(state='disabled')
      self._entry_index.config(state='disabled')
      return

    total = len(self._current_images)

    # Single image:
    if total == 1:
      self._btn_prev.config(state='disabled')
      self._btn_next.config(state='disabled')
      self._update_browse_entry('')
      self._entry_index.config(state='disabled')
      return

    # Multiple images:
    if self._is_video_sequence:
      self._update_browse_entry('')
      self._entry_index.config(state='disabled')
    else:
      self._entry_index.config(state='normal')

    if self._current_index <= 0:
      self._btn_prev.config(state='disabled')
      self._btn_next.config(state='normal')
    elif self._current_index >= total - 1:
      self._btn_prev.config(state='normal')
      self._btn_next.config(state='disabled')
    else:
      self._btn_prev.config(state='normal')
      self._btn_next.config(state='normal')

  def _get_contrast(self):
    """Returns current normalized contrast adjustment value."""
    return 2 * (self._editing_sliders['Contrast'].get() - CONTRAST_MIN) / (CONTRAST_MAX - CONTRAST_MIN) - 1

  def _get_highlight(self):
    """Returns current normalized highlight adjustment value."""
    return 2 * (self._editing_sliders['Highlights'].get() - HIGHLIGHTS_MIN) / (HIGHLIGHTS_MAX - HIGHLIGHTS_MIN) - 1

  def _get_shadow(self):
    """Returns current normalized shadow adjustment value."""
    return 2 * (self._editing_sliders['Shadows'].get() - SHADOWS_MIN) / (SHADOWS_MAX - SHADOWS_MIN) - 1

  def _get_saturation(self):
    """Returns current normalized saturation adjustment value."""
    return 2 * (self._editing_sliders['Saturation'].get() - SATURATION_MIN) / (SATURATION_MAX - SATURATION_MIN) - 1

  def _get_vibrance(self):
    """Returns current normalized vibrance adjustment value."""
    return 2 * (self._editing_sliders['Vibrance'].get() - VIBRANCE_MIN) / (VIBRANCE_MAX - VIBRANCE_MIN) - 1

  def _get_sharpening(self):
    """Returns current sharpening value mapped to [0, 50]."""
    return 50 * (self._editing_sliders['Sharpening'].get() - SHARPENING_MIN) / (SHARPENING_MAX - SHARPENING_MIN)

  def _get_denoising_strength(self):
    """Returns current normalized denoising strength."""
    return (self._details_sliders['Denoising'].get() - DENOISING_MIN) / (DENOISING_MAX - DENOISING_MIN)

  def _get_chroma_denoising(self):
    """Returns current normalized chroma denoising strength."""
    return (self._details_sliders['Chroma'].get() - CHROMA_DENOISING_MIN) / (
       CHROMA_DENOISING_MAX - CHROMA_DENOISING_MIN)

  def _get_luma_denoising(self):
    """Returns current normalized luma denoising strength."""
    return (self._details_sliders['Luma'].get() - LUMA_DENOISING_MIN) / (LUMA_DENOISING_MAX - LUMA_DENOISING_MIN)

  def _get_enhancement_strength(self):
    """Returns current normalized enhancement strength."""
    return (self._details_sliders['Enhance'].get() - ENHANCEMENT_MIN) / (ENHANCEMENT_MAX - ENHANCEMENT_MIN)

  def _get_preview_raw(self):
    img = self._raw_img
    h, w = img.shape[:2]
    max_size = PREVIEW_IMAGE_MAX_SIZE

    if max(h, w) <= max_size:
      return img
    if h > w:
      new_h = max_size
      new_w = int(w * (max_size / h))
    else:
      new_w = max_size
      new_h = int(h * (max_size / w))
    return imresize(img, new_h, new_w, interpolation_method='linear')

  def _get_selected_style_data(self):
    """Gets selected style data."""

    def single_nonzero_index(w):
      nz_indices = [i for i, v in enumerate(w) if v != 0]
      return nz_indices[0] if len(nz_indices) == 1 else None

    gain_params = 0
    gtm_params = 0
    ltm_params = 0
    chroma_params = 0
    gamma_params = 0

    w_gain = [0] * len(self._style_widgets)
    w_gtm = [0] * len(self._style_widgets)
    w_ltm = [0] * len(self._style_widgets)
    w_chroma = [0] * len(self._style_widgets)
    w_gamma = [0] * len(self._style_widgets)
    is_enabled = [True] * len(self._style_widgets)
    for style_idx, w in enumerate(self._style_widgets):
      enabled = w._enabled_var.get()
      if not enabled:
        is_enabled[style_idx] = False
        continue
      _, left, right, bottom, dot_x, dot_y = w.get_values()
      w_gain[style_idx] = left
      w_gtm[style_idx] = bottom
      w_ltm[style_idx] = 100 - dot_y
      w_chroma[style_idx] = dot_x
      w_gamma[style_idx] = right
      if any(param is None for param in (
         self._gain_params[style_idx],
         self._gtm_params[style_idx],
         self._ltm_params[style_idx],
         self._chroma_params[style_idx],
         self._gamma_params[style_idx],
      )):
        outputs = self._pipeline.get_ps_params(
          lsrgb=self._lsrgb_img, style_id=style_idx, post_process_ltm= self._ltm_mode.get() == 'enhanced',
          solver_iter=self._ltm_iters.get(), contrast_amount=self._get_contrast(), vibrance_amount=self._get_vibrance(),
          saturation_amount=self._get_saturation(), highlight_amount=self._get_highlight(),
          shadow_amount=self._get_shadow(), gain_param=None if w_gain[style_idx] else 0,
          gtm_param=None if w_gtm[style_idx] else 0, ltm_param=None if w_ltm[style_idx] else 0,
          chroma_lut_param=None if w_chroma[style_idx] else 0, gamma_param=None if w_gamma[style_idx] else 0)
        self._gain_params[style_idx] = outputs['gain_param']
        self._gtm_params[style_idx] = outputs['gtm_param']
        self._ltm_params[style_idx] = outputs['ltm_param']
        self._chroma_params[style_idx] = outputs['chroma_lut_param']
        self._gamma_params[style_idx] = outputs['gamma_param']

    t_gain = sum(w_gain)
    t_gtm = sum(w_gtm)
    t_ltm = sum(w_ltm)
    t_chroma = sum(w_chroma)
    t_gamma = sum(w_gamma)

    is_singular_gain = max(w_gain) == t_gain and t_gain != 0
    is_singular_gtm = max(w_gtm) == t_gtm and t_gtm != 0
    is_singular_ltm = max(w_ltm) == t_ltm and t_ltm != 0
    is_singular_chroma = max(w_chroma) == t_chroma and t_chroma != 0
    is_singular_gamma = max(w_gamma) == t_gamma and t_gamma != 0

    if is_singular_gain:
      idx_gain = single_nonzero_index(w_gain)
      gain_params = self._gain_params[idx_gain]
      gain_blending_weight = w_gain[idx_gain] / 100
    else:
      gain_blending_weight = None

    if is_singular_gtm:
      idx_gtm = single_nonzero_index(w_gtm)
      gtm_params = self._gtm_params[idx_gtm]
      gtm_blending_weight = w_gtm[idx_gtm] / 100
    else:
      gtm_blending_weight = None

    if is_singular_ltm:
      idx_ltm = single_nonzero_index(w_ltm)
      ltm_params = self._ltm_params[idx_ltm]
      ltm_blending_weight = w_ltm[idx_ltm] / 100
    else:
      ltm_blending_weight = None

    if is_singular_chroma:
      idx_chroma = single_nonzero_index(w_chroma)
      chroma_params = self._chroma_params[idx_chroma]
      chroma_blending_weight = w_chroma[idx_chroma] / 100
    else:
      chroma_blending_weight = None

    if is_singular_gamma:
      idx_gamma = single_nonzero_index(w_gamma)
      gamma_params = self._gamma_params[idx_gamma]
      gamma_blending_weight = w_gamma[idx_gamma] / 100
    else:
      gamma_blending_weight = None

    for style_idx in range(len(self._style_widgets)):
      if not is_enabled[style_idx]:
        continue

      if t_gain and not is_singular_gain:
        gain_params += (w_gain[style_idx] / t_gain) * self._gain_params[style_idx]
      if t_gtm and not is_singular_gtm:
        gtm_params += (w_gtm[style_idx] / t_gtm) * self._gtm_params[style_idx]
      if t_ltm and not is_singular_ltm:
        ltm_params += (w_ltm[style_idx] / t_ltm) * self._ltm_params[style_idx]
      if t_chroma and not is_singular_chroma:
        chroma_params += (w_chroma[style_idx] / t_chroma) * self._chroma_params[style_idx]
      if t_gamma and not is_singular_gamma:
        gamma_params += (w_gamma[style_idx] / t_gamma) * self._gamma_params[style_idx]

    computed_params = {'gain_params': gain_params, 'gtm_params': gtm_params, 'ltm_params': ltm_params,
                       'chroma_params': chroma_params, 'gamma_params': gamma_params,
                       'gain_b_weight': gain_blending_weight, 'gtm_b_weight': gtm_blending_weight,
                       'ltm_b_weight': ltm_blending_weight, 'chroma_b_weight': chroma_blending_weight,
                       'gamma_b_weight': gamma_blending_weight}

    return computed_params

  def _get_wb_data(self) -> Tuple[
    Union[np.ndarray, None], Union[np.ndarray, None], Union[bool, None], Union[float, None], Union[float, None]]:
    """Gets white-balance data."""
    wb_mode = self._wb_var.get()
    if wb_mode == 'as_shot' or wb_mode == 'custom':
      if 'cam_illum' in self._metadata:
        key = 'cam_illum'
      else:
        key = 'illum_color'
      illum = np.array(self._metadata[key], dtype=np.float32)
      if 'color_matrix' in self._metadata:
        key = 'color_matrix'
      else:
        key = 'ccm'
      ccm = np.array(self._metadata[key], dtype=np.float32)
      pref_awb = False
      if wb_mode == 'custom':
        cct = self._cct_slider.get()
        tint = self._tint_slider.get()
      else:
        cct = tint = None

    elif wb_mode == 'auto' or wb_mode == 'auto_neutral':
      illum = ccm = cct = tint = None
      if wb_mode == 'auto':
        pref_awb = True
      else:
        pref_awb = False
    else:
      self._update_status('Error: Unrecognized WB option.', show_message=True, erase_after=False)
      return None, None, None, None, None
    return illum, ccm, pref_awb, cct, tint

  def _call_pipeline(self, illum: np.ndarray, ccm: np.ndarray, pref_awb: bool, cct: float, tint: float,
                     scale: float, gain_param: Optional[torch.Tensor]=None,
                     gtm_param: Optional[torch.Tensor]=None, ltm_param: Optional[torch.Tensor]=None,
                     chroma_param: Optional[torch.Tensor]=None, gamma_param: Optional[torch.Tensor]=None,
                     gain_blending_weight: Optional[float]=None, gtm_blending_weight: Optional[float]=None,
                     ltm_blending_weight: Optional[float] = None, chroma_blending_weight: Optional[float]=None,
                     gamma_blending_weight: Optional[float]=None
                     ):
    self._synch_log()
    with torch.no_grad():
      try:
        outputs = self._pipeline(
          raw=self._raw_img, denoised_raw=self._denoised_raw_img, lsrgb=self._lsrgb_img, illum=illum, ccm=ccm,
          solver_iter=self._ltm_iters.get(), post_process_ltm=self._ltm_mode.get() == 'enhanced',
          use_generic_denoiser=self._denoise_setting.get()=='generic', use_cc_awb=self._awb_setting.get()=='generic',
          awb_user_pref=pref_awb, img_metadata=self._metadata, auto_exposure=self._auto_exp_enabled, log_messages=True,
          contrast_amount=self._get_contrast(), vibrance_amount=self._get_vibrance(),
          saturation_amount=self._get_saturation(), shadow_amount=self._get_shadow(),
          highlight_amount=self._get_highlight(), sharpening_amount=self._get_sharpening() * scale,
          target_cct=cct, target_tint=tint, denoising_strength=self._get_denoising_strength(),
          enhancement_strength=self._get_enhancement_strength(),
          chroma_denoising_strength=self._get_chroma_denoising(),
          luma_denoising_strength=self._get_luma_denoising(), ev_scale=self._ev_slider.get(), report_time=True,
          return_intermediate=False, apply_orientation=self._auto_orientation.get(), always_return_np=True,
          gain_param=gain_param, gtm_param=gtm_param, ltm_param=ltm_param, chroma_lut_param=chroma_param,
          gamma_param=gamma_param, gain_blending_weight=gain_blending_weight, gtm_blending_weight=gtm_blending_weight,
          ltm_blending_weight=ltm_blending_weight, chroma_blending_weight=chroma_blending_weight,
          gamma_blending_weight=gamma_blending_weight
        )
        self._synch_log(get=True)

        return outputs
      except Exception as e:
        self._show_message_dialog('Cannot render the image', f'Error: {e}', level='error')
        return

  @staticmethod
  def _temporal_smooth(pre: list[Union[torch.Tensor, np.ndarray, None]], current: Union[torch.Tensor, np.ndarray],
                       curr_weight: Optional[float] = 0.4, scale_mode: Optional[bool] = False
                       ) -> Union[torch.Tensor, np.ndarray]:
    """Temporal smoothing with optional channel-scale mode."""
    curr_weight = float(max(0.0, min(1.0, curr_weight)))
    prevs = [p for p in pre if p is not None]
    if not prevs or curr_weight >= 0.999:
      return current
    if scale_mode and isinstance(current, torch.Tensor):
      with torch.no_grad():
        prev_means = []
        for p in prevs:
          if isinstance(p, np.ndarray):
            p = torch.from_numpy(p).to(device=current.device, dtype=current.dtype)
          elif not isinstance(p, torch.Tensor):
            continue
          prev_means.append(p)
        if not prev_means:
          return current
        p_mean = torch.stack(prev_means).mean(dim=0)
        dims = list(range(current.ndim))
        dims.remove(1)
        mu_c = current.mean(dim=dims, keepdim=True).clamp(min=EPS)
        blended_mean = curr_weight * mu_c + (1 - curr_weight) * p_mean.view(1, -1, *([1] * (current.ndim - 2)))
        return (current / mu_c) * blended_mean

    if isinstance(current, torch.Tensor):
      valid_prev = [p.to(device=current.device, dtype=current.dtype)
                    for p in prevs if isinstance(p, torch.Tensor) and p.shape == current.shape]
      if not valid_prev:
        return current
      w_prev = (1 - curr_weight) / len(valid_prev)
      with torch.no_grad():
        out = current * curr_weight
        for p in valid_prev:
          out += p * w_prev
      return out

    elif isinstance(current, np.ndarray):
      valid_prev = [p for p in prevs if isinstance(p, np.ndarray) and p.shape == current.shape]
      if not valid_prev:
        return current
      w_prev = (1 - curr_weight) / len(valid_prev)
      out = current * curr_weight
      for p in valid_prev:
        out += p * w_prev
      return out
    return current

  @staticmethod
  def _shift_and_append(pre: List[Union[torch.Tensor, np.ndarray, None]], current: Union[torch.Tensor, np.ndarray]
                        ) -> List[Union[torch.Tensor, np.ndarray, None]]:
    """Shifts the list to the right and inserts current at the start."""
    if not pre:
      return [current if not isinstance(current, torch.Tensor) else current.detach()]
    return [current if not isinstance(current, torch.Tensor) else current.detach()] + pre[:-1]

  def _render_image(self, preview: Optional[bool] = True, style: Optional[Dict] = None):
    """Renders the current image using all active settings."""
    if style is not None:
      pre_gui_status = self._gui_enabled
      if pre_gui_status != 'normal':
        self._set_ui_enabled(True)
      self._apply_style(style, rendering=False)
      self._set_ui_enabled(pre_gui_status == 'normal')

    if self._raw_img is None or self._metadata is None:
      return
    self._update_status('Rendering', erase_after=False)
    self._update_status('\n', show_message=False)

    if preview and self._preview_mode.get() == 'draft' and not isinstance(self._raw_img, torch.Tensor):
      self._raw_img = self._get_preview_raw()
    # Gets WB data
    illum, ccm, pref_awb, cct, tint = self._get_wb_data()

    self._synch_log()
    scale = float(max(self._raw_img.shape)) / self._original_sz if preview else 1.0

    if self._lsrgb_img is None:
      outputs = self._call_pipeline(illum=illum, ccm=ccm, pref_awb=pref_awb, cct=cct, tint=tint, scale=scale)
      if outputs is None:
        return

      self._srgb_img = outputs['srgb']
      self._raw_img = outputs['raw']
      self._lsrgb_img = outputs['lsrgb']
      self._denoised_raw_img = outputs['denoised_raw']
      cct = outputs['cct']
      tint = outputs['tint']
      illum = outputs['illum']
      ccm = outputs['ccm']

      if cct is not None:
        self._cct_slider.set(cct)
      if tint is not None:
        self._tint_slider.set(tint)

      # Checks for temporal data
      if self._is_video_sequence and (self._pre_illum is not None and self._pre_ccm is not None and
                                      illum is not None and ccm is not None):
        self._update_status('Illuminant & CCM temporal smoothing', show_message=False)
        illum = PhotoEditorUI._temporal_smooth(self._pre_illum, illum)
        ccm = PhotoEditorUI._temporal_smooth(self._pre_ccm, ccm)
        self._pre_illum = PhotoEditorUI._shift_and_append(pre=self._pre_illum, current=illum)
        self._pre_ccm = PhotoEditorUI._shift_and_append(pre=self._pre_ccm, current=ccm)
        cct = None
        tint = None

    style_params = self._get_selected_style_data()
    gain_params = style_params['gain_params']
    gain_b_weights = style_params['gain_b_weight']
    gtm_params = style_params['gtm_params']
    gtm_b_weights = style_params['gtm_b_weight']
    ltm_params = style_params['ltm_params']
    ltm_b_weights = style_params['ltm_b_weight']
    chroma_params = style_params['chroma_params']
    chroma_b_weights = style_params['chroma_b_weight']
    gamma_params = style_params['gamma_params']
    gamma_b_weights = style_params['gamma_b_weight']

    all_zero = all(
      (p is None)
      or (isinstance(p, (int, float)) and p == 0)
      or (isinstance(p, torch.Tensor) and torch.all(p == 0))
      for p in (gain_params, gtm_params, ltm_params, chroma_params, gamma_params)
    )

    if all_zero:
      self._srgb_img = self._pipeline.to_np(self._lsrgb_img)
      if self._auto_orientation.get():
        self._on_auto_orientation_changed()
      if 'cct' not in locals():
        cct = None
      if 'tint' not in locals():
        tint = None
    else:
      # Checks for temporal data
      if self._is_video_sequence and (
         self._pre_gain is not None and self._pre_gtm is not None #and self._pre_ltm is not None
         and self._pre_chroma_lut is not None and self._pre_gamma is not None):

        self._update_status('Photofinishing temporal smoothing', show_message=False)
        gain_params = PhotoEditorUI._temporal_smooth(self._pre_gain, gain_params)
        gtm_params = PhotoEditorUI._temporal_smooth(self._pre_gtm, gtm_params)
        chroma_params = PhotoEditorUI._temporal_smooth(self._pre_chroma_lut, chroma_params)
        gamma_params = PhotoEditorUI._temporal_smooth(self._pre_gamma, gamma_params)

        self._pre_gain = PhotoEditorUI._shift_and_append(pre=self._pre_gain, current=gain_params)
        self._pre_gtm = PhotoEditorUI._shift_and_append(pre=self._pre_gtm, current=gtm_params)
        self._pre_chroma_lut = PhotoEditorUI._shift_and_append(pre=self._pre_chroma_lut, current=chroma_params)
        self._pre_gamma = PhotoEditorUI._shift_and_append(pre=self._pre_gamma, current=gamma_params)

      outputs = self._call_pipeline(illum=illum, ccm=ccm, pref_awb=pref_awb, cct=cct, tint=tint, scale=scale,
                                    gain_param=gain_params, gtm_param=gtm_params, ltm_param=ltm_params,
                                    chroma_param=chroma_params, gamma_param=gamma_params,
                                    gain_blending_weight=gain_b_weights,
                                    gtm_blending_weight=gtm_b_weights, ltm_blending_weight=ltm_b_weights,
                                    chroma_blending_weight=chroma_b_weights, gamma_blending_weight=gamma_b_weights,)
      if outputs is None:
        return

      self._srgb_img = outputs['srgb']
      self._raw_img = outputs['raw']
      self._lsrgb_img = outputs['lsrgb']
      self._denoised_raw_img = outputs['denoised_raw']

    if cct is not None:
      self._cct_slider.set(cct)
    if tint is not None:
      self._tint_slider.set(tint)
    self._show_image()
    self._update_status('', erase_after=False)

  def _show_image(self):
    """Displays the rendered sRGB image on a fixed canvas (no vertical drift)."""
    if getattr(self, '_srgb_img', None) is None or not isinstance(self._srgb_img, np.ndarray):
      self._canvas.delete('all')
      self._canvas.create_text(
          self._canvas.winfo_width() // 2,
          self._canvas.winfo_height() // 2,
          text='No image loaded',
          fill=TEXT_COLOR,
          font=self._text_font,
          anchor='center',
      )
      return

    img = np.clip(self._srgb_img, 0, 1)
    img_uint8 = (img * 255).astype(np.uint8)
    pil_img = PIL.Image.fromarray(img_uint8)

    # Get canvas size
    w, h = self._canvas.winfo_width(), self._canvas.winfo_height()
    if w < 2 or h < 2:
      self.root.after(100, self._show_image)
      return

    pil_img = pil_img.copy()
    pil_img.thumbnail((w, h), PIL.Image.LANCZOS)

    # Convert to Tk image and center it
    self._tk_img = PIL.ImageTk.PhotoImage(pil_img)
    self._canvas.delete("all")
    x = (w - pil_img.width) // 2
    y = (h - pil_img.height) // 2
    self._canvas.create_image(x, y, anchor="nw", image=self._tk_img)








