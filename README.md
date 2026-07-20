# Modular Neural Image Signal Processing

#### **[Mahmoud Afifi](https://www.mafifi.info/), [Zhongling Wang](https://ece.uwaterloo.ca/~z2228wan/), [Ran Zhang](https://www.linkedin.com/in/ran-zhang-48b85021/), and [Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)**

---

## 🆕 News

- **Executable binaries are now available** for the interactive photo-editing tool.  
  You can download pre-built standalone executables from [here](gui).

## 📦 Latest Release
**2025-12-30 — v0.1.5 (Beta)**  
Added support for Apple Silicon (MPS).

➡️ See full release history in [CHANGELOG.md](CHANGELOG.md)

  
---

<p align="center">
  <a href="https://www.youtube.com/watch?v=ByhQjQSjxVM" target="_blank">
    <img src="https://img.youtube.com/vi/ByhQjQSjxVM/maxresdefault.jpg" width="100%" />
  </a>
  <br>
  <sub>
    🎬 <a href="https://www.youtube.com/watch?v=ByhQjQSjxVM" target="_blank">Click to watch the video</a>
  </sub>
</p>

We present a **modular neural image signal processing** (ISP) framework that produces high-quality display-referred images while providing a high degree of modularity with explicit control over multiple intermediate stages of the rendering pipeline. Our ISP is fully differentiable and requires no manual tuning, and its modular structure not only improves rendering accuracy but also enhances scalability, debuggability, generalization to unseen cameras, and flexibility to support different user-preference picture styles within a lightweight and efficient design.

On top of this modular neural ISP, we developed a user-interactive photo-editing tool that supports diverse editing operations, different picture styles, and enables unlimited post-editable re-rendering and re-styling. The tool accepts DNG raw images from any camera as well as sRGB images from third-party sources. Across multiple test sets, our method consistently delivers competitive qualitative and quantitative performance.

🔗 **[Paper](https://arxiv.org/abs/2512.08564)**

---

## 🛠️ Installation

To run the standalone executable without a Python setup, see [`here`](gui).

Otherwise, you can set up the environment using Conda or venv:

### Option 1: Using Conda (recommended)

```bash
# Create and activate a new conda environment
conda create -n modular_neural_isp_env python=3.9 -y
conda activate modular_neural_isp_env

# Install PyTorch (adjust CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
pip install -r requirements.txt
```

### Option 2: Using venv
```bash
# Create and activate a virtual environment
python -m venv modular_neural_isp_env
source modular_neural_isp_env/bin/activate        # macOS/Linux
# On Windows:
# .\modular_neural_isp_env\Scripts\activate

# Install PyTorch (adjust CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
pip install -r requirements.txt
```

### 📸 ExifTool (Optional but Recommended)

ExifTool is used for robust raw/DNG metadata extraction and provides the best compatibility across cameras and file variants.  
If available, it will be used automatically. If not, the code falls back to a Python metadata extractor.

#### 1. Install ExifTool (Optional)

**Linux (Ubuntu/Debian):**
```bash
sudo apt install libimage-exiftool-perl
```

**Arch Linux:**
```bash
sudo pacman -S perl-image-exiftool
```

**macOS (Homebrew):**
```bash
brew install exiftool
```

**Windows:**
1. Download the Windows package from: https://exiftool.org  
2. Extract the ZIP file.  
3. Rename `exiftool(-k).exe` → `exiftool.exe`  
4. Place it somewhere permanent, e.g. `C:\exiftool\exiftool.exe`.


#### 2. Configure ExifTool Path (Optional)

If ExifTool is not detected automatically, update `EXIFTOOL_PATH` in  [`utils/constants.py`](utils/constants.py):

```python
import platform

if platform.system() == 'Windows':
    EXIFTOOL_PATH = r'C:\exiftool\exiftool.exe'    # Update accordingly
elif platform.system() == 'Darwin':  # macOS
    EXIFTOOL_PATH = '/usr/local/bin/exiftool'   # Update accordingly
else:  # Linux
    EXIFTOOL_PATH = '/usr/bin/exiftool'   # Update accordingly
```

---

## ▶️ Get Started
For console-based rendering of a DNG raw file taken by any camera (or any sRGB image produced by third-party software) refer to the [`main`](main) directory.  
To launch the interactive photo-editing tool, see the [`gui`](gui) directory.

> **Note:** Inference supports **CUDA, CPU, and Apple MPS**. On MPS, unsupported operators are automatically executed on CPU.


---

## ⚙️ Training
To retrain our framework, begin by training the denoiser network (see [`denoising`](denoising) for instructions). Then train the photofinishing module (refer to [`photofinishing`](photofinishing)). Finally, train the detail-enhancement network (see [`enhancement`](enhancement)).

> **Note:** Training is currently supported on **CUDA and CPU only**. Training on Apple MPS is **not supported**.

---

## 📊 Testing
To test our method, check the testing example [`here`](main#-testing).  
To benchmark other methods using the same evaluation metrics used in our [paper](https://arxiv.org/abs/2512.08564), see the [`evaluation`](evaluation) directory.

---

## 📄 Citation
If you use this code in your research, please cite our paper:
```
@article{afifi2025modular,
  title={Modular Neural Image Signal Processing},
  author={Afifi, Mahmoud and Wang, Zhongling and Zhang, Ran and Brown, Michael S},
  journal={arXiv preprint arXiv:2512.08564},
  year={2025}
}

```
