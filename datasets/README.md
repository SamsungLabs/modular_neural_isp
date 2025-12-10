# 📚 Datasets

We used the [S24 dataset](https://github.com/mahmoudnafifi/time-aware-awb/tree/main/s24-raw-srgb-dataset) in our main experiments.  
We also reported results on the [MIT-Adobe 5K dataset](https://data.csail.mit.edu/graphics/fivek/) (Sec. K.2.2 of the supplementary materials of our [paper](https://arxiv.org/abs/2512.08564)), where we used **Expert C ground truth**.  
To prepare Adobe 5K Expert C data:

1. Run the script [`download_expertc_adobe5k.py`](download_expertc_adobe5k.py) to download the Expert C `.tif` images.  
2. Use Adobe Lightroom/Photoshop to export the images as full-resolution JPG files with 90% quality using the "sRGB IEC 61966-2-1" color space.  
3. Run [`adobe_5k_preprocessing.py`](adobe_5k_preprocessing.py) to split the dataset into training, validation, and testing sets and to extract raw images and DNG metadata.

We also evaluated on the [Zurich raw-to-sRGB dataset](https://aiff22.github.io/pynet.html#dataset) (Sec. B.2 of the supplementary materials of our [paper](https://arxiv.org/abs/2512.08564)).

---

### 📁 Directory Structure

All datasets are expected to follow the directory structure below (applied to each subset, e.g., `train`, `val`, `test`):

- `raw_images/`: input raw PNG-16 images  
- `denoised_raw_images/`: pseudo ground-truth denoised PNG-16 raw images  
- `data/`: metadata JSON files (white-balance gains, CCMs, etc.)  
- `srgb_images/`: corresponding sRGB images  
  - *(this folder name may vary for different picture styles, e.g., `srgb_images_style_0`)*

The **S24 dataset** already follows this structure by default (with `srgb_images_style_0` … `style_5` for the default and five artistic picture styles).  
The **Zurich** and **MIT-Adobe 5K** datasets require preprocessing to match this layout.  
We provide the scripts [`adobe_5k_preprocessing.py`](adobe_5k_preprocessing.py) and [`zurich_raw2srgb_preprocessing.py`](zurich_raw2srgb_preprocessing.py) to pre-process the dataset images and build this structure.  
Please check each file and adjust the dataset paths accordingly.

---

### 📝 Notes on Denoised Raw Images

Only the **S24 dataset** provides PNG-16 denoised raw images.  
Zurich and Adobe 5K do *not* include denoised raw ground-truth, so we processed them as follows:

- **Zurich:**  
  As described in our [paper](https://arxiv.org/abs/2512.08564) (Sec. B.2), we generate pseudo denoised raw ground-truth using the workflow detailed in the supplementary materials.  
  The script [`zurich_raw2srgb_preprocessing.py`](zurich_raw2srgb_preprocessing.py) performs the preprocessing for the Zurich raw-to-sRGB dataset.

- **Adobe 5K:**  
  We follow the same approach as S24: apply Adobe Lightroom AI Denoiser to the DNG images, then extract the denoised demosaiced raw images.  
  The script [`extract_denoised_images.py`](extract_denoised_images.py) extracts 16-bit PNG raw images from the Adobe-denoised DNGs, and [`adobe_5k_preprocessing.py`](adobe_5k_preprocessing.py) handles dataset preprocessing and splitting.

---

### ✉️ Inquiries
For inquiries about the datasets used in this project, please contact Mahmoud Afifi (m.3afifi@gmail.com).
