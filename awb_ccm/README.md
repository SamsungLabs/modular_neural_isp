# 🎨 Color Correction

Color correction is the second stage of our pipeline (Sec. 2.2 in the [paper](https://arxiv.org/abs/2512.08564)), where we apply white-balance correction and a color correction matrix (CCM) to the denoised raw image.

<p align="center">
  <img src="../figures/color-correction.gif" width="600" style="max-width: 100%; height: auto;">
</p>

We provide three options for auto white-balance correction:

- **`As shot`**:  
  Uses the camera-estimated white-balance gains stored in the DNG metadata.

- **`Auto`**:  
  Runs an illuminant estimation model to predict the white-balance gains of the scene.  
  We support two options:
  1. A [**camera-specific illuminant estimation model**](https://github.com/mahmoudnafifi/time-aware-awb) trained on the S24 main camera raw domain
  2. A [**cross-camera illuminant estimation model**](https://github.com/mahmoudnafifi/C5) for images from other cameras  

  For auto estimation, we support:
  - **Neutral white-balance** (removes color casts to appear as if lit by white light)  
  - **Preference-aware white-balance**, which preserves desirable color biases under certain lighting conditions, implemented using a [**post-WB mapping function**](https://github.com/SamsungLabs/aesthetics-pref-awb).

- **`Custom`**:  
  The user manually sets the target correlated color temperature (CCT) and tint values.

We trained the models using the scripts provided in the original repos for the [camera-specific method](https://github.com/mahmoudnafifi/time-aware-awb)  and the [cross-camera model](https://github.com/mahmoudnafifi/C5). Trained illuminant estimation and preference-bias models can be found in the [`models`](models) directory.  

For the camera-specific model, we omitted time-location metadata (not available in many DNGs).  For the cross-camera model, we use a modified version of C5 that relies only on the histogram of the input (test) image and does not require the additional images used in the original method.  

See Sec. I.3 of the [paper](https://arxiv.org/abs/2512.08564) for more details.


---

### ✉️ Inquiries
For inquiries about the core functionalities of the camera-specific AWB, cross-camera AWB, or the white-balance preference-bias mapping, please refer to the original repositories of each method.  
For any other inquiries related to white balancing in this project, please contact Mahmoud Afifi (m.3afifi@gmail.com).

