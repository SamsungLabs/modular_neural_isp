# 🖌️ Photofinishing

After [color correction](../awb_ccm), the image is processed by the photofinishing module, which includes the following stages: digital gain, global tone mapping, local tone mapping, chroma mapping, and gamma correction. Each stage is applied sequentially, where a neural network takes the output of the previous stage(s) and predicts the corresponding coefficient(s) for the next transformation (see Sec. 2.3 of the [paper](https://arxiv.org/abs/2512.08564) for details).

In addition to these main stages of the photofinishing module, we also implemented different editing functions that operate within the photofinishing module as part of our photo-editing tool. All functionality is implemented in the `PhotofinishingModule` inside [`photofinishing_model.py`](photofinishing_model.py).
 

<p align="center">
  <img src="../figures/photofinishing.gif" width="600" style="max-width: 100%; height: auto;">
</p>

---

## ⚙️ Training

We provide trained models for all styles (including the default style) of the [S24 dataset](https://github.com/mahmoudnafifi/time-aware-awb/tree/main/s24-raw-srgb-dataset) in the [`models`](models) directory. For the artistic styles (Style #1, #2, … #5), we also include versions trained with the 3D LUT option.

To train the photofinishing module’s networks, use `train.py`.  
Below is an example:

```bash
python train.py \
    --in-training-dir /path/to/training/denoised/raw/image/folder \
    --gt-training-dir /path/to/training/ground-truth/srgb/image/folder \
    --data-training-dir /path/to/training/metadata/folder \   # optional; defaults to "data" in the same directory as in-training-dir
    --in-validation-dir /path/to/validation/denoised/raw/image/folder \
    --gt-validation-dir /path/to/validation/ground-truth/srgb/image/folder \
    --data-validation-dir /path/to/validation/metadata/folder \   # optional; defaults to "data" in the same directory as in-validation-dir
    --exp-name name-of-dataset   # choose any suffix you prefer
```

This command first creates a temporary folder containing HDF5 (`.h5`) files with batches of paired denoised and ground-truth images (resized to 512 by default; use `--in-size <value>` to change the size).  
The temporary folder name begins with `ps_temp_h5` and is placed in the root directory of the training data.  
You may customize this prefix using `--temp-folder`.

The remainder of the folder name is automatically derived from the batch size, ground-truth directory, and input size.  
If the temporary folder already exists (e.g., from previous experiments), the script will skip regenerating the data. To force a clean rebuild when the folder already exists, use `--overwrite-temp-folder`, which is useful when a previous dataset creation run was incomplete or failed for any reason. To automatically delete the temporary folder after training, use `--delete-temp-folder`. In all of our experiments, we trained on the full image (resized to `--in-size`).
If you prefer to train on patches instead, add `--extract-patches`. To enable learning of a 3D LUT (applied before chroma mapping), use `--use-3d-lut`.

See [`train.py`](train.py) for additional useful arguments.

---

## 📊 Testing
To test the trained photofinishing module separately from the full neural ISP pipeline, use the [`test.py`](test.py) script (to test the entire framework, please refer to [`main/test.py`](../main/test.py)).

Below is an example:


```bash
python test.py \
    --model-path /path/to/trained/photofinishing/module \
    --in-testing-dir /path/to/input/denoised/raw/image/folder \
    --gt-testing-dir /path/to/ground-truth/srgb/image/folder \
    --data-testing-dir /path/to/testing/metadata/folder   # optional; defaults to "data" in the same directory as in-testing-dir
```

This will test the model and report PSNR and SSIM, and will also save the results in `.txt` format inside the `results` directory. Use `--result-dir <path>` to specify a custom output directory.

Use `--post-process-ltm` to apply multi-scale processing and refinement of the local tone-mapping coefficient maps (Sec. B.1 of the supplementary materials of our [paper](https://arxiv.org/abs/2512.08564)). A standalone repository for the GPU-accelerated bilateral solver is available [here](http://github.com/mahmoudnafifi/gpu-accelerated-bilateral-solver).

By default, [`test.py`](test.py) downsamples the images in `--in-testing-dir` and the corresponding ground-truth sRGB images to 1/4 of their original resolution, similar to the behavior in the main pipeline. To disable downsampling, use `--no-ds`. With `--no-ds`, the images in `--in-testing-dir` are processed at their original resolution and compared directly to the full-resolution ground-truth images.

By default, the configuration file associated with the trained model in `--model-path` is expected to be located in a `config` folder inside the same directory as the testing script. To specify a different configuration directory, use `--config-dir <path>`.

---

### ✉️ Inquiries
For inquiries about the photofinishing module and its ablation studies, please contact Mahmoud Afifi (m.3afifi@gmail.com).
