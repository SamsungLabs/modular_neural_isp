# 🎛️ Neural ISP Pipeline

The core logic of our pipeline (including the backend functionality of the photo-editing tool) is implemented in the `PipeLine` class located in [`pipeline.py`](pipeline.py).

The `PipeLine` class accepts paths to pre-trained models for the following: [`denoising`](../denoising), [`photofinishing`](../photofinishing), and [`enhancement`](../enhancement).

It can also optionally load models for: [auto white balance (AWB)](../awb_ccm), [raw compression / JPEG Adapter](../io_), and [sRGB linearization](../io_).

These components together form the full Modular Neural ISP pipeline.
<p align="center">
  <img src="../figures/pipeline.jpg" style="width: 100%; max-width: 100%; height: auto;">
</p>


---

## ▶️ Get Started

To run the pipeline from a console-based interface, use [`demo.py`](demo.py).  
For the graphical user interface (GUI), refer to the [`gui`](../gui) folder.

Below is an example:


```bash
python demo.py \
    --input-file path/to/input/file \    # Can be a DNG from any camera, a JPEG saved by our tool, or any third-party JPEG/PNG
    --denoising-model-path path/to/pre-trained/denoising/model \
    --photofinishing-model-path path/to/pre-trained/photofinishing/model \
    --enhancement-model-path path/to/pre-trained/detail-enhancement/model
```

If `--enhancement-model-path` is omitted, the detail-enhancement step in our pipeline will be disabled.

### Mixing Picture Styles
You can interpolate between different picture styles by providing paths to multiple photofinishing models (each trained for a specific style) using the `--multi-style-photofinishing-model-paths` argument.
Below is an example:

```bash
python demo.py \
    --input-file path/to/input/file \
    --denoising-model-path path/to/pre-trained/denoising/model \
    --multi-style-photofinishing-model-paths \
        path/to/pre-trained/photofinishing/model/style_1 \
        path/to/pre-trained/photofinishing/model/style_2 \
        path/to/pre-trained/photofinishing/model/style_3 \
    --multi-style-weights 20 50 30 \
    --enhancement-model-path path/to/pre-trained/detail-enhancement/model
```

In the example above, three picture styles are blended using weights of 20%, 50%, and 30% for style 1, style 2, and style 3, respectively.

Intermediate stages of the photofinishing module can also be mixed independently using:

- `--multi-style-gain-weights` for digital gain  
- `--multi-style-gtm-weights` for global tone mapping  
- `--multi-style-ltm-weights` for local tone mapping  
- `--multi-style-chroma-weights` for chroma mapping  
- `--multi-style-gamma-weights` for gamma correction  

**Note:** `--multi-style-weights` overrides all individual stage weights. You should use either `--multi-style-weights` or all the intermediate stage weights (`--multi-style-gain-weights`, `--multi-style-gtm-weights`, … `--multi-style-gamma-weights`), but not both.

Below is an example of mixing chroma mapping (20% / 80%) between two picture styles, while keeping all other stages from the first style:

```bash
python demo.py \
    --input-file path/to/input/file \
    --denoising-model-path path/to/pre-trained/denoising/model \
    --multi-style-photofinishing-model-paths \
        path/to/pre-trained/photofinishing/model/style_1 \
        path/to/pre-trained/photofinishing/model/style_2 \
    --multi-style-gain-weights 100 0 \
    --multi-style-gtm-weights 100 0 \
    --multi-style-ltm-weights 100 0 \
    --multi-style-chroma-weights 20 80 \
    --multi-style-gamma-weights 100 0 \
    --enhancement-model-path path/to/pre-trained/detail-enhancement/model
```

You can also disable any stage of the photofinishing module by setting its weight to zero.
Below is an example showing how to turn off both global and local tone mapping:

```bash
python demo.py \
    --input-file path/to/input/file \
    --denoising-model-path path/to/pre-trained/denoising/model \
    --multi-style-photofinishing-model-paths \
        path/to/pre-trained/photofinishing/model 
    --multi-style-gain-weights 100 \
    --multi-style-gtm-weights 0 \
    --multi-style-ltm-weights 0 \
    --multi-style-chroma-weights 100 \
    --multi-style-gamma-weights 100 \
    --enhancement-model-path path/to/pre-trained/detail-enhancement/model
```

### Exposure Adjustment

You can adjust exposure manually using the `--ev-value` argument.  
Example: `--ev-value 1.5`.  
The valid range for exposure values is defined in [`utils/constants.py`](../utils/constants.py).

You can also enable automatic exposure adjustment using `--auto-exposure` (see Sec. I.2 of the supplementary materials in our [paper](https://arxiv.org/abs/2512.08564) for details).


### White-Balance Correction

We provide multiple options for white-balance correction:

1. **Using camera-provided illuminant metadata**  
   When processing raw DNG files (or JPEGs previously rendered with `--store-raw` argument),  
   the illuminant stored in metadata is used by default.

2. **Fixed illuminant for third-party sRGB inputs**  
   If the input is an sRGB image produced by a third-party camera or software,  
   a fixed illuminant is used (not recommended).  
   In this case, we strongly recommend enabling an auto mode (see below).

3. **Estimating the illuminant using AWB models**  
   You can force re-estimation of the illuminant by using `--re-compute-awb`. This ignores metadata (if any) and runs one of our trained illuminant estimators: 1) a camera-specific AWB model (trained for the S24 main camera), or 2) a cross-camera AWB model.  The appropriate model is selected automatically based on the input image.  
Paths to these models are defined in [`utils/constants`](../utils/constants.py). For user-preference AWB (where a learned white-balance bias is applied) use `--re-compute-awb --pref-awb`.

4. **Manual white-balance control**  
You can manually set correlated color temperature (CCT) and/or tint using `--target-cct` and `--target-tint` arguments. Example: `--target-cct 5000 --target-tint -20`; valid ranges are 1800-10000 K for CCT and -100 to +100 for tint.

### Detail Processing Options

You can control raw denoising strength using `--denoising-strength <value>` (where `<value>` ∈ [0, 1]). Additional non-learning-based denoising steps are available: `--luma-denoising-strength` and `--chroma-denoising-strength`. 

Example:
`--denoising-strength 0.5 --luma-denoising-strength 0.2 --chroma-denoising-strength 0.3`
In this example raw denoising is applied at 50%, luma denoising at 20%,  and chroma denoising at 30%.

Detail enhancement strength is controlled by `--enhancement-strength <value>` (where `<value>` ∈ [0, 1]). Example: `--enhancement-strength 0.8` applies 80% of the enhancement module.

### Editing Options

Use the following flag to refine local tone mapping (LTM) `--post-process-ltm`. This applies our multi-scale and spatial refinement to reduce potential halo artifacts  
(Sec. B.1 of the supplementary materials of our [paper](https://arxiv.org/abs/2512.08564)). The refinement uses our GPU-accelerated iterative bilateral solver (see Sec. A of the supplementary materials). Control the number of iterations using `--solver-iterations <num>`.  Example: `--post-process-ltm --solver-iterations 50`. 

A standalone repository for the GPU-accelerated bilateral solver is available [here](http://github.com/mahmoudnafifi/gpu-accelerated-bilateral-solver).

Additional editing controls:

- Contrast: `--contrast-amount` (range −1 to 1)  
- Vibrance: `--vibrance-amount` (range −1 to 1)  
- Saturation: `--saturation-amount` (range −1 to 1)  
- Sharpening: `--sharpening-amount` (range 0 to 50)  
- Highlights: `--highlight-amount` (range −1 to 1)  
- Shadows: `--shadow-amount` (range −1 to 1)

### Additional Options

- Apply auto-orientation (based on metadata): `--apply-orientation` 
- Save intermediate photofinishing outputs as PNG-16 files: `--save-intermediate`
- Save the input raw image as PNG-16: `--save-input-raw`
- Embed raw and metadata into the output JPEG for future re-editing: `--store-raw`
- Choose output folder: `--output-dir <path>`
- Select device (CPU/GPU):  `--device <device>` (where device can be `gpu` or `cpu`).

Here is an example of rendering an image located at `/images/image_0.dng` using the generic denoiser, automatic preference-bias white balancing, a mixture of picture styles (Style #1 and Style #5 from the [S24 dataset](https://github.com/mahmoudnafifi/time-aware-awb/tree/main/s24-raw-srgb-dataset)), multi-scale halo-reduction processing, highlight and shadow adjustments, raw+metadata embedding in the output JPEG, and saving intermediate photofinishing outputs.
```bash
python demo.py \
    --input-file /images/image_0.dng \
    --denoising-model-path ../denoising/models/generic_base.pth \
    --multi-style-photofinishing-model-paths \
        ../photofinishing/models/photofinishing_s24-style-1.pth \
        ../photofinishing/models/photofinishing_s24-style-5.pth \
    --enhancement-model-path ../enhancement/models/enhancement_s24-style-0.pth \
    --multi-style-weights 50 50 \
    --post-process-ltm \
    --re-compute-awb \
    --pref-awb \
    --highlight-amount -0.4 \
    --shadow-amount 0.3 \
    --store-raw \
    --save-intermediate
```

After running the command in the above example, you should see the following files generated:

- `image_0-1-denoised.png`: Denoised raw RGB image (16-bit PNG).
- `image_0-2-lsrgb.png`: Linear sRGB image (16-bit PNG).
- `image_0-3-gain.png`: Output after applying digital gain (16-bit PNG).
- `image_0-4-gtm.png`: Output after applying global tone mapping (16-bit PNG).
- `image_0-5-ltm.png`: Output after applying local tone mapping (16-bit PNG).
- `image_0-6-cbcr-lut.png`: Output after applying chroma mapping (16-bit PNG).
- `image_0-7-gamma.png`: Output after gamma correction (16-bit PNG).
- `image_0-output.jpg`: Final output (8-bit JPEG) with raw data and metadata embedded.

<p align="center">
  <img src="../figures/output-images.gif" width="600" style="max-width: 100%; height: auto;">
</p>


---


## 📊 Testing

To test the entire framework on a testing set, where the set includes paired raw and ground-truth image folders along with a metadata folder (as described in [`datasets`](../datasets)), you can use [`test.py`](test.py).  
Below is an example:

```bash
python test.py \
    --denoising-model-path \path\to\trained\denoising\model \
    --photofinishing-model-path \path\to\trained\photofinishing\module \
    --enhancement-model-path \path\to\trained\detail-enhancement\model \
    --in-testing-dir /path/to/input/raw/16-PNG/image/folder \
    --gt-testing-dir /path/to/ground-truth/srgb/image/folder \
    --data-testing-dir /path/to/metadata/json/file/folder
```

The `--data-testing-dir` argument is optional. If it is not provided, the code assumes that the metadata folder is located in the root directory of `in-testing-dir` and is named `data`.

The above command reports PSNR and SSIM between the rendered images and the ground-truth. The results are printed to the console and also saved as a `.txt` file inside a folder named `results`.  To specify a different folder for saving results, use `--result-dir <folder-path>`. To append a postfix to the result filename, use `--result-file-postfix <value>`. To additionally report LPIPS and Delta E 2000, include `--report-lpips`
and `--report-delta-e`.  For halo-artifact mitigation in the LTM stage, add `--post-process-ltm`. To disable image downsampling before photofinishing (and skip guided upsampling afterward), use `--no-downsampling`. 

For a general evaluation script that reports the same metrics on images produced by any external method, refer to the
[`evaluation`](../evaluation) directory.



---

### ✉️ Inquiries
For inquiries about the pipeline or the photo-editing tool’s backend functionalities, please contact Mahmoud Afifi (m.3afifi@gmail.com).
