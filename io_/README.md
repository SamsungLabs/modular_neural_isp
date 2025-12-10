# 🔁 Input–Output Processing

Our photo-editing tool enables unlimited post-editing, allowing the output JPEG image to be re-rendered at any time after saving, with negligible accuracy loss compared to processing the original raw data. This is achieved through the [Raw-JPEG Adapter](https://github.com/mahmoudnafifi/raw-jpeg-adapter) pipeline, which transforms raw sensor data into a form suitable for JPEG compression. At decoding time, the adapted data is converted back to raw for high-quality re-rendering.

Another key advantage of our tool is its ability to accept sRGB images from any third-party source, including images from other cameras, software pipelines, or even AI-generated content. We accomplish this by linearizing the input using the [CIE XYZ Net](https://github.com/mahmoudnafifi/CIE_XYZ_NET), and then converting the resulting representation from XYZ to a raw-like space.

For more details, please refer to the supplemental materials of the [paper](https://arxiv.org/abs/2512.08564) (Secs. I.10 and I.11).

<p align="center">
      <img src="../figures/io.gif" width="800" style="max-width: 100%; height: auto;">
</p>

Trained Raw–JPEG Adapter models and the linearization model are provided in the [`models`](models) directory.


---

### ✉️ Inquiries

For inquiries about raw embedding or image linearization, please contact Mahmoud Afifi (m.3afifi@gmail.com)
