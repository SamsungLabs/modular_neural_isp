# 🔼 Upsampling

We chose to perform the photofinishing module in low resolution (1/4 of the original high-resolution input) to improve runtime efficiency, as the photofinishing module (and the external editing functions) may become a bottleneck if applied directly on high-resolution images and negatively impact speed.


<p align="center">
      <img src="../figures/upsampling.gif" width="600" style="max-width: 100%; height: auto;">
</p>


As explained in the main [paper](https://arxiv.org/abs/2512.08564) (Sec. 2.4), we rely on the bilateral guided upsampling ([BGU](https://people.csail.mit.edu/hasinoff/pubs/ChenEtAl16-bgu.pdf)) method. However, instead of using the Halide BGU constraints and regularization, we propose a gated regularization that improves the upsampling results.

The upsampling stage is applied after photofinishing to transfer the low-resolution output of the photofinishing module to the high-resolution domain before applying the subsequent stages in our pipeline.

---

### ✉️ Inquiries
For inquiries about the guided upsampling development or its ablation studies, please contact Zhongling Wang (z.wang2@samsung.com).
