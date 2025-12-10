# 📊 Evaluation

We provide a standalone evaluation script that reports PSNR, SSIM, LPIPS, and Delta E 2000 between result images and their corresponding ground-truth images.  
This script is useful for benchmarking external methods. To evaluate our method specifically, please refer to [`main`](../main#-testing).

To evaluate a set of result images against ground-truth images, use:

```bash
python evaluation.py \
    --result_dir /path/to/result/image/folder \
    --gt_dir /path/to/ground-truth/image/folder \
    --device gpu   # options: gpu or cpu
```

---

### ✉️ Inquiries

For inquiries related to the evaluation of other methods reported in the [paper](https://arxiv.org/abs/2512.08564), please contact Ran Zhang (ran.zhang@samsung.com).
