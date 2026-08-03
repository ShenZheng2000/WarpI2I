<div align="center">

# 🌀 WarpI2I: Image Warping for Image-to-Image Translation

<h3>🔥 ECCV 2026</h3>

[Shen Zheng](https://shenzheng2000.github.io/), [Anurag Ghosh](https://anuragxel.github.io/), [Gaurav Parmar](https://gauravparmar.com/), and [Srinivasa Narasimhan](https://www.cs.cmu.edu/~srinivas/)

Carnegie Mellon University

<a href='https://arxiv.org/abs/2606.31018'><img src='https://img.shields.io/badge/arXiv-2606.31018-red'></a>&nbsp;
<a href='https://shenzheng2000.github.io/WarpI2I.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>&nbsp;
<a href='https://huggingface.co/spaces/ShenZheng2000/WarpI2I-demo'><img src='https://img.shields.io/badge/🤗%20HuggingFace-Demo-yellow'></a>&nbsp;
<img src='https://visitor-badge.laobi.icu/badge?page_id=WarpI2I.visitors&left_color=green&right_color=red' alt='visitors'>

</div>


## 📢 News

* **2026/08/02**: Demo Code is released.
* **2026/06/30**: Our paper is available at [arXiv](https://arxiv.org/abs/2606.31018).


## 👀 Overview

![Architecture](assets/images/ECCV26_Warping_Architecture_NEW.png)

We warp the input image to enlarge small salient regions (e.g., objects, faces, eyes) to better preserve fine details in the compressed latent space during image-to-image translation. In the figure above, original latents are shown in 🟥, and warped latents in 🟩.


## 🌟 Highlights
* **Detail-preserving** — preserves fine details in latent diffusion models.
* **Model-agnostic** — no architectural modifications needed.
* **Ultra-efficient** — only **0.006s** additional latency and **zero** extra learnable parameters.


## 🎬 Videos

Please check our [Project Page](https://shenzheng2000.github.io/WarpI2I.github.io) for more details.


## 🤗 HuggingFace Demo

Try the interactive demo directly in your browser — no installation needed:

👉 **[https://huggingface.co/spaces/ShenZheng2000/WarpI2I-demo](https://huggingface.co/spaces/ShenZheng2000/WarpI2I-demo)**

The demo covers all three tasks via separate tabs:

| Tab | Input | Options |
|-----|-------|---------|
| **Human Relighting** | Portrait photo | Moonlight / Golden Sunlight / Foggy / Noon Sunlight |
| **Driving Relighting** | Driving scene | Foggy / Golden Sunlight |
| **Driving Domain Transfer** | Driving scene | Day→Night / Clear→Rainy / Cityscapes→Foggy / Cityscapes→Dark |

To run the demo locally:
```bash
pip install gradio huggingface_hub
python app.py
# then open http://127.0.0.1:7860
```


## ⚙️ Installation

See [docs/install.md](docs/install.md).


## 📥 Dataset Preparation

See [docs/dataset.md](docs/dataset.md).


## 🚀 Training and Testing

See [docs/train_test.md](docs/train_test.md).


## 🔌 Warp–Unwarp Integration

Key warp–unwarp code insertions are marked with **✅**.

The main insertion points are:
- Data loading → `src/my_utils/training_utils.py` (`PairedDataset`, `UnpairedDataset`)
- Pix2Pix-Turbo → `src/train_pix2pix_turbo.py`
- CycleGAN-Turbo → `src/train_cyclegan_turbo.py`
- Warp utilities → `src/warp_utils/`


## ✍️ Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zheng2026warpi2i,
  title     = {WarpI2I: Image Warping for Image-to-Image Translation},
  author    = {Zheng, Shen and Ghosh, Anurag and Parmar, Gaurav and Narasimhan, Srinivasa},
  booktitle = {ECCV},
  year      = {2026}
}
```

## 🤝 Contact

If you have any questions, feel free to raise an issue (recommended) or send an email to `shenzhen@andrew.cmu.edu`.


## 🙏 Acknowledgement

This project builds upon the following excellent open-source works:
- [img2img-turbo](https://github.com/GaParmar/img2img-turbo)
- [Instance-Warp](https://github.com/ShenZheng2000/Instance-Warp)
- [Two-Plane Prior](https://github.com/geometriczoom/two-plane-prior)