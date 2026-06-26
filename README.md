
<div align="center">

# 🌀 WarpI2I: Image Warping for Image-to-Image Translation

<h3>🔥 ECCV 2026</h3>

[Shen Zheng](https://shenzheng2000.github.io/), [Anurag Ghosh](https://anuragxel.github.io/), [Gaurav Parmar](https://gauravparmar.com/), and [Srinivasa Narasimhan](https://www.cs.cmu.edu/~srinivas/)

Carnegie Mellon University

<a href='#'><img src='https://img.shields.io/badge/arXiv-Coming_Soon-red'></a>&nbsp;
<a href='https://shenzheng2000.github.io/WarpI2I.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>&nbsp;
<img src='https://visitor-badge.laobi.icu/badge?page_id=WarpI2I.visitors&left_color=green&right_color=red' alt='visitors'>

</div>


# 🎬 Videos

<table>
  <tr>
    <td width="33%" align="center"><b>Input</b></td>
    <td width="33%" align="center"><b>Golden Sunlight</b></td>
    <td width="34%" align="center"><b>Foggy</b></td>
  </tr>
  <tr>
    <td colspan="3">
      <video src="assets/demos/demo_1.mp4" autoplay loop muted playsinline controls width="100%"></video>
    </td>
  </tr>
  <tr>
    <td colspan="3">
      <video src="assets/demos/demo_2.mp4" autoplay loop muted playsinline controls width="100%"></video>
    </td>
  </tr>
  <tr>
    <td colspan="3">
      <video src="assets/demos/demo_3.mp4" autoplay loop muted playsinline controls width="100%"></video>
    </td>
  </tr>
</table>


# 🤗 Overview

![Architecture](assets/images/ECCV26_Warping_Architecture_NEW.png)

We warp the input image to enlarge small salient regions (e.g., objects, faces, eyes) to better preserve fine details in the compressed latent space during image-to-image translation. In the figure above, original latents are shown in 🟥, and warped latents in 🟩.


# 🌟 Highlights
* **Detail-preserving** — preserves fine details in latent diffusion models.
* **Model-agnostic** — no architectural modifications needed.
* **Ultra-efficient** — only **0.006s** additional latency and **zero** extra learnable parameters.



## 📌 TODO Lists

- [ ] Add demo
- [ ] Add Train & Test code
- [ ] Add arXiv link



## ⭐️ Star History
[![Star History Chart](https://api.star-history.com/svg?repos=ShenZheng2000/WarpI2I&type=Date)](https://star-history.com/#ShenZheng2000/WarpI2I&Date)



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


## 🙏 Acknowledgement

This project builds upon the following excellent open-source works:
- [img2img-turbo](https://github.com/GaParmar/img2img-turbo)
- [Instance-Warp](https://github.com/ShenZheng2000/Instance-Warp)
- [Two-Plane Prior](https://github.com/geometriczoom/two-plane-prior)