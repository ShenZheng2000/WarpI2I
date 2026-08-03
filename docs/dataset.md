# Dataset Preparation


## 1. Download Datasets


### 1.1 Unpaired Data
For **BDD100K**, download `100K Images` and `Labels` from [here](http://bdd-data.berkeley.edu/download.html) and `coco_labels` from [here](https://drive.google.com/drive/folders/1Hqf1S_I2Q_PG77wD8GGgRN0Z7h8ocqCc?usp=drive_link).

For **Cityscapes**, download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` from [here](https://www.cityscapes-dataset.com/downloads/), and `gt_detection` from [here](https://drive.google.com/drive/folders/1yYBRz96Xf_Hld9DWu-4I-DvQuHtMVvUd?usp=drive_link).

For **Dark Zurich**, download [`Dark_Zurich_train_anon.zip`](https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_train_anon.zip), and [`Dark_Zurich_val_anno.zip`](https://drive.google.com/file/d/1cM6X0WpqUdOceGRILvlPUdASVdA-zyQi/view?usp=drive_link). 

For **ACDC**, download `rgb_anon_trainvaltest.zip` and `gt_trainval.zip` from [here](https://acdc.vision.ee.ethz.ch/download), and `gt_detection` from [here](https://drive.google.com/drive/folders/1LwJwM3heHy-U9u9bfpNEl8h0f9_5yJw4?usp=drive_link).


### 1.2 Paired Test Data
For **VITON-HD**, download test images from [here](https://drive.google.com/file/d/1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w/view).

For **ROADWork** (boston), download test images from [here](https://drive.google.com/file/d/11weqyiw3ODjwyG1aWklqYhVNk-hsOPwY/view?usp=drive_link).

### 1.3 Paired Train Data (Synthetic)
Paired training data for **pix2pix-Turbo** is generated using our relighting pipeline. *The pipeline will be made public soon — stay tuned.*


<details><summary><strong>📂 Dataset Structure</strong></summary>

```
Datasets/
├── relighting/
│   ├── VITON/
│   │   ├── test/
│   │   │   └── image/
│   │   └── train_debug_100/
│   │       └── image/
│   └── workzone_segm/
│       ├── boston/
│       │   └── image/
│       └── pittsburgh/
│           └── image/
└── driving/
    ├── BDD100K/
    │   └── 100k/
    │       ├── coco_labels/
    │       ├── images/
    │       └── labels/
    ├── cityscapes/
    │   ├── gt_detection/
    │   └── leftImg8bit/
    ├── dark_zurich/
    │   └── rgb_anon/
    └── acdc/
        ├── gt_detection/
        └── rgb_anon/
```

</details>
<br>



## 2. Split and Convert datasets

For splitting BDD100K (day/night) and (clear/rainy), and converting to img2img-turbo format, see `shen_scripts/prepare_bdd100k_dataset.py`.

For converting other datasets, see `shen_scripts/prepare_driving_dataset.py`.
<details><summary><strong>📂 Dataset Structure</strong></summary>

```
/home/shenzhen/Datasets/driving/BDD100K_clear2rainy
├── train_A
│ ├── 0.png
│ └── ...
├── train_B
│ ├── 0.png
│ └── ...
├── test_A
│ ├── 0.png
│ └── ...
├── test_B
│ ├── 0.png
│ └── ...
├── train_prompts.json
└── test_prompts.json
```

</details>
<br>


## 3. Warp Images on Salient Regions

See `warp.sh` for example commands. 
<details><summary><strong>📂 Dataset Structure</strong></summary>

```
/home/shenzhen/Datasets/driving/BDD100K_clear2rainy
├── train_A
│ ├── 0.png
│ ├── 0.inv.pth  <-- NEW: Inverse grid files added here
│ └── ...
├── train_B
│ ├── 0.png
│ ├── 0.inv.pth  <-- NEW: Inverse grid files added here
│ └── ...
├── test_A
│ ├── 0.png
│ └── ...
├── test_B
│ ├── 0.png
│ └── ...
├── train_prompts.json
└── test_prompts.json
```

</details>
<br>