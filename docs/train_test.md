
# Training and Testing

## 1. Model Training

NOTE: Requires ~40GB GPU memory.

Configure GPUs via `accelerate config`:

- pix2pix-Turbo (relighting): 4 GPUs → see `train_paired.sh`
- CycleGAN-Turbo (I2I): 8 GPUs → see `train_unpaired.sh`

<br>

## 2. Model Testing

## 2.1 Pretrained Models (img2img-Turbo)
- [BDD100K day2night](https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl)
- [BDD100K night2day](https://www.cs.cmu.edu/~img2img-turbo/models/night2day.pkl) *(optional, can reuse day2night in reverse)*
- [BDD100K clear2rainy](https://www.cs.cmu.edu/~img2img-turbo/models/clear2rainy.pkl)
- [BDD100K rainy2clear](https://www.cs.cmu.edu/~img2img-turbo/models/rainy2clear.pkl) *(optional, can reuse clear2rainy in reverse)*


## 2.2 Our Models (Warp–Unwarp)

### Pix2Pix-Turbo (Human Relighting)
| Lighting | HF Download |
|----------|----------|
| Foggy | [foggy_1.pkl](https://huggingface.co/ShenZheng2000/WarpI2I/resolve/main/pix2pix_turbo/exp_1_10_1_warped_128_eyes/foggy_1.pkl) |
| Golden Sunlight | [golden_sunlight_1.pkl](https://huggingface.co/ShenZheng2000/WarpI2I/resolve/main/pix2pix_turbo/exp_1_10_1_warped_128_eyes/golden_sunlight_1.pkl) |
| Moonlight | [moonlight_1.pkl](https://huggingface.co/ShenZheng2000/WarpI2I/resolve/main/pix2pix_turbo/exp_1_10_1_warped_128_eyes/moonlight_1.pkl) |
| Noon Sunlight | [noon_sunlight_1.pkl](https://huggingface.co/ShenZheng2000/WarpI2I/resolve/main/pix2pix_turbo/exp_1_10_1_merged_warped_128_eyes/noon_sunlight_1.pkl) |

### Pix2Pix-Turbo (Driving Relighting)
| Lighting | HF Download |
|----------|----------|
| Foggy | [foggy_1.pkl](https://huggingface.co/ShenZheng2000/WarpI2I/resolve/main/pix2pix_turbo/2_24_drive_v2_warped_128/foggy_1.pkl) |
| Golden Sunlight | [golden_sunlight_1.pkl](https://huggingface.co/ShenZheng2000/WarpI2I/resolve/main/pix2pix_turbo/2_24_drive_v2_warped_128/golden_sunlight_1.pkl) |

### CycleGAN-Turbo (Driving I2I)
| Task | HF Download |
|------|----------|
| BDD100K clear→rainy | [BDD100K_clear2rainy_warped_128.pkl](https://huggingface.co/ShenZheng2000/WarpI2I/resolve/main/cyclegan_turbo/BDD100K_clear2rainy_warped_128.pkl) |
| BDD100K day→night | [BDD100K_day2night.pkl](https://huggingface.co/ShenZheng2000/WarpI2I/resolve/main/cyclegan_turbo/BDD100K_day2night.pkl) |
| Cityscapes→ACDC fog | [cityscapes_to_acdc_fog_warped_128.pkl](https://huggingface.co/ShenZheng2000/WarpI2I/resolve/main/cyclegan_turbo/cityscapes_to_acdc_fog_warped_128.pkl) |
| Cityscapes→Dark Zurich | [cityscapes_to_dark_zurich_warped_128.pkl](https://huggingface.co/ShenZheng2000/WarpI2I/resolve/main/cyclegan_turbo/cityscapes_to_dark_zurich_warped_128.pkl) |

## 2.3 Inference
See `inf_paired.sh` for human relighting and driving scene relighting with pix2pix-Turbo.

See `inf_unpaired.sh` for driving scene weather and time-of-day translation with CycleGAN-Turbo. 
