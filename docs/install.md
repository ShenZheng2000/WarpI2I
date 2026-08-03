# Installation

```
git clone https://github.com/ShenZheng2000/WarpI2I
cd WarpI2I
conda env create -f environment.yaml
conda activate img2img-turbo

pip install huggingface_hub==0.25.0
pip install wandb
pip install vision_aided_loss
pip install accelerate
pip install kornia

pip install insightface==0.7.3
pip install numpy==1.26.4
pip install onnxruntime-gpu==1.17.1

pip install ultralytics==8.4.19
pip install omegaconf
```

**Download YOLOWorld model** (required for driving-scene warping only):
```
yolo download model=yolov8x-world.pt
```
Place the downloaded `yolov8x-world.pt` in the repo root. Alternatively, `YOLOWorld("yolov8x-world.pt")` will auto-download on first run if the file is not found.