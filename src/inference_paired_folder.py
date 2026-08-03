import os
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm

from pix2pix_turbo import Pix2Pix_Turbo
from warp_utils.warp_pipeline import (
    detect_face_bbox,
    apply_forward_warp,
    apply_unwarp,
    get_face_app,
    resize_longest_side,
    crop_to_foreground,
    center_crop_pil,
    custom_classes,
    detect_yolo_bbox,
    largest_divisible_by_32_leq,
    resize_keep_aspect,
    resize_keep_aspect_min,
    load_with_inheritance
)

from ultralytics import YOLOWorld

import time
from omegaconf import OmegaConf


# ============================================================
# Config
# ============================================================
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)

    parser.add_argument('--full_image_bbox', action='store_true')

    cli_args = parser.parse_args()

    # load YAML then override with CLI values
    cfg = load_with_inheritance(cli_args.exp_config)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_dict["input_dir"] = cli_args.input_dir
    config_dict["output_dir"] = cli_args.output_dir
    config_dict["prompt"] = cli_args.prompt
    config_dict["model_path"] = cli_args.model_path
    config_dict["full_image_bbox"] = cli_args.full_image_bbox

    args = argparse.Namespace(**config_dict)
    os.makedirs(args.output_dir, exist_ok=True)
    return args


# ============================================================
# Image loading & preprocessing
# ============================================================
def load_and_crop_image(input_path, args):
    """Returns (img, cropped_size)"""

    if args.center_crop:
        img = Image.open(input_path).convert("RGB")

        if args.keep_aspect:
            img = resize_keep_aspect_min(img, args.target_size)  # ensure shortest side >= target

        # crop center square region
        img = center_crop_pil(img, args.target_size, args.target_size)
        cropped_size = (args.target_size, args.target_size)

        # optional resize after crop (e.g. downscale for faster inference)
        if args.bw == 0 and args.crop_resize_size and args.crop_resize_size != args.target_size:
            img = img.resize((args.crop_resize_size, args.crop_resize_size), Image.LANCZOS)

    else:
        # remove background padding and crop to foreground subject
        img, cropped_size = crop_to_foreground(input_path)

        if args.keep_aspect:
            img = resize_keep_aspect(img, args.target_size)
        else:
            img = img.resize((args.target_size, args.target_size), Image.LANCZOS)

    return img, cropped_size


# ============================================================
# Detection
# ============================================================
def get_bbox(img, face_app, yolo_model, args):
    """Detect subject bounding box. Returns bbox or None if not found."""

    if args.use_yoloworld and yolo_model is not None:
        # use YOLO for general object detection
        w, h = img.size
        imgsz = largest_divisible_by_32_leq(min(h, w))  # YOLO requires size divisible by 32
        return detect_yolo_bbox(img, yolo_model, imgsz=imgsz)

    # fallback to face detection
    return detect_face_bbox(img, face_app, include_eyes=args.include_eyes)


# ============================================================
# Warp pipeline
# ============================================================
def save_warp_intermediates(warped, saliency, output_path, cropped_size, args):
    """Save warped image and saliency map as debug outputs."""

    warped_pil = transforms.ToPILImage()(warped[0].cpu().clamp(0, 1))
    if not args.keep_aspect:
        warped_pil = resize_longest_side(warped_pil, cropped_size, args.target_size)
    warped_pil.save(output_path.with_name(output_path.stem + "_warp.png"))

    # normalize saliency to [0,1] and save as grayscale
    sal = saliency.float()[0, 0]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    sal_pil = transforms.ToPILImage()(sal.unsqueeze(0).cpu())
    sal_pil.save(output_path.with_name(output_path.stem + "_saliency.png"))


def save_warp_relit(output_image, output_path, cropped_size, args):
    """Save warped+relit intermediate result for debugging."""

    warped_relit_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
    if not args.keep_aspect:
        warped_relit_pil = resize_longest_side(warped_relit_pil, cropped_size, args.target_size)
    warped_relit_pil.save(output_path.with_name(output_path.stem + "_warp_relight.png"))


def run_warp_relight_unwarp(c_t, bbox, model, output_path, cropped_size, args):
    """Warp → relight → unwarp pipeline. Returns final output tensor."""

    warp_output_shape = (args.crop_resize_size, args.crop_resize_size) if args.crop_resize_size else None

    # stretch subject to fill frame so model sees it at higher resolution
    warped, warp_grid, saliency = apply_forward_warp(
        c_t, bbox.to(c_t.device), args.bw, args.separable, output_shape=warp_output_shape
    )
    save_warp_intermediates(warped, saliency, output_path, cropped_size, args)

    # run relighting on the warped image
    output_image = model(warped, args.prompt)
    save_warp_relit(output_image, output_path, cropped_size, args)

    # unwarp back to original geometry
    return apply_unwarp(warp_grid, output_image, args.separable)


# ============================================================
# Output
# ============================================================
def save_final_output(output_image, output_path, cropped_size, args):
    """Resize back to original dimensions and save final result."""

    output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
    if not args.keep_aspect:
        output_pil = resize_longest_side(output_pil, cropped_size, args.target_size)
    output_pil.save(output_path.with_name(output_path.stem + "_warp_relight_unwarp.png"))
    print(f"✅ Saved results for {output_path.name}")


# ============================================================
# Noise Jitter (for bboxes)
# ============================================================
def jitter_bbox(bbox, img, noise_std):
    if noise_std <= 0:
        return bbox
    img_w, img_h = img.size

    print(f"bbox BEFORE jitter: {bbox}, img_w={img_w}, img_h={img_h}")

    w, h = bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]
    noise = torch.randn_like(bbox) * torch.stack([w, h, w, h], dim=1) * noise_std
    bbox = (bbox + noise).clamp(min=torch.tensor([0, 0, 0, 0], dtype=torch.float32),
                                max=torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32))
    bbox[:, 2] = torch.max(bbox[:, 2], bbox[:, 0] + 1)
    bbox[:, 3] = torch.max(bbox[:, 3], bbox[:, 1] + 1)

    print(f"bbox AFTER jitter: {bbox}, img_w={img_w}, img_h={img_h}")
    
    return bbox


# ============================================================
# Orchestrator
# ============================================================
def process_image(input_path, model, face_app, yolo_model, args):

    img, cropped_size = load_and_crop_image(input_path, args)

    # convert to tensor and move to GPU
    c_t = F.to_tensor(img).unsqueeze(0).cuda()
    if args.use_fp16:
        c_t = c_t.half()

    # mirror input directory structure in output
    rel_dir = input_path.parent.relative_to(args.input_dir)
    out_dir = Path(args.output_dir) / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / input_path.name

    with torch.no_grad():
        if args.bw > 0:
            
            if args.full_image_bbox:
                # use full image as bbox (for debugging or non-human subjects)
                w, h = img.size
                bbox = torch.tensor([[0, 0, w, h]], dtype=torch.float32)
            else:
                bbox = get_bbox(img, face_app, yolo_model, args)

            if bbox is None:
                # no subject found, relight directly without warping
                output_image = model(c_t, args.prompt)
            else:
                # optionally jitter the bbox to add robustness to detection errors
                noise_std = getattr(args, 'bbox_noise_std', 0.0)
                bbox = jitter_bbox(bbox, img, noise_std)

                output_image = run_warp_relight_unwarp(
                    c_t, bbox, model, output_path, cropped_size, args
                )

        else:
            # warp disabled, relight directly
            output_image = model(c_t, args.prompt)

    save_final_output(output_image, output_path, cropped_size, args)


# ============================================================
# Main
# ============================================================
def main():

    args = load_config()

    model = Pix2Pix_Turbo(pretrained_name=None, pretrained_path=args.model_path)
    model.set_eval()
    if args.use_fp16:
        model.half()

    face_app = get_face_app()

    yolo_model = None
    if args.use_yoloworld:
        yolo_model = YOLOWorld(args.yolo_model_path)
        yolo_model.set_classes(custom_classes)

    input_root = Path(args.input_dir)

    # expect images under input_root/image/
    images_dir = input_root / "image"
    if not (images_dir.exists() and images_dir.is_dir()):
        raise RuntimeError(f"❌ Expected images directory at {images_dir}")

    img_list = sorted([
        p for p in images_dir.glob("*")
        if p.suffix.lower() in (".jpg", ".png", ".jpeg", ".webp")
    ])
    if len(img_list) == 0:
        raise RuntimeError("❌ No images found to process.")
    print(f"📁 Found {len(img_list)} images in {images_dir}")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for input_path in tqdm(img_list, desc="Processing"):
        process_image(input_path, model, face_app, yolo_model, args)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    print(f"\n⏱ Total inference time: {total_time:.2f} seconds")
    print(f"⏱ Avg time per image: {total_time / len(img_list):.3f} seconds")


if __name__ == "__main__":
    main()