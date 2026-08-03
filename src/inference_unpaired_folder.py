import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform
from warp_utils.warp_pipeline import (
                                    get_gt_bbox, 
                                    load_bbox_map, 
                                    apply_forward_warp, 
                                    apply_unwarp,
                                    load_with_inheritance
)
from pathlib import Path
from omegaconf import OmegaConf

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp', '.gif'])


# ============================================================
# Helper: parse YAML config
# ============================================================
def load_config():
    parser = argparse.ArgumentParser()

    # Required CLI args
    parser.add_argument('--exp_config', type=str, required=True)
    parser.add_argument('--direction', type=str, required=True, choices=['a2b', 'b2a'])

    cli_args = parser.parse_args()

    # Load YAML with inheritance
    cfg = load_with_inheritance(cli_args.exp_config)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Extract selected direction config
    if cli_args.direction not in config_dict:
        raise ValueError(f"Direction '{cli_args.direction}' not defined in the YAML config.")

    direction_settings = config_dict.pop(cli_args.direction)

    # Remove the other unused direction block
    unused_direction = 'b2a' if cli_args.direction == 'a2b' else 'a2b'
    config_dict.pop(unused_direction, None)

    # Merge selected direction settings into main config
    config_dict.update(direction_settings)

    # Save direction itself
    config_dict['direction'] = cli_args.direction

    # Convert to Namespace
    args = argparse.Namespace(**config_dict)

    os.makedirs(args.output_dir, exist_ok=True)
    return args


# ============================================================
# Main processing for one image
# ============================================================
def process_image(input_path, output_path, model, bbox_map, args, T_val):
    input_image = Image.open(input_path).convert('RGB')
    orig_w, orig_h = input_image.width, input_image.height
    out_path = Path(output_path)
    # ext = out_path.suffix   # keep original extension (.jpg / .png)
    ext = ".png"   # always save as PNG

    with torch.no_grad():
        input_img = T_val(input_image)
        
        # --- warp / relight / unwarp pipeline ---
        if args.bw > 0:

            x_01 = transforms.ToTensor()(input_img).unsqueeze(0).cuda()   # UNNORMALIZE version of x_t

            # (0) get GT bbox and apply forward warp
            base_name = os.path.basename(input_path)
            bbox = get_gt_bbox(base_name, input_image, bbox_map, device=x_01.device)
            # warped, warp_grid = apply_forward_warp(x_t, bbox, bw=args.bw, separable=args.separable)
            warped_01, warp_grid, saliency = apply_forward_warp(x_01, bbox, bw=args.bw, separable=args.separable)

            # (1) save warped image
            warped_pil = transforms.ToPILImage()(warped_01[0].cpu().clamp(0,1))
            warped_pil = warped_pil.resize((orig_w, orig_h), Image.LANCZOS)
            warped_pil.save(out_path.with_name(out_path.stem + "_warp" + ext))

            # (2) model on warped
            warped_norm = (warped_01 * 2.0 - 1.0)
            output_image = model(warped_norm, direction=args.direction, caption=args.prompt)

            # (3) save warped + relit
            warped_relit_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
            warped_relit_pil = warped_relit_pil.resize((orig_w, orig_h), Image.LANCZOS)
            warped_relit_pil.save(out_path.with_name(out_path.stem + "_warp_relight" + ext))

            # (4) unwarp back
            output_image = apply_unwarp(warp_grid, output_image, separable=args.separable)

        else:
            # no warp → just model(x_t)
            x_t = transforms.ToTensor()(input_img)
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
            output_image = model(x_t, direction=args.direction, caption=args.prompt)

        # (5) ALWAYS save final as warp_relight_unwarp
        final_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        final_pil = final_pil.resize((orig_w, orig_h), Image.LANCZOS)
        final_pil.save(out_path.with_name(out_path.stem + "_warp_relight_unwarp" + ext))


# ============================================================
# Main
# ============================================================
def main():
    # args = parse_args()
    args = load_config()

    # model = CycleGAN_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model = CycleGAN_Turbo(pretrained_name=None, pretrained_path=args.model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()

    T_val = build_transform(args.image_prep)

    os.makedirs(args.output_dir, exist_ok=True)

    # bbox_map = load_bbox_map(args.train_bbox_json, args.val_bbox_json)
    bbox_map = load_bbox_map(args.bbox_json)

    # gather images
    image_paths = []
    for root, _, files in os.walk(args.input_dir):
        for filename in files:
            if is_image_file(filename):
                image_paths.append(os.path.join(root, filename))

    # run
    for input_path in tqdm(image_paths, desc="Processing images", ncols=100):
        relative_root = os.path.relpath(os.path.dirname(input_path), args.input_dir)
        if relative_root == ".":
            relative_root = ""
        output_folder = os.path.join(args.output_dir, relative_root)
        os.makedirs(output_folder, exist_ok=True)

        filename = os.path.basename(input_path)
        output_path = os.path.join(output_folder, filename)

        # process
        process_image(input_path, output_path, model, bbox_map, args, T_val)


if __name__ == "__main__":
    main()