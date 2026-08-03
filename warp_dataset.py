import os
os.environ["INSIGHTFACE_FORCE_TORCH"] = "1"
import shutil
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLOWorld
from omegaconf import OmegaConf

# ✅ warp utils
from src.warp_utils.warp_pipeline import (
                                        get_face_app, 
                                        detect_face_bbox, 
                                        get_gt_bbox, 
                                        visualize_bbox, 
                                        load_bbox_map, 
                                        apply_forward_warp,
                                        save_identity_inv_for_image,
                                        custom_classes,
                                        detect_yolo_bbox,
                                        load_with_inheritance
)

from src.warp_utils.warping_layers import invert_grid, save_img_warp, load_img_warp

# ============================================================
# 1. Parse Args
# ============================================================
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config', type=str, required=True)

    # REQUIRED CLI args
    parser.add_argument('--input_root', type=str, required=True)
    parser.add_argument('--target_prefix', type=str, required=True)
    parser.add_argument('--relight_type', type=str, default=None)
    
    cli_args = parser.parse_args()

    # load YAML (for shared settings only)
    cfg = load_with_inheritance(cli_args.exp_config)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # override / inject CLI values
    config_dict["input_root"] = cli_args.input_root
    config_dict["target_prefix"] = cli_args.target_prefix
    config_dict["relight_type"] = cli_args.relight_type
    config_dict["exp_config"] = cli_args.exp_config   # ✅ ADD THIS

    args = argparse.Namespace(**config_dict)

    return args


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--target_prefix", type=str, required=True)
#     p.add_argument("--relight_type", type=str, default=None)
#     p.add_argument("--bw", type=int, default=128)
#     p.add_argument("--no-separable", action="store_false", dest="separable")
#     p.add_argument("--include-eyes", action="store_true")
#     p.add_argument("--input_root", type=str, default="/home/shenzhen/Datasets/relighting")
#     p.add_argument("--bbox-json", nargs="+", default=None)
#     p.add_argument("--warp-subfolders", nargs="+", default=["train_A"],
#                 help="List of subfolders to warp (others will be copied only)")
#     p.add_argument("--use-yoloworld", action="store_true",
#                 help="Use YOLO-World to detect bbox on-the-fly (no bbox json needed).")
#     p.add_argument("--yolo-model-path", type=str, default=None)

#     p.set_defaults(separable=True)
#     return p.parse_args()


# ============================================================
# 2. Main pipeline
# ============================================================
class WarpDatasetPipeline:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bandwidth_scale = args.bw

        # --------------- paths ----------------
        if args.relight_type:
            self.input_root = os.path.join(args.input_root, args.target_prefix, args.relight_type)
        else:
            self.input_root = os.path.join(args.input_root, args.target_prefix)

        # mode_tag = "" if args.separable else "_nonsep"
        # eye_tag = "_eyes" if args.include_eyes else ""
        # self.output_root = os.path.join(
        #     args.input_root,
        #     f"{args.target_prefix}_warped_{self.bandwidth_scale}{eye_tag}{mode_tag}"
        # )

        self.output_root = os.path.join(
            args.input_root,
            os.path.splitext(os.path.basename(args.exp_config))[0]
        )

        if args.relight_type:
            self.output_root = os.path.join(self.output_root, args.relight_type)
        os.makedirs(self.output_root, exist_ok=True)

        # --------------- bbox mode ----------------
        self.bbox_map = load_bbox_map(args.bbox_json)

        if args.use_yoloworld:
            self.bbox_map = None
            self.face_app = None
            self.yolo_model = YOLOWorld(args.yolo_model_path) # NOTE: use a larger model for better detection (especially small faces)
            self.yolo_model.set_classes(custom_classes)  # hardcoded
            print("✅ Using YOLO-World bbox detector. GT bbox-json and face detector disabled.")
        elif self.bbox_map is not None:
            self.face_app = None
            print("✅ Using merged GT bounding boxes (train + val). Face detector disabled.")
        else:
            self.bbox_map = None
            self.face_app = get_face_app(local_rank=0)
            print("✅ Using face detector (no bbox-jsons provided).")

        print(f"📂 input_root  = {self.input_root}")
        print(f"📂 output_root = {self.output_root}")


    # --------------------------------------------------------
    def process_image(self, img_path, warped_dir):
        img_pil = Image.open(img_path).convert("RGB")
        c_t = load_img_warp(img_path).to(self.device)
        base = os.path.basename(img_path)

        # --- bbox source ---
        if hasattr(self, "yolo_model") and self.yolo_model is not None:
            bbox = detect_yolo_bbox(img_pil, self.yolo_model)
            if bbox is None:
                print("⚠️  No bbox detected by YOLO-World for image:", img_path)
                shutil.copy2(img_path, os.path.join(warped_dir, base))
                save_identity_inv_for_image(c_t, base, warped_dir)   # <-- NEW (one line)
                return None

        elif self.bbox_map is not None:  # GT-only mode (no detector)
            # bbox = self.get_gt_bbox(base, img_pil)
            bbox = get_gt_bbox(base, img_pil, self.bbox_map, device=self.device)
        else:
            # detector mode
            bbox = detect_face_bbox(img_pil, self.face_app, include_eyes=self.args.include_eyes)
            if bbox is None:
                print("⚠️  No face bbox detected for image:", img_path)
                shutil.copy2(img_path, os.path.join(warped_dir, base))
                save_identity_inv_for_image(c_t, base, warped_dir)   # <-- NEW (one line)
                return None
            bbox = bbox.to(self.device)

        # --- visualize bbox ---
        debug_dir = os.path.join(self.output_root, "_debug_bbox")
        visualize_bbox(img_pil, bbox, base, debug_dir)

        # --- warp images and compute inverse grid --
        warped_img, warp_grid, _ = apply_forward_warp(
            c_t, 
            bbox, 
            self.bandwidth_scale, 
            separable=self.args.separable, 
        )

        out_h, out_w = c_t.shape[2], c_t.shape[3]

        inverse_grid = invert_grid(warp_grid, (1, 3, out_h, out_w), separable=self.args.separable)

        # --- save images and grids ---
        base_noext, ext = os.path.splitext(base)
        warped_path = os.path.join(warped_dir, f"{base_noext}{ext}")
        inv_grid_path = os.path.join(warped_dir, f"{base_noext}.inv.pth")
        save_img_warp(warped_img, warped_path)
        torch.save(inverse_grid.cpu(), inv_grid_path)

    # --------------------------------------------------------
    def process_and_warp_folder(self, sub_folder):
        src_dir = os.path.join(self.input_root, sub_folder)
        if not os.path.isdir(src_dir):
            return
        dst_dir = os.path.join(self.output_root, sub_folder)
        os.makedirs(dst_dir, exist_ok=True)

        imgs = sorted([f for f in os.listdir(src_dir)
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        for fname in tqdm(imgs, desc=f"Warping {sub_folder}", ncols=80):
            self.process_image(os.path.join(src_dir, fname), dst_dir)

    def copy_prompts(self):
        for fname in os.listdir(self.input_root):
            if fname.endswith((".json", ".txt")):
                shutil.copy2(
                    os.path.join(self.input_root, fname),
                    os.path.join(self.output_root, fname)
                )

    # --------------------------------------------------------
    def run(self):
        all_subfolders = ["train_A", "train_B", "test_A", "test_B"]
        warp_set = set(self.args.warp_subfolders)

        for sub in all_subfolders:
            if sub in warp_set:
                self.process_and_warp_folder(sub)
            else:
                # copy-only (non-warped)
                src_dir = os.path.join(self.input_root, sub)
                dst_dir = os.path.join(self.output_root, sub)
                if os.path.isdir(src_dir):
                    os.makedirs(dst_dir, exist_ok=True)
                    for fname in os.listdir(src_dir):
                        src_path = os.path.join(src_dir, fname)
                        dst_path = os.path.join(dst_dir, fname)
                        if os.path.isfile(src_path):
                            shutil.copy2(src_path, dst_path)
        self.copy_prompts()
        print(f"✅ Warp done for {sorted(warp_set)}; others copied only.")


# ============================================================
def main():
    # args = parse_args()
    args = load_config()
    WarpDatasetPipeline(args).run()


if __name__ == "__main__":
    main()