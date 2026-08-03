"""
warp_pipeline.py
Reusable warp–unwarp utilities for Pix2Pix-Turbo (used in both train & inference).
"""

import numpy as np
import torch
from .warping_layers import PlainKDEGrid, warp, invert_grid

# optional import (only used for face detection)
from insightface.app import FaceAnalysis
from torchvision import transforms
import torch.nn.functional as F
import os
import json
from PIL import Image, ImageOps
from pathlib import Path
from omegaconf import OmegaConf

custom_classes = [
    # BDD100K classes (excluding train)
    "person","rider","car","truck","bus","motorcycle","bicycle",
    "traffic light","traffic sign",
    # "train",
    # Roadwork classes
    "Police Officer","Police Vehicle","Cone","Fence","Drum",
    "Barricade","Barrier","Work Vehicle","Vertical Panel",
    "Tubular Marker","Arrow Board","Bike Lane","Work Equipment",
    "Worker","Other Roadwork Objects",
    "Temporary Traffic Control Message Board",
    "Temporary Traffic Control Sign"
]

# ===============================================================
# ✅ Initialize face detector once
# ===============================================================
def get_face_app(local_rank=0):
    """Initialize and return a FaceAnalysis detector (explicit init, no lazy caching)."""
    _face_app = FaceAnalysis(name='buffalo_l', providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    _face_app.prepare(ctx_id=local_rank)
    return _face_app

def detect_face_bbox(image_pil, face_app=None, include_eyes=False):
    """
    Detects the largest face in the image (PIL.Image).
    Returns:
        - torch.Tensor of shape [1, 4] if include_eyes=False → face only
        - torch.Tensor of shape [3, 4] if include_eyes=True  → [face, left_eye, right_eye]
    """
    img_cv2 = np.array(image_pil)[:, :, ::-1]
    faces = face_app.get(img_cv2) if face_app is not None else []
    h, w = image_pil.height, image_pil.width

    if faces:
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        face = faces[0]
        x1, y1, x2, y2 = map(int, face.bbox)

        if include_eyes and hasattr(face, "kps") and face.kps is not None:
            left_eye_box, right_eye_box = _compute_eye_boxes(face, x1, y1, x2, y2)
            return torch.tensor([[x1, y1, x2, y2], left_eye_box, right_eye_box], dtype=torch.float32)

        return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)

    # ✅ no face found (FIXED: no face instead of image-center as face-center!)
    return None

def _compute_eye_boxes(face, x1, y1, x2, y2):
    """Compute left/right eye boxes from 5-point keypoints."""
    kps = face.kps.astype(int)
    left_eye, right_eye = kps[0], kps[1]
    eye_w = int((x2 - x1) * 0.25)
    eye_h = int((y2 - y1) * 0.15) # NOTE: use 0.15 instead of 0.20 to avoid too large boxes

    lx1, ly1 = int(left_eye[0] - eye_w / 2), int(left_eye[1] - eye_h / 2)
    lx2, ly2 = int(left_eye[0] + eye_w / 2), int(left_eye[1] + eye_h / 2)
    rx1, ry1 = int(right_eye[0] - eye_w / 2), int(right_eye[1] - eye_h / 2)
    rx2, ry2 = int(right_eye[0] + eye_w / 2), int(right_eye[1] + eye_h / 2)

    return [lx1, ly1, lx2, ly2], [rx1, ry1, rx2, ry2]


def get_gt_bbox(base_name, img_pil, bbox_map, device="cuda"):
    """
    Given:
        base_name : image filename (e.g. 0001.jpg)
        img_pil   : PIL image
        bbox_map  : dict { filename: [ [x,y,w,h], ... ] }
        device    : torch device for tensor output

    Returns:
        bbox tensor of shape [N, 4] with (x1, y1, x2, y2)
        or full-image box if missing.
    """
    if bbox_map is None:
        # fallback: full image
        w_img, h_img = img_pil.size
        return torch.tensor([[0., 0., float(w_img), float(h_img)]], device=device)

    if base_name not in bbox_map:
        # fallback: full image
        w_img, h_img = img_pil.size
        return torch.tensor([[0., 0., float(w_img), float(h_img)]], device=device)

    valid = []
    for (x, y, w, h) in bbox_map[base_name]:
        if w <= 1 or h <= 1 or any(v != v for v in (x, y, w, h)):
            continue
        valid.append([x, y, x + w, y + h])

    if not valid:
        # no valid GT → fallback to full image
        w_img, h_img = img_pil.size
        return torch.tensor([[0., 0., float(w_img), float(h_img)]], device=device)

    return torch.tensor(valid, device=device)


def visualize_bbox(img_pil, bbox_tensor, base_name, save_dir):
    """
    Draw green rectangle(s) on the image for debugging.

    Args:
        img_pil      : input PIL image
        bbox_tensor  : [N, 4] torch tensor
        base_name    : file name for output
        save_dir     : directory where the bbox visualization is saved
    """
    try:
        import cv2
        import numpy as np
        os.makedirs(save_dir, exist_ok=True)

        img_cv = np.array(img_pil)[:, :, ::-1].copy()   # PIL RGB → OpenCV BGR
        for box in bbox_tensor:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out_path = os.path.join(save_dir, base_name)
        cv2.imwrite(out_path, img_cv)
    except Exception as e:
        print(f"[WARN] Failed to visualize bbox for {base_name}: {e}")


def load_bbox_map(json_list):
    """
    Load and merge MANY COCO JSON bbox maps.
    Keyed by basename.
    Crashes on any basename collision.
    """
    if not json_list:
        return None

    merged = {}

    for path in json_list:
        if not path or not os.path.exists(path):
            continue

        with open(path, "r") as f:
            coco = json.load(f)

        # build id -> basename
        id2name = {}
        seen = set()
        for img in coco["images"]:
            base = os.path.basename(img["file_name"])
            if base in seen:
                raise RuntimeError(f"❌ Basename collision INSIDE {path}: {base}")
            seen.add(base)
            id2name[img["id"]] = base

        # collect bboxes
        part = {}
        for ann in coco["annotations"]:
            fn = id2name[ann["image_id"]]
            part.setdefault(fn, []).append(ann["bbox"])

        print(f"📘 Loaded {len(part)} entries from {path}")

        # merge, but forbid overwriting
        dup = set(merged).intersection(part)
        if dup:
            raise RuntimeError(
                f"❌ Basename collision ACROSS JSONs: {sorted(list(dup))[:5]}"
            )

        merged.update(part)

    print(f"✅ Total merged bbox entries: {len(merged)}")

    # ---- DEBUG PREVIEW ----
    print("🔎 [DEBUG] bbox_map preview:")
    keys = list(merged.keys())

    for i, k in enumerate(keys[:4]):
        print(f"  [TOP {i}] {k} -> {merged[k]}")

    if len(keys) > 4:
        print("  ...")

    for i, k in enumerate(keys[-4:]):
        print(f"  [BOT {len(keys)-4+i}] {k} -> {merged[k]}")

    return merged


def resize_longest_side(img_pil, cropped_size, target_size):
    cw, ch = cropped_size
    aspect_ratio = cw / ch
    if aspect_ratio >= 1:
        new_w, new_h = target_size, int(target_size / aspect_ratio)
    else:
        new_w, new_h = int(target_size * aspect_ratio), target_size
    return img_pil.resize((new_w, new_h), Image.LANCZOS)


def find_fg_mask_path(input_path):
    """
    ONLY responsible for finding mask_path.
    Priority matches inference image selection:
    flat > legacy
    """
    input_path = Path(input_path)

    # -------------------------
    # Flat layout (HIGH PRIORITY)
    #   input_dir/images/xxx.jpg
    #   input_dir/fg_masks/xxx.png
    # -------------------------
    if input_path.parent.name == "image":
        root = input_path.parent.parent
        fg_dir = root / "fg_masks"
        if fg_dir.exists():
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                candidate = fg_dir / f"{input_path.stem}{ext}"
                if candidate.exists():
                    return candidate

    # -------------------------
    # Legacy layout (FALLBACK)
    #   seq_x/image.jpg
    #   seq_x/pre_processing/black_fg_mask_groundedsam2.png
    # -------------------------
    legacy_mask = (
        input_path.parent / "pre_processing" / "black_fg_mask_groundedsam2.png"
    )
    if legacy_mask.exists():
        return legacy_mask

    return None


def crop_to_foreground(input_path):
    # input_root = input_path.parent  # e.g. .../8seconds_men_shirts_034
    # mask_path = input_root / "pre_processing" / "black_fg_mask_groundedsam2.png"

    # print(f"\n🟢 Image: {input_path}")
    # print(f"🔍 Mask:  {mask_path}")

    mask_path = find_fg_mask_path(input_path)

    img = Image.open(input_path).convert("RGB")

    if mask_path.exists():
        mask = Image.open(mask_path).convert("L")
        inverted_mask = ImageOps.invert(mask)
        bbox = inverted_mask.getbbox()
        if bbox:
            img = img.crop(bbox)
            # print(f"✅ Cropped to bbox {bbox}")
        else:
            print("⚠️ Empty mask, using full image.")
    else:
        print("⚠️ Mask not found, using full image.")

    return img, img.size


def center_crop_pil(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = img.size
    if w < target_w or h < target_h:
        scale = max(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.BICUBIC)
        w, h = img.size
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def build_identity_inv_grid(H, W, device="cpu", dtype=torch.float32):
    xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # [1,H,W,2]


def save_identity_inv_for_image(image_tensor, base_name, output_dir):
    """
    Create and save identity inverse grid matching image size.

    image_tensor : [1,C,H,W]
    base_name    : original filename (with extension)
    output_dir   : folder to save .inv.pth
    """
    _, _, H, W = image_tensor.shape
    base_noext, _ = os.path.splitext(base_name)

    inv_path = os.path.join(output_dir, f"{base_noext}.inv.pth")

    grid = build_identity_inv_grid(H, W)
    torch.save(grid.cpu(), inv_path)

def detect_yolo_bbox(img_pil, yolo_model, imgsz=768):
    res = yolo_model(img_pil, imgsz=imgsz)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    return res.boxes.xyxy.clone()

def largest_divisible_by_32_leq(x: int) -> int:
    # avoid 0; YOLO needs something reasonable
    return max(32, (x // 32) * 32)

def resize_keep_aspect(img, target_size):
    w, h = img.size
    scale = target_size / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)

    # ensure divisible by 8 for diffusion models
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8

    return img.resize((new_w, new_h), Image.LANCZOS)

def resize_keep_aspect_min(img, target_size):
    w, h = img.size

    # scale so the shorter side becomes target_size
    scale = target_size / min(w, h)

    new_w = round(w * scale)
    new_h = round(h * scale)

    # ensure divisible by 8 for diffusion models
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8

    return img.resize((new_w, new_h), Image.LANCZOS)

def load_with_inheritance(path):
    import os
    cfg = OmegaConf.load(path)

    if "inherit" in cfg:
        parent_path = os.path.join(os.path.dirname(path), cfg.inherit)
        parent_cfg = load_with_inheritance(parent_path)
        cfg = OmegaConf.merge(parent_cfg, cfg)

    return cfg

# ===============================================================
# ✅ Forward warp (returns warped tensor + warp grid)
# ===============================================================
def apply_forward_warp(image_tensor, bbox_tensor, bw, separable=True, output_shape=None):
    """
    Applies KDE-based forward warp around bbox region.
    Returns (warped_tensor, warp_grid).
    """
    device = image_tensor.device
    bbox_tensor = bbox_tensor.to(device)

    _, _, H, W = image_tensor.shape

    # --- NEW LOGIC ---
    if output_shape is None:
        out_H, out_W = H, W
    else:
        out_H, out_W = output_shape

    grid_net = PlainKDEGrid(
        input_shape=(H, W),
        output_shape=(out_H, out_W),   # <--- FIXED HERE
        separable=separable,
        bandwidth_scale=bw,
        amplitude_scale=1.0,
    ).to(device)
    # warp_grid = grid_net(image_tensor, gt_bboxes=bbox_tensor.unsqueeze(0))
    warp_grid, saliency = grid_net(image_tensor, gt_bboxes=bbox_tensor.unsqueeze(0), return_saliency=True)
    warped = warp(warp_grid, image_tensor)
    return warped, warp_grid, saliency

# ===============================================================
# ✅ Unwarp (returns restored tensor)
# ===============================================================
def apply_unwarp(warp_grid, warped_output, separable=True):
    """
    Applies inverse KDE grid to unwarp the warped output tensor.
    """
    device = warped_output.device  # make sure we stay on the same GPU
    _, _, H, W = warped_output.shape

    inv_grid = invert_grid(warp_grid.to(device), (1, 3, H, W), separable=separable)
    restored = F.grid_sample(warped_output, inv_grid.to(device), align_corners=True)
    return restored