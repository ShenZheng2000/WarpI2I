import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# cached_download was removed in huggingface_hub>=0.28; patch it back for diffusers==0.25.1
import huggingface_hub as _hf_hub
if not hasattr(_hf_hub, "cached_download"):
    _hf_hub.cached_download = _hf_hub.hf_hub_download

import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from huggingface_hub import hf_hub_download

from pix2pix_turbo import Pix2Pix_Turbo
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform
from warp_utils.warp_pipeline import (
    get_face_app, detect_face_bbox,
    apply_forward_warp, apply_unwarp,
    resize_longest_side,
)

# ── checkpoint registry ───────────────────────────────────────────
HF_REPO = "ShenZheng2000/WarpI2I"

# ── demo examples ─────────────────────────────────────────────────
DEMO_IDS = ["00006_00", "00017_00", "00055_00", "00069_00"]

def _download_examples():
    imgs, masks = [], []
    for name in DEMO_IDS:
        img_path  = hf_hub_download(repo_id=HF_REPO, filename=f"examples/VITON/image/{name}.jpg")
        mask_path = hf_hub_download(repo_id=HF_REPO, filename=f"examples/VITON/fg_masks/{name}.png")
        imgs.append(img_path)
        masks.append(mask_path)
    return imgs, masks

HUMAN_CKPTS = {
    "Moonlight":       "pix2pix_turbo/exp_1_10_1_warped_128_eyes/moonlight_1.pkl",
    "Golden Sunlight": "pix2pix_turbo/exp_1_10_1_warped_128_eyes/golden_sunlight_1.pkl",
    "Foggy":           "pix2pix_turbo/exp_1_10_1_warped_128_eyes/foggy_1.pkl",
    "Noon Sunlight":   "pix2pix_turbo/exp_1_10_1_merged_warped_128_eyes/noon_sunlight_1.pkl",
}

DRIVE_RELIGHT_CKPTS = {
    "Foggy":           "pix2pix_turbo/2_24_drive_v2_warped_128/foggy_1.pkl",
    "Golden Sunlight": "pix2pix_turbo/2_24_drive_v2_warped_128/golden_sunlight_1.pkl",
}

CYCLEGAN_CKPTS = {
    "Day → Night":        ("cyclegan_turbo/BDD100K_day2night.pkl",                   "driving in the night"),
    "Clear → Rainy":      ("cyclegan_turbo/BDD100K_clear2rainy_warped_128.pkl",      "driving in heavy rain"),
    "Cityscapes → Foggy": ("cyclegan_turbo/cityscapes_to_acdc_fog_warped_128.pkl",   "driving in heavy fog"),
    "Cityscapes → Dark":  ("cyclegan_turbo/cityscapes_to_dark_zurich_warped_128.pkl","driving in the night"),
}

LIGHTING_PROMPTS = {
    "Moonlight":       "Relit with cold moonlight in a minimalist nighttime scene, casting crisp soft shadows and bathing the subject in icy blue highlights to create a tranquil, distant mood.",
    "Golden Sunlight": "Relit with warm golden sunlight during the late afternoon, casting gentle directional shadows and surrounding the subject in soft amber tones to create a calm, radiant mood.",
    "Foggy":           "Relit with dense fog in a muted outdoor setting, casting soft diffused shadows and surrounding the subject in pale gray light to create a quiet, atmospheric mood.",
    "Noon Sunlight":   "Relit with bright noon sunlight in a clear outdoor setting, casting soft natural shadows and surrounding the subject in crisp white light to create a clean, vibrant daytime mood.",
}

# ── model + face-detector cache ───────────────────────────────────
_model_cache = {}
_face_app = None


def get_face_detector():
    global _face_app
    if _face_app is None:
        _face_app = get_face_app()
    return _face_app


def load_pix2pix(hf_path):
    if hf_path not in _model_cache:
        ckpt = hf_hub_download(repo_id=HF_REPO, filename=hf_path)
        m = Pix2Pix_Turbo(pretrained_path=ckpt)
        m.set_eval()
        m.cpu()
        _model_cache[hf_path] = m
    return _model_cache[hf_path]


def load_cyclegan(hf_path):
    if hf_path not in _model_cache:
        ckpt = hf_hub_download(repo_id=HF_REPO, filename=hf_path)
        m = CycleGAN_Turbo(pretrained_path=ckpt)
        m.eval()
        try:
            m.unet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        m.cpu()
        _model_cache[hf_path] = m
    return _model_cache[hf_path]


def inference_with_model(model, fn):
    """Move model to GPU, run fn(), move back to CPU."""
    model.cuda()
    try:
        result = fn()
    finally:
        model.cpu()
        torch.cuda.empty_cache()
    return result


def to_pil(output_tensor):
    return transforms.ToPILImage()(output_tensor[0].cpu() * 0.5 + 0.5)


# ── tab 1: human relighting ───────────────────────────────────────
def _crop_to_fg(image_pil, mask_pil):
    """Crop image to tight foreground bbox; returns (cropped_img, cropped_size)."""
    from PIL import ImageOps
    mask_gray = mask_pil.convert("L")
    inverted  = ImageOps.invert(mask_gray)
    bbox = inverted.getbbox()
    if bbox:
        cropped = image_pil.crop(bbox)
        return cropped, cropped.size
    return image_pil, image_pil.size


def run_human_relight(image_pil, fg_mask_pil, lighting_choice):
    model = load_pix2pix(HUMAN_CKPTS[lighting_choice])
    prompt = LIGHTING_PROMPTS[lighting_choice]

    img = image_pil.convert("RGB")
    cropped_size = None
    if fg_mask_pil is not None:
        img, cropped_size = _crop_to_fg(img, fg_mask_pil)

    img = img.resize((784, 784), Image.LANCZOS)
    c_t = TF.to_tensor(img).unsqueeze(0)

    bbox = detect_face_bbox(img, get_face_detector(), include_eyes=True)

    def _run():
        x = c_t.cuda()
        with torch.no_grad():
            if bbox is not None:
                warped, warp_grid, _ = apply_forward_warp(x, bbox.cuda(), bw=128, separable=True)
                out = model(warped, prompt)
                out = apply_unwarp(warp_grid, out, separable=True)
            else:
                out = model(x, prompt)
        return out

    output = inference_with_model(model, _run)
    output_pil = to_pil(output)
    if cropped_size is not None:
        output_pil = resize_longest_side(output_pil, cropped_size, 784)
    return output_pil


# ── tab 2: driving relighting ─────────────────────────────────────
def run_drive_relight(image_pil, lighting_choice):
    model = load_pix2pix(DRIVE_RELIGHT_CKPTS[lighting_choice])
    prompt = LIGHTING_PROMPTS[lighting_choice]

    img = image_pil.convert("RGB").resize((512, 512), Image.LANCZOS)
    c_t = TF.to_tensor(img).unsqueeze(0)

    def _run():
        with torch.no_grad():
            return model(c_t.cuda(), prompt)

    return to_pil(inference_with_model(model, _run))


# ── tab 3: driving domain transfer ───────────────────────────────
T_val = build_transform("resize_256x256")

def run_drive_transfer(image_pil, transfer_choice):
    hf_path, caption = CYCLEGAN_CKPTS[transfer_choice]
    model = load_cyclegan(hf_path)

    img = T_val(image_pil.convert("RGB"))
    x_t = transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(img)).unsqueeze(0)

    def _run():
        with torch.no_grad():
            return model(x_t.cuda(), direction="a2b", caption=caption)

    return to_pil(inference_with_model(model, _run))


# ── UI ────────────────────────────────────────────────────────────
_demo_imgs, _demo_masks = _download_examples()

with gr.Blocks(title="WarpI2I Demo") as demo:
    gr.Markdown("# WarpI2I — Relighting & Domain Transfer")

    with gr.Tab("Human Relighting"):
        h_light = gr.Dropdown(choices=list(HUMAN_CKPTS), value="Moonlight", label="Lighting Style")
        with gr.Row():
            h_input  = gr.Image(type="pil", label="Input Portrait")
            h_mask   = gr.Image(type="pil", label="Foreground Mask (optional)")
            h_output = gr.Image(type="pil", label="Output")
        h_btn = gr.Button("Run")
        h_btn.click(run_human_relight, [h_input, h_mask, h_light], h_output)
        gr.Examples(
            examples=[[img, mask] for img, mask in zip(_demo_imgs, _demo_masks)],
            inputs=[h_input, h_mask],
            label="Example Portraits",
        )

    with gr.Tab("Driving Relighting"):
        with gr.Row():
            with gr.Column():
                dr_input  = gr.Image(type="pil", label="Input Driving Scene")
                dr_light  = gr.Dropdown(choices=list(DRIVE_RELIGHT_CKPTS), value="Foggy", label="Lighting Style")
                dr_btn    = gr.Button("Run")
            dr_output = gr.Image(type="pil", label="Output")
        dr_btn.click(run_drive_relight, [dr_input, dr_light], dr_output)

    with gr.Tab("Driving Domain Transfer"):
        gr.Markdown("🚧 **Coming Soon** — This feature is still under development.")
        with gr.Row():
            with gr.Column():
                t_input  = gr.Image(type="pil", label="Input Driving Scene", interactive=False)
                t_choice = gr.Dropdown(choices=list(CYCLEGAN_CKPTS), value="Day → Night", label="Transfer", interactive=False)
                t_btn    = gr.Button("Run", interactive=False)
            t_output = gr.Image(type="pil", label="Output", interactive=False)

if __name__ == "__main__":
    demo.launch()
