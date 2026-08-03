import os
import random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.nn.functional import interpolate
from glob import glob
from pathlib import Path

def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_clipsim", default=5.0, type=float)

    # dataset options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=8, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=5000,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def parse_args_unpaired_training():
    """
    Parses command-line arguments used for configuring an unpaired session (CycleGAN-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """

    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")

    # fixed random seed
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_idt", default=1, type=float)
    parser.add_argument("--lambda_cycle", default=1, type=float)
    parser.add_argument("--lambda_cycle_lpips", default=10.0, type=float)
    parser.add_argument("--lambda_idt_lpips", default=1.0, type=float)

    # args for dataset and dataloader options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_img_prep", required=True)
    parser.add_argument("--val_img_prep", required=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)

    # args for the model
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sd-turbo")
    parser.add_argument("--revision", default=None, type=str)
    parser.add_argument("--variant", default=None, type=str)
    parser.add_argument("--lora_rank_unet", default=128, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # args for validation and logging
    parser.add_argument("--viz_freq", type=int, default=20)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, required=True)
    parser.add_argument("--validation_steps", type=int, default=500,)
    parser.add_argument("--validation_num_images", type=int, default=-1, help="Number of images to use for validation. -1 to use all images.")
    parser.add_argument("--checkpointing_steps", type=int, default=5000)

    # args for the optimization options
    parser.add_argument("--learning_rate", type=float, default=5e-6,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.",)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # memory saving options
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--gradient_checkpointing", action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")

    parser.add_argument("--restricted_pairs", action="store_true",
                        help="Use source-stem-matched target instead of random target pairing.")

    args = parser.parse_args()
    return args


# def build_transform(image_prep):
#     """
#     Constructs a transformation pipeline based on the specified image preparation method.

#     Parameters:
#     - image_prep (str): A string describing the desired image preparation

#     Returns:
#     - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
#     """
#     if image_prep == "resized_crop_512":
#         T = transforms.Compose([
#             transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
#             transforms.CenterCrop(512),
#         ])
#     elif image_prep == "resize_286_randomcrop_256x256_hflip":
#         T = transforms.Compose([
#             transforms.Resize((286, 286), interpolation=Image.LANCZOS),
#             transforms.RandomCrop((256, 256)),
#             transforms.RandomHorizontalFlip(),
#         ])
#     elif image_prep in ["resize_256", "resize_256x256"]:
#         T = transforms.Compose([
#             transforms.Resize((256, 256), interpolation=Image.LANCZOS)
#         ])
#     elif image_prep in ["resize_512", "resize_512x512"]:
#         T = transforms.Compose([
#             transforms.Resize((512, 512), interpolation=Image.LANCZOS)
#         ])
#     # NOTE: add resize of 784 and 384 for the relight dataset
#     elif image_prep in ["resize_784"]:
#         T = transforms.Compose([
#             transforms.Resize((784, 784), interpolation=Image.LANCZOS)
#         ])
#     elif image_prep in ["resize_384"]:
#         T = transforms.Compose([
#             transforms.Resize((384, 384), interpolation=Image.LANCZOS)
#         ])
#     elif image_prep in ["resize_416"]:
#         T = transforms.Compose([
#             transforms.Resize((416, 416), interpolation=Image.LANCZOS)
#         ])
#     elif image_prep in ["resize_448"]:
#         T = transforms.Compose([
#             transforms.Resize((448, 448), interpolation=Image.LANCZOS)
#         ])
#     elif image_prep in ["resize_480"]:
#         T = transforms.Compose([
#             transforms.Resize((480, 480), interpolation=Image.LANCZOS)
#         ])
#     elif image_prep == "no_resize":
#         T = transforms.Lambda(lambda x: x)
#     return T


# ------------------------------------------------
# 🔹 Helper functions for grids
# ------------------------------------------------
def _resize_grid(grid, size):
    """
    Resize flow/grid tensor to new size.

    Accepts:
      - [H, W, 2]
      - [1, H, W, 2]

    Returns:
      - [1, H_new, W_new, 2]
    """
    # make sure we have a batch dim
    if grid.dim() == 3:
        grid = grid.unsqueeze(0)  # [1, H, W, 2]

    # [1, H, W, 2] -> [1, 2, H, W]
    grid = grid.permute(0, 3, 1, 2)

    # resize
    grid = interpolate(grid, size=size, mode="bilinear", align_corners=True)  # [1, 2, H', W']

    # back to [1, H', W', 2]
    grid = grid.permute(0, 2, 3, 1)
    return grid


def _crop_grid(grid, i, j, h, w):
    """
    Crop grid using same params as image crop.

    Accepts [H, W, 2] or [1, H, W, 2].
    Returns [1, h, w, 2].
    """
    if grid.dim() == 3:
        grid = grid.unsqueeze(0)  # [1, H, W, 2]

    grid = grid[:, i:i + h, j:j + w, :]
    return grid


def _flip_grid(grid):
    """
    Horizontal flip + negate x-coordinates.

    Accepts [H, W, 2] or [1, H, W, 2].
    Returns [1, H, W, 2].
    """
    if grid.dim() == 3:
        grid = grid.unsqueeze(0)  # [1, H, W, 2]

    # flip along width axis (dim=2)
    grid = torch.flip(grid, dims=[2])
    grid[..., 0] = -grid[..., 0]
    return grid


# ------------------------------------------------
# 🔹 Unified Resize / Crop / Flip Transform
# ------------------------------------------------
def _transform_resize_crop_flip(inputs, resize_size=None, crop_size=None, hflip_prob=0.0):
    """
    Apply Resize (+ optional RandomCrop + optional RandomHorizontalFlip)
    to both image and grid, keeping spatial consistency.
    """
    if isinstance(inputs, tuple):
        img, grid = inputs
    else:
        img, grid = inputs, None

    # # ✅ Optional Resize
    # img = F.resize(img, resize_size, interpolation=Image.LANCZOS)
    # if grid is not None:
    #     grid = _resize_grid(grid, resize_size)

    # ✅ Resize only if resize_size is not None
    if resize_size is not None:
        img = F.resize(img, resize_size, interpolation=Image.LANCZOS)
        if grid is not None:
            grid = _resize_grid(grid, resize_size)

    # ✅ Optional RandomCrop
    if crop_size is not None:
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=crop_size)
        img = F.crop(img, i, j, h, w)
        if grid is not None:
            grid = _crop_grid(grid, i, j, h, w)

    # ✅ Optional RandomHorizontalFlip
    if hflip_prob > 0 and torch.rand(1) < hflip_prob:
        img = F.hflip(img)
        if grid is not None:
            grid = _flip_grid(grid)

    return (img, grid) if grid is not None else img



# ------------------------------------------------
# 🔹 Unified Transform (no nested function)
# ------------------------------------------------
def build_transform(image_prep):
    """
    Unified transform builder (auto grid support).
    Returns a callable that applies the same operations to (img) or (img, grid).
    """
    if image_prep == "resize_286_randomcrop_256x256_hflip":
        return lambda x: _transform_resize_crop_flip(
            x, resize_size=(286, 286), crop_size=(256, 256), hflip_prob=0.5
        )

    elif image_prep == "randomcrop_256x256_hflip":
        return lambda x: _transform_resize_crop_flip(
            x, resize_size=None, crop_size=(256, 256), hflip_prob=0.5
        )

    elif image_prep in ["resize_256", "resize_256x256"]:
        return lambda x: _transform_resize_crop_flip(x, resize_size=(256, 256))

    elif image_prep in ["resize_512", "resize_512x512"]:
        return lambda x: _transform_resize_crop_flip(x, resize_size=(512, 512))

    elif image_prep == "no_resize":
        return lambda x: x

    else:
        raise NotImplementedError(f"Unsupported image_prep: {image_prep}")



class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        Itialize the paired dataset object for loading and transforming paired data samples
        from specified dataset folders.

        This constructor sets up the paths to input and output folders based on the specified 'split',
        loads the captions (or prompts) for the input images, and prepares the transformations and
        tokenizer to be applied on the data.

        Parameters:
        - dataset_folder (str): The root folder containing the dataset, expected to include
                                sub-folders for different splits (e.g., 'train_A', 'train_B').
        - split (str): The dataset split to use ('train' or 'test'), used to select the appropriate
                       sub-folders and caption files within the dataset folder.
        - image_prep (str): The image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        if split == "train":
            self.input_folder = os.path.join(dataset_folder, "train_A")
            self.output_folder = os.path.join(dataset_folder, "train_B")
            captions = os.path.join(dataset_folder, "train_prompts.json")
        elif split == "test":
            self.input_folder = os.path.join(dataset_folder, "test_A")
            self.output_folder = os.path.join(dataset_folder, "test_B")
            captions = os.path.join(dataset_folder, "test_prompts.json")
        with open(captions, "r") as f:
            self.captions = json.load(f)
        self.img_names = list(self.captions.keys())
        self.T = build_transform(image_prep)
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.captions)

    def __getitem__(self, idx):
        """
        Retrieves a dataset item given its index. Each item consists of an input image, 
        its corresponding output image, the captions associated with the input image, 
        and the tokenized form of this caption.

        This method performs the necessary preprocessing on both the input and output images, 
        including scaling and normalization, as well as tokenizing the caption using a provided tokenizer.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        dict: A dictionary containing the following key-value pairs:
            - "output_pixel_values": a tensor of the preprocessed output image with pixel values 
            scaled to [-1, 1].
            - "conditioning_pixel_values": a tensor of the preprocessed input image with pixel values 
            scaled to [0, 1].
            - "caption": the text caption.
            - "input_ids": a tensor of the tokenized caption.

        Note:
        The actual preprocessing steps (scaling and normalization) for images are defined externally 
        and passed to this class through the `image_prep` parameter during initialization. The 
        tokenization process relies on the `tokenizer` also provided at initialization, which 
        should be compatible with the models intended to be used with this dataset.
        """
        img_name = self.img_names[idx]
        input_img = Image.open(os.path.join(self.input_folder, img_name))
        output_img = Image.open(os.path.join(self.output_folder, img_name))
        caption = self.captions[img_name]


        # ✅ detect inverse grid if available
        inv_grid_path = os.path.join(self.input_folder, img_name.replace(".png", ".inv.pth")).replace(".jpg", ".inv.pth")
        inv_grid = (
            torch.load(inv_grid_path, map_location="cpu")
            if os.path.exists(inv_grid_path)
            else torch.empty(0)
        )

        # ✅ input images scaled to 0,1
        # img_t = self.T(input_img)
        if inv_grid.numel() > 0:
            img_t, inv_grid = self.T((input_img, inv_grid))
        else:
            img_t = self.T(input_img)

        img_t = F.to_tensor(img_t)

        # output images scaled to -1,1
        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "input_ids": input_ids,
            "has_inv_grid": inv_grid.numel() > 0,
            "inv_grid": inv_grid,
        }


class UnpairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        A dataset class for loading unpaired data samples from two distinct domains (source and target),
        typically used in unsupervised learning tasks like image-to-image translation.

        The class supports loading images from specified dataset folders, applying predefined image
        preprocessing transformations, and utilizing fixed textual prompts (captions) for each domain,
        tokenized using a provided tokenizer.

        Parameters:
        - dataset_folder (str): Base directory of the dataset containing subdirectories (train_A, train_B, test_A, test_B)
        - split (str): Indicates the dataset split to use. Expected values are 'train' or 'test'.
        - image_prep (str): he image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        if split == "train":
            self.source_folder = os.path.join(dataset_folder, "train_A")
            self.target_folder = os.path.join(dataset_folder, "train_B")
        elif split == "test":
            self.source_folder = os.path.join(dataset_folder, "test_A")
            self.target_folder = os.path.join(dataset_folder, "test_B")
        self.tokenizer = tokenizer
        with open(os.path.join(dataset_folder, "fixed_prompt_a.txt"), "r") as f:
            self.fixed_caption_src = f.read().strip()
            self.input_ids_src = self.tokenizer(
                self.fixed_caption_src, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids

        with open(os.path.join(dataset_folder, "fixed_prompt_b.txt"), "r") as f:
            self.fixed_caption_tgt = f.read().strip()
            self.input_ids_tgt = self.tokenizer(
                self.fixed_caption_tgt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
        # find all images in the source and target folders with all IMG extensions
        self.l_imgs_src = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_src.extend(glob(os.path.join(self.source_folder, ext)))
        self.l_imgs_tgt = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_tgt.extend(glob(os.path.join(self.target_folder, ext)))
        self.T = build_transform(image_prep)

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.l_imgs_src) + len(self.l_imgs_tgt)

    def sample_paths(self, index):
        """Override this to change pairing policy."""
        if index < len(self.l_imgs_src):
            img_path_src = self.l_imgs_src[index]
        else:
            img_path_src = random.choice(self.l_imgs_src)
        img_path_tgt = random.choice(self.l_imgs_tgt)
        return img_path_src, img_path_tgt

    def __getitem__(self, index):
        """
        Fetches a pair of unaligned images from the source and target domains along with their 
        corresponding tokenized captions.

        For the source domain, if the requested index is within the range of available images,
        the specific image at that index is chosen. If the index exceeds the number of source
        images, a random source image is selected. For the target domain,
        an image is always randomly selected, irrespective of the index, to maintain the 
        unpaired nature of the dataset.

        Both images are preprocessed according to the specified image transformation `T`, and normalized.
        The fixed captions for both domains
        are included along with their tokenized forms.

        Parameters:
        - index (int): The index of the source image to retrieve.

        Returns:
        dict: A dictionary containing processed data for a single training example, with the following keys:
            - "pixel_values_src": The processed source image
            - "pixel_values_tgt": The processed target image
            - "caption_src": The fixed caption of the source domain.
            - "caption_tgt": The fixed caption of the target domain.
            - "input_ids_src": The source domain's fixed caption tokenized.
            - "input_ids_tgt": The target domain's fixed caption tokenized.
        """
        # if index < len(self.l_imgs_src):
        #     img_path_src = self.l_imgs_src[index]
        # else:
        #     img_path_src = random.choice(self.l_imgs_src)
        # img_path_tgt = random.choice(self.l_imgs_tgt)
        img_path_src, img_path_tgt = self.sample_paths(index)

        # detect + load inverse grids
        inv_grid_path_src = img_path_src.replace(".png", ".inv.pth").replace(".jpg", ".inv.pth")
        inv_grid_src = (
            torch.load(inv_grid_path_src, map_location="cpu")
            if os.path.exists(inv_grid_path_src)
            else torch.empty(0)
        )
        inv_grid_path_tgt = img_path_tgt.replace(".png", ".inv.pth").replace(".jpg", ".inv.pth")
        inv_grid_tgt = (
            torch.load(inv_grid_path_tgt, map_location="cpu")
            if os.path.exists(inv_grid_path_tgt)
            else torch.empty(0)
        )

        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")

        # img_t_src = F.to_tensor(self.T(img_pil_src))
        # img_t_tgt = F.to_tensor(self.T(img_pil_tgt))
        # ✅ apply unified transform (auto-handles grid if needed)
        if inv_grid_src.numel() > 0:
            img_pil_src, inv_grid_src = self.T((img_pil_src, inv_grid_src))
        else:
            img_pil_src = self.T(img_pil_src)

        if inv_grid_tgt.numel() > 0:
            img_pil_tgt, inv_grid_tgt = self.T((img_pil_tgt, inv_grid_tgt))
        else:
            img_pil_tgt = self.T(img_pil_tgt)

        img_t_src = F.to_tensor(img_pil_src)
        img_t_tgt = F.to_tensor(img_pil_tgt)

        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])
        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt,
            "has_inv_grid_src": inv_grid_src.numel() > 0,  
            "inv_grid_src": inv_grid_src,  
            "has_inv_grid_tgt": inv_grid_tgt.numel() > 0,   
            "inv_grid_tgt": inv_grid_tgt,                 
        }


class RestrictedUnpairedDataset(UnpairedDataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer, sep="__"):
        super().__init__(dataset_folder, split, image_prep, tokenizer)
        self.sep = sep

        # We sample SOURCE first (train_A). Each source has a stem like:
        #   <stem>__crop_<panoid>.jpg
        # Target (train_B) is the snowy image named:
        #   <stem>.png  (or .jpg/.jpeg)
        #
        # Build a fast lookup: stem -> target_path (only for targets that exist)
        self.tgt_by_stem = {}
        for p in self.l_imgs_tgt:
            stem = Path(p).stem   # e.g., "boreas-2021-01-26-..."
            self.tgt_by_stem[stem] = p

        # ---- PRINT ONCE so you know indexing looks right ----
        print("[RestrictedUnpairedDataset] #targets indexed by stem:", len(self.tgt_by_stem))
        # for k in list(self.tgt_by_stem.keys())[:5]:
        #     print(f"  stem={k} -> tgt={os.path.basename(self.tgt_by_stem[k])}")

    def sample_paths(self, index):
        # pick source exactly like the base UnpairedDataset
        img_path_src, _ = super().sample_paths(index)

        # extract stem from source crop filename:
        #   boreas-xxx__crop_yyy.jpg  ->  boreas-xxx
        src_name = os.path.basename(img_path_src)

        if self.sep in src_name:
            # e.g. "boreas-xxx__crop_yyy.jpg" → "boreas-xxx"
            stem = src_name.split(self.sep, 1)[0]
        else:
            # e.g. "boreas-xxx.png" → "boreas-xxx"
            stem = Path(src_name).stem

        # find the matched target (must exist for valid source)
        img_path_tgt = self.tgt_by_stem.get(stem)
        if img_path_tgt is None:
            # reject this source and resample
            new_index = random.randint(0, len(self) - 1)
            return self.sample_paths(new_index)

        # ---- PRINT PAIRS so you can verify training is correct ----
        # print(f"[PAIR] stem={stem} | src={os.path.basename(img_path_src)} -> tgt={os.path.basename(img_path_tgt)}")

        return img_path_src, img_path_tgt