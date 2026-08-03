import os
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def write_fixed_prompts(output_root, prompt_A, prompt_B):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Ensure each txt has ONLY ONE line (no internal newlines)
    prompt_A = " ".join(str(prompt_A).splitlines()).strip()
    prompt_B = " ".join(str(prompt_B).splitlines()).strip()

    (output_root / "fixed_prompt_a.txt").write_text(prompt_A + "\n")
    (output_root / "fixed_prompt_b.txt").write_text(prompt_B + "\n")


def copy_images_recursive(src_dir, dst_dir, beta=None, dry_run=False):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    count = 0

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    for path in src_dir.rglob("*"):
        if path.suffix.lower() in IMG_EXTS:
            # If beta is specified, only keep files whose filename contains "_beta_{beta}"
            if beta is not None and f"_beta_{beta}" not in path.name:
                continue

            count += 1

            if dry_run:
                continue

            # dst_path = dst_dir / path.name
            name = path.name
            if beta is not None:
                name = name.replace(f"_foggy_beta_{beta}", "")
            dst_path = dst_dir / name

            # avoid overwrite collisions
            if dst_path.exists():
                stem = path.stem
                suffix = path.suffix
                i = 1
                while dst_path.exists():
                    dst_path = dst_dir / f"{stem}_{i}{suffix}"
                    i += 1

            shutil.copy2(path, dst_path)

    return count


def main(
    train_A_root,
    test_A_root,
    train_B_root,
    test_B_root,
    output_root,
    prompt_A,
    prompt_B,
    beta_B=None,
    dry_run=False,
):
    train_A_root = Path(train_A_root)
    test_A_root = Path(test_A_root)
    train_B_root = Path(train_B_root)
    test_B_root = Path(test_B_root)
    output_root = Path(output_root)

    write_fixed_prompts(output_root, prompt_A, prompt_B)

    stats = {}

    # A
    stats["train_A"] = copy_images_recursive(
        train_A_root,
        output_root / "train_A",
        beta=None,
        dry_run=dry_run,
    )
    stats["test_A"] = copy_images_recursive(
        test_A_root,
        output_root / "test_A",
        beta=None,
        dry_run=dry_run,
    )

    # B (optionally filter Foggy Cityscapes by beta)
    stats["train_B"] = copy_images_recursive(
        train_B_root,
        output_root / "train_B",
        beta=beta_B,
        dry_run=dry_run,
    )
    stats["test_B"] = copy_images_recursive(
        test_B_root,
        output_root / "test_B",
        beta=beta_B,
        dry_run=dry_run,
    )

    print("\n📊 Summary:")
    for k, v in stats.items():
        print(f"  {k:8s}: {v:,} images")

    if dry_run:
        print("\n🧪 Dry run only — no files copied.")
    else:
        print("\n✅ Copy finished.")


# Example 1: Cityscapes -> ACDC foggy
# main(
#     train_A_root="/ssd0/shenzhen/Datasets/driving/cityscapes/leftImg8bit/train",
#     test_A_root="/ssd0/shenzhen/Datasets/driving/cityscapes/leftImg8bit/val",
#     train_B_root="/ssd0/shenzhen/Datasets/driving/dark_zurich/rgb_anon/train/night",
#     test_B_root="/ssd0/shenzhen/Datasets/driving/dark_zurich/rgb_anon/val/night",
#     output_root="/ssd0/shenzhen/Datasets/driving/cityscapes_to_dark_zurich",
#     prompt_A="driving in the day",
#     prompt_B="driving in the night",
#     beta_B=None,
#     dry_run=False,
# )

# # Example 2: Cityscapes -> ACDC foggy (no beta filter)
# main(
#     train_A_root="/ssd0/shenzhen/Datasets/driving/cityscapes/leftImg8bit/train",
#     test_A_root="/ssd0/shenzhen/Datasets/driving/cityscapes/leftImg8bit/val",
#     train_B_root="/ssd0/shenzhen/Datasets/driving/acdc/rgb_anon/fog/train",
#     test_B_root="/ssd0/shenzhen/Datasets/driving/acdc/rgb_anon/fog/val",
#     output_root="/ssd0/shenzhen/Datasets/driving/cityscapes_to_acdc_fog",
#     prompt_A="driving in the day",
#     prompt_B="driving in heavy fog",
#     beta_B=None,
#     dry_run=False,
# )