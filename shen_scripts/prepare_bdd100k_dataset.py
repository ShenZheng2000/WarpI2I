#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

IMG_EXTS = [".jpg", ".jpeg", ".png"]

def norm(x) -> str:
    return str(x).strip().lower()

def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def read_attrs(jpath: Path) -> tuple[str, str, str]:
    data = json.loads(jpath.read_text())
    attrs = data.get("attributes") or {}
    name = data.get("name") or jpath.stem
    timeofday = norm(attrs.get("timeofday", "undefined"))
    weather   = norm(attrs.get("weather", "undefined"))
    return name, timeofday, weather

def copy(img: Path, dst_dir: Path, dry_run: bool):
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / img.name
    if dry_run:
        print(f"[DRY] {img} -> {dst}")
    else:
        shutil.copy2(img, dst)

def write_fixed_prompts(out_root: Path, prompt_a: str, prompt_b: str):
    out_root.mkdir(parents=True, exist_ok=True)
    prompt_a = " ".join(str(prompt_a).splitlines()).strip()
    prompt_b = " ".join(str(prompt_b).splitlines()).strip()
    (out_root / "fixed_prompt_a.txt").write_text(prompt_a + "\n")
    (out_root / "fixed_prompt_b.txt").write_text(prompt_b + "\n")

def parse_kv(kvs: list[str]) -> dict[str, str]:
    d = {}
    for kv in kvs:
        k, v = kv.split("=", 1)
        d[norm(k)] = norm(v)
    return d

def out_prefix(split: str) -> str:
    # img2img-turbo convention: train -> train_*, val -> test_*
    return "train" if split == "train" else "test"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Root with images/{train,val,...} and labels/{train,val,...}")
    ap.add_argument("--out", type=Path, required=True, help="Output root containing train_A/train_B/test_A/test_B + fixed prompts")
    ap.add_argument("--split", nargs="+", default=["train", "val"], help="Splits to process; val will be written as test_*")
    ap.add_argument("--A", nargs="+", required=True, help='Filter for domain A, e.g. timeofday=daytime weather=clear')
    ap.add_argument("--B", nargs="+", required=True, help='Filter for domain B, e.g. timeofday=night weather=clear')
    ap.add_argument("--prompt_a", type=str, required=True, help='fixed_prompt_a.txt content, e.g. "driving in the day"')
    ap.add_argument("--prompt_b", type=str, required=True, help='fixed_prompt_b.txt content, e.g. "driving in the night"')
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    A = parse_kv(args.A)
    B = parse_kv(args.B)

    write_fixed_prompts(args.out, args.prompt_a, args.prompt_b)

    # ✅ counters
    counts = {
        "train_A": 0,
        "train_B": 0,
        "test_A": 0,
        "test_B": 0,
    }

    for sp in args.split:
        labels_dir = args.root / "labels" / sp
        images_dir = args.root / "images" / sp
        if not labels_dir.exists() or not images_dir.exists():
            print(f"[WARN] missing dirs for split={sp}: {labels_dir} or {images_dir}")
            continue

        prefix = out_prefix(sp)
        out_A = args.out / f"{prefix}_A"
        out_B = args.out / f"{prefix}_B"

        for jpath in labels_dir.glob("*.json"):
            try:
                name, tod, wth = read_attrs(jpath)
            except Exception as e:
                print(f"[WARN] bad json {jpath}: {e}")
                continue

            img = find_image(images_dir, name) or find_image(images_dir, jpath.stem)
            if img is None:
                continue

            attrs = {"timeofday": tod, "weather": wth}
            if all(attrs.get(k) == v for k, v in A.items()):
                copy(img, out_A, args.dry_run)
                counts[f"{prefix}_A"] += 1
            elif all(attrs.get(k) == v for k, v in B.items()):
                copy(img, out_B, args.dry_run)
                counts[f"{prefix}_B"] += 1

    # ✅ final summary
    print("\n📊 Summary:")
    for k in ["train_A", "train_B", "test_A", "test_B"]:
        print(f"  {k:8s}: {counts[k]:,} images")


if __name__ == "__main__":
    main()

"""
Example (Day -> Night, clear only):
python prepare_bdd100k_dataset.py \
  --root /ssd0/shenzhen/Datasets/driving/BDD100K/100k \
  --out  /ssd0/shenzhen/Datasets/driving/BDD100K_day2night \
  --split train val \
  --A timeofday=daytime weather=clear \
  --B timeofday=night   weather=clear \
  --prompt_a "driving in the day" \
  --prompt_b "driving in the night"

Example (Clear -> Rainy, daytime only):
python prepare_bdd100k_dataset.py \
  --root /ssd0/shenzhen/Datasets/driving/BDD100K/100k \
  --out  /ssd0/shenzhen/Datasets/driving/BDD100K_clear2rainy \
  --split train val \
  --A timeofday=daytime weather=clear \
  --B timeofday=daytime weather=rainy \
  --prompt_a "driving in the day" \
  --prompt_b "driving in heavy rain"
"""