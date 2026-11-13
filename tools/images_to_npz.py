import argparse
import os
from typing import List, Tuple

import numpy as np
from PIL import Image


def list_pngs(input_dir: str, recursive: bool = False) -> List[str]:
    if recursive:
        paths = []
        for root, _, files in os.walk(input_dir):
            for name in files:
                if name.lower().endswith(".png"):
                    paths.append(os.path.join(root, name))
    else:
        paths = [
            os.path.join(input_dir, name)
            for name in os.listdir(input_dir)
            if name.lower().endswith(".png")
        ]
    paths.sort()
    return paths


def load_image_as_uint8_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)  # H x W x 3 in [0,255]
    return arr


def infer_shape(paths: List[str]) -> Tuple[int, int, int]:
    if not paths:
        raise ValueError("No PNG images found.")
    arr = load_image_as_uint8_rgb(paths[0])
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unexpected image shape for {paths[0]}: {arr.shape}")
    height, width, channels = arr.shape
    return height, width, channels


def save_npz_from_directory(
    input_dir: str,
    out_path: str = "",
    recursive: bool = False,
    limit: int = 0,
    progress_interval: int = 1000,
):
    paths = list_pngs(input_dir, recursive=recursive)
    if limit and limit > 0:
        paths = paths[:limit]
    num_images = len(paths)
    if num_images == 0:
        raise ValueError(f"No PNG images found in: {input_dir}")

    h, w, c = infer_shape(paths)

    # 预分配 (N, H, W, C)，uint8，等价于参考代码中 .to(torch.uint8).permute(0,2,3,1).cpu().numpy()
    all_images_np = np.empty((num_images, h, w, c), dtype=np.uint8)

    # 加载并填充
    for idx, p in enumerate(paths):
        arr = load_image_as_uint8_rgb(p)
        if arr.shape != (h, w, c):
            raise ValueError(
                f"Image shape mismatch at {p}: got {arr.shape}, expected {(h, w, c)}"
            )
        all_images_np[idx] = arr
        if progress_interval > 0 and (idx + 1) % progress_interval == 0:
            print(f"Loaded {idx + 1}/{num_images} images...")

    # 确定输出路径
    if not out_path:
        shape_str = "x".join([str(x) for x in all_images_np.shape])
        out_path = os.path.join(input_dir, f"samples_{shape_str}.npz")
    else:
        # 如果是目录，则自动生成文件名
        if os.path.isdir(out_path):
            shape_str = "x".join([str(x) for x in all_images_np.shape])
            out_path = os.path.join(out_path, f"samples_{shape_str}.npz")
        else:
            out_dir = os.path.dirname(out_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

    print(f"Saving to {out_path}")
    # 与参考实现一致：仅保存图像数组本身
    np.savez(out_path, all_images_np)
    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将文件夹中的 PNG 图片读取为 uint8，并按 NHWC 形状保存为 NPZ。\n"
            "转换方式对齐参考：(images * 127.5 + 128).clip(0,255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()\n"
            "由于输入已是保存的 PNG，脚本等价地产生 [0,255] 的 uint8 NHWC 数组。\n"
            "默认递归遍历子目录中的 PNG。"
        )
    )
    parser.add_argument("--input_dir", type=str, required=True, help="包含 PNG 图片的目录")
    parser.add_argument("--out", type=str, default="", help="输出 npz 路径或目录；留空则自动命名")
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="不递归子目录，仅读取顶层目录",
    )
    parser.set_defaults(recursive=True)
    parser.add_argument("--limit", type=int, default=0, help="最多读取的图片数（0 表示不限制）")
    parser.add_argument("--progress-interval", type=int, default=1000, help="进度打印的间隔")
    return parser.parse_args()


def main():
    args = parse_args()
    save_npz_from_directory(
        input_dir=args.input_dir,
        out_path=args.out,
        recursive=args.recursive,
        limit=args.limit,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()

