import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from colorblind import colorblind

# paths (update to your paths)
image_path = Path("/home/vishnu/Desktop/Ashoka/sem_5/pos/assignment/images/src")
save_path  = Path("/home/vishnu/Desktop/Ashoka/sem_5/pos/assignment/images/save")
save_path.mkdir(parents=True, exist_ok=True)

image_extensions = ('.png', '.jpg', '.jpeg')
image_files = [p for p in image_path.iterdir() if p.suffix.lower() in image_extensions]

# types we'll generate
types = ["protanopia", "deuteranopia", "tritanopia"]


def simulate_achromatopsia(rgb):
    """Grayscale simulation for monochromacy / achromatopsia."""
    rgb_f = rgb.astype(np.float32)
    Y = (
        0.2126 * rgb_f[..., 0] +
        0.7152 * rgb_f[..., 1] +
        0.0722 * rgb_f[..., 2]
    )
    ach = np.repeat(Y[..., None], 3, axis=-1)
    return np.clip(ach, 0, 255).astype(np.uint8)


for img_path in image_files:
    name = img_path.stem
    out_dir = save_path / name
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"Warning: failed to read {img_path}, skipping.")
        continue
    rgb = bgr[..., ::-1]  # BGR -> RGB

    achromatopsia = simulate_achromatopsia(rgb)
    Image.fromarray(achromatopsia).save(out_dir / "achromatopsia_sim.png")

    Image.fromarray(rgb).save(out_dir / "normal.png")

    for t in types:
        # simulate
        sim_rgb = colorblind.simulate_colorblindness(rgb, colorblind_type=t)
        # ensure uint8
        sim_rgb_u8 = np.clip(sim_rgb, 0, 255).astype(np.uint8)
        Image.fromarray(sim_rgb_u8).save(out_dir / f"{t}_sim.png")
    print(f"Processed {img_path.name} -> {out_dir}")
