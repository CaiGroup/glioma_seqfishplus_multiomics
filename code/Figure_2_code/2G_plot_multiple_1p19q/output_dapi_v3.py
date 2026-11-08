import os
import numpy as np
import tifffile
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Configuration (matches the scatter plot script)
# ---------------------------------------------------------------------------
DAPI_PATH = "/home/jjfox/resnick_ci2_gbm_/V2_runs/25-06-13_2122_2543_4668_dna/roi_1/dapi_max_aligned.tiff"
OUT_DIR = "../../../figures/Figure_2/Figure_2C"
OUT_NAME_FULL = "full_dapi_with_roi.png"

PIXELS_PER_MICRON = 9.62

# Same regions as the scatter plot, in microns: (label, x_range, y_range)
REGIONS = [
    ("region1", (2150, 2450), (650, 950)),
    ("region2", (2800, 3100), (750, 1050)),
]

# Match the rendered scatter resolution. The scatter pipeline now renders at
# TARGET_PT * DPI / 72 pixels on the long edge (99 pt at 1200 DPI = ~1650 px),
# so the DAPI crops will composite pixel-for-pixel with the scatter PNGs.
TARGET_PT = 99
DPI = 1200
SCATTER_LONG_EDGE_PX = int(round(TARGET_PT * DPI / 72))  # ~1650 px

# Contrast stretch percentiles for DAPI display
PCT_LOW = 1.0
PCT_HIGH = 99.5

# Full-image overview output settings
FULL_MAX_DIM = 2000          # longest edge in pixels for the overview PNG
BOX_LINE_WIDTH = 6           # white box stroke width in overview pixels

# ---------------------------------------------------------------------------
# Helpers (mirrors logic in the scatter plot script)
# ---------------------------------------------------------------------------
def should_rotate(x_range, y_range):
    """Rotate iff the region is taller than wide (portrait), so the rendered
    image is always landscape (wider than tall)."""
    return (y_range[1] - y_range[0]) > (x_range[1] - x_range[0])

def stretch_to_uint8(arr, pct_low=PCT_LOW, pct_high=PCT_HIGH):
    """Percentile contrast stretch -> uint8."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [pct_low, pct_high])
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)

def region_to_pixel_box(x_range, y_range, w, h):
    """Convert a micron (x_range, y_range) to clamped pixel bounds (x0, x1, y0, y1)."""
    x0 = int(round(x_range[0] * PIXELS_PER_MICRON))
    x1 = int(round(x_range[1] * PIXELS_PER_MICRON))
    y0 = int(round(y_range[0] * PIXELS_PER_MICRON))
    y1 = int(round(y_range[1] * PIXELS_PER_MICRON))

    x0 = max(0, min(x0, w))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h))
    y1 = max(0, min(y1, h))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"Region out of bounds: x_px=({x0}, {x1}), y_px=({y0}, {y1}), "
            f"image is {w} x {h} (W x H)"
        )
    return x0, x1, y0, y1

def crop_region(img, x_range, y_range, label):
    """Crop, contrast-stretch, optionally rotate, and resize a region.

    Returns the PIL Image and its (x0, x1, y0, y1) full-image pixel bounds.
    """
    h, w = img.shape
    x0_px, x1_px, y0_px, y1_px = region_to_pixel_box(x_range, y_range, w, h)

    crop = img[y0_px:y1_px, x0_px:x1_px]
    print(f"[{label}] cropped shape (H, W): {crop.shape}")

    crop8 = stretch_to_uint8(crop)

    if should_rotate(x_range, y_range):
        crop8 = np.rot90(crop8, k=1)
        print(f"[{label}] rotated 90 deg CCW to match landscape scatter orientation")

    h_out, w_out = crop8.shape
    if w_out >= h_out:
        new_w = SCATTER_LONG_EDGE_PX
        new_h = int(round(h_out * (SCATTER_LONG_EDGE_PX / w_out)))
    else:
        new_h = SCATTER_LONG_EDGE_PX
        new_w = int(round(w_out * (SCATTER_LONG_EDGE_PX / h_out)))

    pil_img = Image.fromarray(crop8, mode="L").resize(
        (new_w, new_h), resample=Image.LANCZOS
    )
    return pil_img, (x0_px, x1_px, y0_px, y1_px), (new_w, new_h)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    img = tifffile.imread(DAPI_PATH)
    if img.ndim > 2:
        img = np.squeeze(img)
        if img.ndim > 2:
            img = img.max(axis=tuple(range(img.ndim - 2)))

    h, w = img.shape
    print(f"DAPI image shape (H, W): {h} x {w}")

    # -----------------------------------------------------------------------
    # 1) Per-region cropped PNGs
    # -----------------------------------------------------------------------
    region_pixel_bounds = []  # collect to draw overview boxes
    for label, region_x, region_y in REGIONS:
        pil_img, px_bounds, (new_w, new_h) = crop_region(
            img, region_x, region_y, label
        )
        out_path = os.path.join(OUT_DIR, f"{label}_dapi.png")
        pil_img.save(out_path)
        print(f"[{label}] saved {out_path} ({new_w} x {new_h})")
        region_pixel_bounds.append((label, px_bounds))

    # -----------------------------------------------------------------------
    # 2) Full DAPI overview with one white ROI box per region
    # -----------------------------------------------------------------------
    full8 = stretch_to_uint8(img)

    long_edge = max(h, w)
    scale = FULL_MAX_DIM / long_edge if long_edge > FULL_MAX_DIM else 1.0
    new_full_w = int(round(w * scale))
    new_full_h = int(round(h * scale))

    full_pil = Image.fromarray(full8, mode="L").resize(
        (new_full_w, new_full_h), resample=Image.LANCZOS
    )
    full_rgb = full_pil.convert("RGB")
    draw = ImageDraw.Draw(full_rgb)

    for label, (x0_px, x1_px, y0_px, y1_px) in region_pixel_bounds:
        box_x0 = int(round(x0_px * scale))
        box_x1 = int(round(x1_px * scale))
        box_y0 = int(round(y0_px * scale))
        box_y1 = int(round(y1_px * scale))
        draw.rectangle(
            [(box_x0, box_y0), (box_x1, box_y1)],
            outline=(255, 255, 255),
            width=BOX_LINE_WIDTH,
        )
        print(f"[{label}] drew overview box at {(box_x0, box_y0, box_x1, box_y1)}")

    out_path_full = os.path.join(OUT_DIR, OUT_NAME_FULL)
    full_rgb.save(out_path_full)
    print(
        f"Saved full DAPI overview with {len(region_pixel_bounds)} ROI boxes "
        f"to {out_path_full} ({new_full_w} x {new_full_h})"
    )

if __name__ == "__main__":
    main()