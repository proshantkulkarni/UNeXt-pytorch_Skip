import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# ----------------------------- CONFIG -----------------------------
EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

# Colors in BGR for OpenCV, will convert to RGB for plotting
COLOR_GT      = (0, 255, 0)     # Green
COLOR_BASE    = (255, 0, 0)     # Red
COLOR_MLFC    = (0, 0, 255)     # Blue

COLOR_BASE_ONLY      = (255, 0, 255)  # Magenta  (baseline vs mlfc)
COLOR_MLFC_ONLY      = (0, 255, 255)  # Cyan     (baseline vs mlfc)

COLOR_FP_BASE_VS_GT  = (255, 0, 0)    # Red   (baseline predicts lesion, GT says background)
COLOR_FN_BASE_VS_GT  = (0, 255, 0)    # Green (GT lesion, baseline missed)

# ----------------------------- PATHS ------------------------------
images_dir   = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic2/test/images'
gt_dir       = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic2/test/masks'
baseline_dir = '/content/drive/MyDrive/Prashant/UNeXt-pytorch_Skip/models/Baseline_early_50'
mlfc_dir     = '/content/drive/MyDrive/Prashant/UNeXt-pytorch_Skip/models/MLFC'

output_folder = '/content/drive/MyDrive/Prashant/UNeXt-pytorch_Skip/models/Baseline_vs_MLFC'
os.makedirs(output_folder, exist_ok=True)

# --------------------------- UTILITIES ----------------------------
def get_filenames(directory, extensions=EXTENSIONS):
    names = set()
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        if os.path.isfile(path):
            name, ext = os.path.splitext(f)
            if ext.lower() in extensions:
                names.add(name)
    return names

def find_common_files(dirs):
    common = None
    for d in dirs:
        fn = get_filenames(d)
        if common is None:
            common = fn
        else:
            common &= fn
    return sorted(list(common))

def find_file(base_dir, name, extensions=EXTENSIONS):
    for ext in extensions:
        p = os.path.join(base_dir, name + ext)
        if os.path.isfile(p):
            return p
    return None

def load_image_rgb(path):
    return np.array(Image.open(path).convert('RGB'))

def load_mask_binary(path):
    # Load as grayscale and binarize (>0 -> 1)
    m = np.array(Image.open(path).convert('L'))
    return (m > 0).astype(np.uint8)

def ensure_same_size(img, *masks):
    h, w = img.shape[:2]
    resized = []
    for m in masks:
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        resized.append(m)
    return resized

def draw_contours_on(image_rgb, mask_bin, color_bgr, thickness=2):
    """Draw contours of mask_bin on top of image_rgb. Returns a new image_rgb."""
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mask_uint8 = (mask_bin * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_bgr, contours, -1, color_bgr, thickness)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def boundary_overlay(image_rgb, gt, base, mlfc):
    out = image_rgb.copy()
    out = draw_contours_on(out, gt, COLOR_GT, thickness=2)
    out = draw_contours_on(out, base, COLOR_BASE, thickness=2)
    out = draw_contours_on(out, mlfc, COLOR_MLFC, thickness=2)
    return out

def diff_baseline_vs_gt(base, gt):
    """
    Red:  baseline predicts lesion, GT background (FP)
    Green: GT lesion, baseline background (FN)
    """
    fp = (base == 1) & (gt == 0)
    fn = (gt == 1) & (base == 0)

    h, w = base.shape
    diff = np.zeros((h, w, 3), dtype=np.uint8)
    diff[fp] = COLOR_FP_BASE_VS_GT[::-1]  # convert BGR->RGB for visualization
    diff[fn] = COLOR_FN_BASE_VS_GT[::-1]
    return diff

def diff_baseline_vs_mlfc(base, mlfc):
    """
    Magenta: baseline only
    Cyan:    mlfc only
    """
    base_only = (base == 1) & (mlfc == 0)
    mlfc_only = (mlfc == 1) & (base == 0)

    h, w = base.shape
    diff = np.zeros((h, w, 3), dtype=np.uint8)
    diff[base_only] = COLOR_BASE_ONLY[::-1]  # BGR->RGB
    diff[mlfc_only] = COLOR_MLFC_ONLY[::-1]
    return diff

def save_panel(image_rgb, gt, boundary_img, base_vs_gt, base_vs_mlfc, save_path, title=None):
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    panels = [
        (image_rgb, 'Image'),
        (gt * 255, 'GT'),
        (boundary_img, 'Boundary Overlay'),
        (base_vs_gt, 'Baseline - GT'),
        (base_vs_mlfc, 'Baseline - MLFC')
    ]

    for ax, (im, t) in zip(axes, panels):
        if im.ndim == 2:  # grayscale
            ax.imshow(im, cmap='gray')
        else:
            ax.imshow(im)
        ax.set_title(t if title is None else t)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)

# ----------------------------- MAIN ------------------------------
def main():
    folders = [images_dir, gt_dir, baseline_dir, mlfc_dir]
    common_files = find_common_files(folders)
    print(f"Found {len(common_files)} common files.")

    for name in common_files:
        img_p   = find_file(images_dir, name)
        gt_p    = find_file(gt_dir, name)
        base_p  = find_file(baseline_dir, name)
        mlfc_p  = find_file(mlfc_dir, name)

        if None in [img_p, gt_p, base_p, mlfc_p]:
            print(f"[SKIP] Missing file for {name}")
            continue

        image_rgb = load_image_rgb(img_p)
        gt_bin    = load_mask_binary(gt_p)
        base_bin  = load_mask_binary(base_p)
        mlfc_bin  = load_mask_binary(mlfc_p)

        gt_bin, base_bin, mlfc_bin = ensure_same_size(image_rgb, gt_bin, base_bin, mlfc_bin)

        boundary_img   = boundary_overlay(image_rgb, gt_bin, base_bin, mlfc_bin)
        base_vs_gt_img = diff_baseline_vs_gt(base_bin, gt_bin)
        base_vs_mlfc_img = diff_baseline_vs_mlfc(base_bin, mlfc_bin)

        save_path = os.path.join(output_folder, f"{name}_comparison.jpg")
        save_panel(image_rgb, gt_bin, boundary_img, base_vs_gt_img, base_vs_mlfc_img, save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
