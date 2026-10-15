"""Supplementary S2.4: target-indexed corner views with a shared view budget.

Generate top-left, top-right, bottom-left and bottom-right crops per target;
concatenate in target order and keep at most M views. Crop dimensions use
nearest-integer rounding. The runner resizes each crop to native input size.
"""
from PIL import Image, ImageFilter
from .config import ETConfig


def corner_views(image, targets, cfg=ETConfig()):
    if not targets or any(not str(target).strip() for target in targets):
        return
    image = image.convert("RGB")
    width, height = image.size
    cw = max(1, min(width, round(width * cfg.crop_ratio)))
    ch = max(1, min(height, round(height * cfg.crop_ratio)))
    boxes = (("top_left", (0, 0, cw, ch)),
             ("top_right", (width-cw, 0, width, ch)),
             ("bottom_left", (0, height-ch, cw, height)),
             ("bottom_right", (width-cw, height-ch, width, height)))
    count = 0
    for target in targets[:cfg.k_targets]:
        for name, box in boxes:
            if count >= cfg.max_views:
                return
            yield f"{target}:{name}", image.crop(box)
            count += 1


def weaken_gaussian(image, radius):
    if radius <= 0:
        raise ValueError("Gaussian weakening radius must be positive")
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius))


def resize_regional_view(image, native_size):
    """Resize a regional crop directly to (width, height) visual input size."""
    if (not isinstance(native_size, (tuple, list)) or len(native_size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in native_size)):
        raise ValueError("Native visual input size must contain two positive integers")
    return image.convert("RGB").resize(tuple(native_size), Image.Resampling.BICUBIC)

