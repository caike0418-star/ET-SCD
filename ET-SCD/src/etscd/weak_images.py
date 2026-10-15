"""Prepare frequency-filtered, blurred, or edge-overlay image inputs."""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
from PIL import Image, __version__ as pillow_version

from .data import read_jsonl
from .views import weaken_gaussian


TRANSFORMS = ("frequency-lowpass", "gaussian-blur", "edge-overlay")


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be finite and positive") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _strength(value):
    if isinstance(value, bool):
        raise ValueError("Edge strength must be between 0 and 1")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("Edge strength must be between 0 and 1") from None
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Edge strength must be between 0 and 1")
    return value


def frequency_lowpass(image, cutoff_ratio=0.16, order=4.0):
    """Apply a radial exponential low-pass mask to each RGB Fourier spectrum."""
    cutoff_ratio = _positive(cutoff_ratio, "Frequency cutoff ratio")
    order = _positive(order, "Frequency filter order")
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width, _ = arr.shape
    yy, xx = np.ogrid[:height, :width]
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    cutoff = max(2.0, cutoff_ratio * min(height, width))
    mask = np.exp(-np.power(radius / cutoff, order)).astype(np.float32)
    out = np.zeros_like(arr)
    for channel in range(3):
        spectrum = np.fft.fftshift(np.fft.fft2(arr[:, :, channel]))
        filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
        out[:, :, channel] = filtered
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def edge_overlay(image, strength=0.58):
    """Blend RGB pixels with percentile-normalized grayscale gradient edges."""
    strength = _strength(strength)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    if min(rgb.shape[:2]) < 2:
        raise ValueError("Edge overlay requires an image at least 2 by 2 pixels")
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx * gx + gy * gy)
    scale = np.percentile(magnitude, 99.0)
    if scale <= 1e-6:
        scale = 1.0
    edges = np.clip(magnitude / scale, 0.0, 1.0) * 255.0
    edges_rgb = np.repeat(edges[:, :, None], 3, axis=2)
    mixed = (1.0 - strength) * rgb + strength * edges_rgb
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8), mode="RGB")


def _parameters(transform, cutoff_ratio, order, blur_radius, strength):
    if transform == "frequency-lowpass":
        return {"cutoff_ratio": _positive(cutoff_ratio, "Frequency cutoff ratio"),
                "order": _positive(order, "Frequency filter order")}
    if transform == "gaussian-blur":
        return {"radius": _positive(blur_radius, "Gaussian blur radius")}
    if transform == "edge-overlay":
        return {"strength": _strength(strength)}
    raise ValueError(f"Unknown image transform: {transform}")


def _image_paths(rows, image_folder):
    if not rows:
        raise ValueError("Empty inference questions")
    image_folder = Path(image_folder).resolve()
    if not image_folder.is_dir():
        raise NotADirectoryError(image_folder)
    identities, names = set(), set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "image", "question"}:
            raise ValueError("Questions must contain only id, image, and question")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in row):
            raise ValueError("id, image, and question must be nonempty strings")
        if row["id"] in identities:
            raise ValueError(f"Duplicate sample ID: {row['id']}")
        identities.add(row["id"])
        name = row["image"]
        relative = PurePosixPath(name)
        if (relative.is_absolute() or PureWindowsPath(name).drive or "\\" in name
                or ".." in relative.parts or relative.as_posix() != name
                or name == "manifest.json"):
            raise ValueError(f"Image must use a relative path within the image folder: {name}")
        names.add(name)
    paths = {}
    for name in sorted(names):
        path = (image_folder / name).resolve()
        if not path.is_relative_to(image_folder):
            raise ValueError(f"Image path escapes the declared image folder: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in Image.registered_extensions():
            raise ValueError(f"Unsupported output image extension: {name}")
        paths[name] = path
    return paths


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_weak_images(questions, image_folder, output_folder, *, transform,
                        fll_cutoff_ratio=0.16, fll_order=4.0, blur_radius=3.0,
                        edge_strength=0.58, jpeg_quality=95):
    """Save transformed images under their question-file relative paths."""
    parameters = _parameters(transform, fll_cutoff_ratio, fll_order, blur_radius, edge_strength)
    if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int) or not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be an integer between 1 and 100")
    rows = read_jsonl(questions)
    image_paths = _image_paths(rows, image_folder)
    output_folder = Path(output_folder).resolve()
    if output_folder.exists():
        raise FileExistsError(f"Output folder already exists: {output_folder}")
    manifest = {
        "status": "running", "transform": transform, "parameters": parameters,
        "jpeg_quality": jpeg_quality, "question_count": len(rows),
        "image_count": len(image_paths), "images": list(image_paths),
        "questions_sha256": _sha256(questions),
        "image_folder": str(Path(image_folder).resolve()),
        "output_folder": str(output_folder),
        "image_sha256": {name: _sha256(path) for name, path in image_paths.items()},
        "weak_image_sha256": {},
        "library_versions": {"numpy": np.__version__, "pillow": pillow_version},
    }
    output_folder.mkdir(parents=True, exist_ok=False)
    manifest_path = output_folder / "manifest.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save_manifest()
    try:
        for name, path in image_paths.items():
            target = output_folder / name
            with Image.open(path) as raw:
                image = raw.convert("RGB")
            if transform == "frequency-lowpass":
                transformed = frequency_lowpass(image, **parameters)
            elif transform == "gaussian-blur":
                transformed = weaken_gaussian(image, **parameters)
            else:
                transformed = edge_overlay(image, **parameters)
            target.parent.mkdir(parents=True, exist_ok=True)
            transformed.save(target, quality=jpeg_quality)
            manifest["weak_image_sha256"][name] = _sha256(target)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_manifest()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--transform", choices=TRANSFORMS, required=True)
    parser.add_argument("--fll-cutoff-ratio", type=float, default=0.16)
    parser.add_argument("--fll-order", type=float, default=4.0)
    parser.add_argument("--blur-radius", type=float, default=3.0)
    parser.add_argument("--edge-strength", type=float, default=0.58)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    try:
        manifest = prepare_weak_images(**vars(args))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"transform": manifest["transform"], "images": manifest["image_count"],
                      "manifest": str(Path(args.output_folder) / "manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
