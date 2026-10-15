"""Create a clean ET-SCD source ZIP using an explicit allowlist."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (".gitignore", "README.md", "ENVIRONMENT.md", "LICENSE", "CITATION.cff", "pyproject.toml",
         "requirements.txt", "requirements-eval.txt", "docs/IMPLEMENTATION.md")
TREES = ("src", "configs", "data", "scripts")


def main():
    paths = [ROOT / name for name in FILES]
    paths.extend((ROOT / "tests").glob("test_*.py"))
    for tree in TREES:
        paths.extend(path for path in (ROOT / tree).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts
                     and not any(part.endswith((".egg-info", ".dist-info")) for part in path.parts)
                     and path.suffix not in {".pyc", ".pyo"})
    paths = sorted(set(paths))
    for path in paths:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Invalid release file {path}")
    required = ROOT / "src/etscd/chair_eval.py"
    if required not in paths:
        raise ValueError("Missing required CHAIR evaluation entry point")
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    release = ROOT / "dist/ET-SCD-v0.1.0-source.zip"
    release.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(release, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            handle.write(path, "ET-SCD/" + path.relative_to(ROOT).as_posix())
        handle.writestr("ET-SCD/RELEASE_MANIFEST.json", json.dumps({"version": "0.1.0", "files_sha256": hashes}, indent=2))
    with zipfile.ZipFile(release) as handle:
        if handle.testzip() is not None:
            raise ValueError("Release archive failed CRC check")
        for name, digest in hashes.items():
            if hashlib.sha256(handle.read("ET-SCD/" + name)).hexdigest() != digest:
                raise ValueError(f"Release hash mismatch: {name}")
    print(f"{release}\n{len(paths)} source files; all hashes and ZIP CRC verified")


if __name__ == "__main__":
    main()
