from __future__ import annotations

# Run: uv run python .\scripts\build_python_repomix.py

import os
import re
import shutil
import subprocess
from pathlib import Path


OUTPUT_NAME = "python_repomix.xml"
EXCLUDED_DIR_NAMES = {
    ".venv",
    "venv",
    "outputs",
    "targets",
    "scripts",
    "tests",
}
EXCLUDED_FILE_NAMES = {
    ".python-version",
    "uv.lock",
    OUTPUT_NAME,
}


def _project_python_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_npx() -> str:
    candidates = [
        shutil.which("npx.cmd"),
        shutil.which("npx"),
        str(Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs" / "npx.cmd"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError("Could not locate npx.cmd. Install Node.js and ensure npx.cmd is available.")


def _is_excluded_dir(path: Path) -> bool:
    name = path.name.lower()
    return name in EXCLUDED_DIR_NAMES or "cache" in name


def _is_excluded_file(path: Path) -> bool:
    name = path.name
    if name in EXCLUDED_FILE_NAMES:
        return True
    lowered = name.lower()
    if lowered.startswith("repomix") and lowered.endswith(".xml"):
        return True
    if lowered.endswith("_repomix.xml"):
        return True
    return False


def _collect_manifest_files(python_root: Path) -> list[str]:
    files: list[str] = []
    for current_root, dir_names, file_names in os.walk(python_root):
        current_path = Path(current_root)
        dir_names[:] = sorted(
            [name for name in dir_names if not _is_excluded_dir(current_path / name)]
        )
        for file_name in sorted(file_names):
            path = current_path / file_name
            if _is_excluded_file(path):
                continue
            files.append(path.relative_to(python_root).as_posix())
    return files


def _ignore_patterns() -> str:
    patterns = [
        ".venv/**",
        "venv/**",
        "outputs/**",
        "targets/**",
        "scripts/**",
        "tests/**",
        "**/__pycache__/**",
        "**/.pytest_cache/**",
        "**/.uv-cache/**",
        "**/*cache*/**",
        ".python-version",
        "uv.lock",
        OUTPUT_NAME,
        "repomix*.xml",
        "*_repomix.xml",
    ]
    return ",".join(patterns)


def _run_repomix(*, python_root: Path, manifest_files: list[str], output_path: Path) -> None:
    if not manifest_files:
        raise RuntimeError("No files matched the repomix manifest.")

    npx_cmd = _resolve_npx()
    subprocess.run(
        [
            npx_cmd,
            "repomix",
            "--style",
            "xml",
            "--output",
            str(output_path.name),
            "--ignore",
            _ignore_patterns(),
            "--no-gitignore",
            "--no-dot-ignore",
            "--no-default-patterns",
            "--quiet",
        ],
        cwd=python_root,
        check=True,
    )


def _extract_xml_paths(output_path: Path) -> list[str]:
    text = output_path.read_text(encoding="utf-8")
    paths = re.findall(r'<file path="([^"]+)">', text)
    return sorted({path.replace("\\", "/").lstrip("./") for path in paths})


def _verify_output(*, manifest_files: list[str], xml_paths: list[str]) -> None:
    manifest_set = set(manifest_files)
    xml_set = set(xml_paths)
    missing = sorted(manifest_set - xml_set)
    unexpected = sorted(xml_set - manifest_set)
    if missing or unexpected:
        lines = ["Repomix verification failed."]
        if missing:
            lines.append("Missing files:")
            lines.extend(f"  - {path}" for path in missing)
        if unexpected:
            lines.append("Unexpected files:")
            lines.extend(f"  - {path}" for path in unexpected)
        raise RuntimeError("\n".join(lines))


def main() -> int:
    python_root = _project_python_root()
    output_path = python_root / OUTPUT_NAME
    manifest_files = _collect_manifest_files(python_root)
    _run_repomix(python_root=python_root, manifest_files=manifest_files, output_path=output_path)
    xml_paths = _extract_xml_paths(output_path)
    _verify_output(manifest_files=manifest_files, xml_paths=xml_paths)

    print(f"Created repomix: {output_path}")
    print(f"Included file count: {len(manifest_files)}")
    for path in manifest_files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
