from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}


def slide_sort_key(name: str) -> int:
    match = re.search(r"slide(\d+)\.xml$", name)
    return int(match.group(1)) if match else 10**9


def verify_pptx(pptx_path: Path, expected_slides: int = 10) -> None:
    if not pptx_path.exists():
        raise FileNotFoundError(f"PPTX not found: {pptx_path}")

    with ZipFile(pptx_path, "r") as zf:
        names = zf.namelist()
        slide_names = sorted(
            [
                n
                for n in names
                if n.startswith("ppt/slides/slide") and n.endswith(".xml")
            ],
            key=slide_sort_key,
        )
        media_names = [n for n in names if n.startswith("ppt/media/")]

        if len(slide_names) != expected_slides:
            raise AssertionError(
                f"Expected {expected_slides} slides, found {len(slide_names)}"
            )

        print(f"PPTX file: {pptx_path}")
        print(f"Slides: {len(slide_names)} | Embedded media files: {len(media_names)}")

        global_text_runs = 0
        global_images = 0
        font_sizes_pt: list[float] = []

        for slide_name in slide_names:
            xml_data = zf.read(slide_name)
            root = ET.fromstring(xml_data)

            text_runs = [
                t.text.strip()
                for t in root.findall(".//a:t", NS)
                if t.text and t.text.strip()
            ]
            text_count = len(text_runs)
            image_count = len(root.findall(".//a:blip", NS))
            shape_count = len(root.findall(".//p:sp", NS))

            for rpr in root.findall(".//a:rPr", NS):
                sz = rpr.attrib.get("sz")
                if sz is not None:
                    try:
                        font_sizes_pt.append(int(sz) / 100.0)
                    except ValueError:
                        pass

            if text_count == 0 and image_count == 0 and shape_count == 0:
                raise AssertionError(f"Slide appears blank: {slide_name}")

            global_text_runs += text_count
            global_images += image_count
            print(
                f"- {slide_name}: text_runs={text_count}, images={image_count}, shapes={shape_count}"
            )

        print("\nVerification summary")
        print(f"- Total text runs: {global_text_runs}")
        print(f"- Total image references: {global_images}")
        if font_sizes_pt:
            print(
                f"- Explicit font size range: {min(font_sizes_pt):.1f}pt to {max(font_sizes_pt):.1f}pt"
            )
            if min(font_sizes_pt) < 8 or max(font_sizes_pt) > 48:
                raise AssertionError("Detected suspicious font size range in slides")

        if global_text_runs < 40:
            raise AssertionError("Too little text found; deck may be incomplete")

        if global_images < 8:
            raise AssertionError("Too few images found; visual content may be missing")

        print("\nResult: PASS - PPTX is non-blank and structurally populated.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify generated PPTX structure")
    parser.add_argument("pptx", type=Path, help="Path to .pptx file")
    parser.add_argument("--expected-slides", type=int, default=10)
    args = parser.parse_args()

    verify_pptx(args.pptx, expected_slides=args.expected_slides)


if __name__ == "__main__":
    main()
