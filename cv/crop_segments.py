"""
crop_segments.py

Given a list of Detection objects from the CV pipeline, returns a cropped
PIL Image for each detection containing only the masked pixels (everything
outside the mask is transparent). The crop is tight to the mask bounding box.

Usage as a library:
    from crop_segments import crop_segments
    crops = crop_segments(image, detections)  # list of (label, PIL.Image RGBA)

Usage as a script:
    python3 cv/crop_segments.py artifacts/capture.jpg
    # saves artifacts/segment_mug_0.png, artifacts/segment_shoe_1.png, ...
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def crop_segments(image: Image.Image, detections) -> list[tuple[str, Image.Image]]:
    """
    For each detection, return a tight RGBA crop containing only masked pixels.

    Returns a list of (label, image) pairs in the same order as detections.
    """
    rgb = np.array(image.convert('RGB'))
    results = []

    for d in detections:
        mask = d.mask  # bool (H, W)

        rows, cols = np.where(mask)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1

        crop_rgb   = rgb[r0:r1, c0:c1]
        crop_mask  = mask[r0:r1, c0:c1]

        rgba = np.zeros((r1 - r0, c1 - c0, 4), dtype=np.uint8)
        rgba[..., :3] = crop_rgb
        rgba[..., 3]  = (crop_mask * 255).astype(np.uint8)

        results.append((d.label, Image.fromarray(rgba, mode='RGBA')))

    return results


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 cv/crop_segments.py <image_path>')
        sys.exit(1)

    image_path = Path(sys.argv[1])
    out_dir = image_path.parent

    # Re-run the pipeline on the provided image
    sys.path.insert(0, str(Path(__file__).parent))
    from pipeline import CVPipeline

    image = Image.open(image_path).convert('RGB')
    pipeline = CVPipeline()
    detections = pipeline.run(image)

    if not detections:
        print('No detections.')
        return

    crops = crop_segments(image, detections)
    label_counts: dict[str, int] = {}
    for label, crop in crops:
        idx = label_counts.get(label, 0)
        label_counts[label] = idx + 1
        out_path = out_dir / f'segment_{label.replace(" ", "_")}_{idx}.png'
        crop.save(out_path)
        print(f'Saved {crop.width}x{crop.height} crop → {out_path}')


if __name__ == '__main__':
    main()
