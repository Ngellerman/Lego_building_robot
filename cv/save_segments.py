"""
save_segments.py

Serializes CV pipeline detections to a JSON file and saves per-segment
RGBA crop images alongside it.

Usage as a library:
    from save_segments import save_segments
    save_segments(image, detections, out_dir=Path("artifacts_cv"))
    # writes: artifacts_cv/segments.json
    #         artifacts_cv/crop_sneaker_0.png, ...

The JSON schema:
    {
      "timestamp": "<ISO-8601>",
      "image_size": [width, height],
      "segments": [
        {
          "id": 0,
          "label": "sneaker",
          "confidence": 0.665,
          "class_scores": {"sneaker": 0.80, "flip flop": 0.12, "slipper": 0.08},
          "box": [x1, y1, x2, y2],
          "centroid_uv": [col, row],
          "direction_uv": [dx, dy],
          "mask_coverage": 0.097,
          "crop_image": "crop_sneaker_0.png"
        },
        ...
      ]
    }
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from crop_segments import crop_segments


def _mask_direction_uv(mask: np.ndarray) -> list:
    """Return a unit vector [cos θ, sin θ] of the shoe's long axis in image UV coords."""
    pts = np.column_stack(np.where(mask.astype(np.uint8)))[:, ::-1]  # (N,2) xy
    if len(pts) < 5:
        return [1.0, 0.0]
    _, (w, h), angle_deg = cv2.minAreaRect(pts)
    if w < h:
        angle_deg += 90.0
    angle_rad = np.radians(angle_deg)
    return [round(float(np.cos(angle_rad)), 4), round(float(np.sin(angle_rad)), 4)]


def save_segments(
    image: Image.Image,
    detections,
    out_dir: Path | str = Path("artifacts_cv"),
    json_name: str = "segments.json",
) -> Path:
    """
    Save detection crops and a segments JSON file.

    Returns the path to the written JSON file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crops = crop_segments(image, detections)
    label_counts: dict[str, int] = {}
    segments = []

    for det, (label, crop_img) in zip(detections, crops):
        idx = label_counts.get(label, 0)
        label_counts[label] = idx + 1

        crop_filename = f"crop_{label.replace(' ', '_')}_{idx}.png"
        crop_img.save(out_dir / crop_filename)
        np.save(out_dir / f"mask_{len(segments)}.npy", det.mask)

        rows, cols = np.where(det.mask)
        centroid = (
            [round(float(cols.mean()), 1), round(float(rows.mean()), 1)]
            if len(rows) > 0
            else [0.0, 0.0]
        )

        segments.append({
            "id": len(segments),
            "label": label,
            "confidence": round(det.confidence, 4),
            "class_scores": {k: round(v, 4) for k, v in det.class_scores.items()},
            "box": [round(v, 1) for v in det.box],
            "centroid_uv": centroid,
            "direction_uv": _mask_direction_uv(det.mask),
            "mask_coverage": round(float(det.mask.mean()), 4),
            "crop_image": crop_filename,
        })

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_size": [image.width, image.height],
        "segments": segments,
    }

    json_path = out_dir / json_name
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return json_path
