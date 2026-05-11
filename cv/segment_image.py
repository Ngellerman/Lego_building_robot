import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline import CVPipeline, DEFAULT_CLASSES
from save_segments import save_segments

# One distinct colour per class
PALETTE = {
    "sneaker":   (255,  87,  87),
    "flip flop": ( 87, 187, 255),
    "slipper":   ( 87, 255, 161),
}
MASK_ALPHA = 0.45


def visualise(image: Image.Image, detections) -> Image.Image:
    out = image.copy().convert("RGBA")

    for d in detections:
        colour = PALETTE.get(d.label, (200, 200, 200))

        # --- mask overlay ---
        rgba_mask = np.zeros((*d.mask.shape, 4), dtype=np.uint8)
        rgba_mask[d.mask] = (*colour, int(255 * MASK_ALPHA))
        mask_layer = Image.fromarray(rgba_mask, mode="RGBA")
        out = Image.alpha_composite(out, mask_layer)

    # Draw boxes and labels on top
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=18)
    except OSError:
        font = ImageFont.load_default()

    for d in detections:
        colour = PALETTE.get(d.label, (200, 200, 200))
        x1, y1, x2, y2 = (round(v) for v in d.box)

        # Box
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)

        # Label background + text
        label_text = f"{d.label} {d.confidence:.2f}"
        bbox = draw.textbbox((x1, y1), label_text, font=font)
        draw.rectangle(bbox, fill=(*colour, 220))
        draw.text((x1, y1), label_text, fill=(0, 0, 0), font=font)

    return out.convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Space-separated class names (quote multi-word names)")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_stem(input_path.stem + "_detections")
    classes = args.classes or DEFAULT_CLASSES

    image = Image.open(input_path).convert("RGB")
    pipeline = CVPipeline()
    detections = pipeline.run(image, classes=classes)

    if not detections:
        print("No objects detected.")
        return

    print(f"\nDetected {len(detections)} object(s):\n")
    for i, d in enumerate(detections):
        scores_str = "  ".join(
            f"{k}={v:.3f}"
            for k, v in sorted(d.class_scores.items(), key=lambda x: -x[1])
        )
        print(f"[{i + 1}] {d.label}  conf={d.confidence:.3f}")
        print(f"     box : {[round(v, 1) for v in d.box]}")
        print(f"     mask: {d.mask.sum()} px  ({d.mask.mean() * 100:.1f}% of image)")
        print(f"     class scores: {scores_str}")
        print()

    annotated = visualise(image, detections)
    annotated.save(output_path)
    print(f"Saved annotated image to: {output_path}")

    json_path = save_segments(image, detections, out_dir=output_path.parent)
    print(f"Saved segments JSON to:   {json_path}")


if __name__ == "__main__":
    main()