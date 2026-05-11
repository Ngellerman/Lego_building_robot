from __future__ import annotations

import PIL.Image
if not hasattr(PIL.Image, 'Resampling'):
    class _Resampling:
        NEAREST  = PIL.Image.NEAREST
        BILINEAR = PIL.Image.BILINEAR
        BICUBIC  = PIL.Image.BICUBIC
        LANCZOS  = PIL.Image.ANTIALIAS
        BOX      = PIL.Image.BOX
        HAMMING  = PIL.Image.HAMMING
    PIL.Image.Resampling = _Resampling

import numpy as np
import torch
from dataclasses import dataclass
from PIL import Image
from transformers import (
    AutoProcessor,
    GroundingDinoForObjectDetection,
    SamModel,
    SamProcessor,
)

DEFAULT_CLASSES = ["sneaker", "flip flop", "slipper"]

GDINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam-vit-base"


@dataclass
class Detection:
    label: str
    confidence: float
    class_scores: dict[str, float]  # softmax distribution over CLASSES
    box: list[float]                # [x1, y1, x2, y2] in pixels
    mask: np.ndarray                # bool (H, W)


class CVPipeline:
    def __init__(self, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._load_models()

    def _load_models(self):
        print(f"Loading GroundingDINO on {self.device}...")
        self.gdino_proc = AutoProcessor.from_pretrained(GDINO_ID)
        self.gdino = (
            GroundingDinoForObjectDetection.from_pretrained(GDINO_ID)
            .to(self.device)
            .eval()
        )

        print(f"Loading SAM on {self.device}...")
        self.sam_proc = SamProcessor.from_pretrained(SAM_ID)
        self.sam = SamModel.from_pretrained(SAM_ID).to(self.device).eval()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        image: Image.Image,
        classes: list[str] | None = None,
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        min_confidence: float = 0.0,
        min_mask_coverage: float = 0.05,
        max_mask_coverage: float = 0.2,
    ) -> list[Detection]:
        """
        Detect and segment objects in the image.
        Returns one Detection per object found, sorted by confidence descending.
        """
        if classes is None:
            classes = DEFAULT_CLASSES
        text_prompt = " . ".join(classes) + " ."

        image = image.convert("RGB")
        boxes, scores, labels, query_logits = self._detect(
            image, box_threshold, text_threshold, text_prompt
        )

        if len(boxes) == 0:
            return []

        masks = self._segment(image, boxes)

        image_pixels = image.width * image.height
        detections = []
        for box, score, label, logits, mask in zip(
            boxes, scores, labels, query_logits, masks
        ):
            if float(score) < min_confidence:
                continue
            coverage = mask.sum() / image_pixels
            if coverage < min_mask_coverage or coverage > max_mask_coverage:
                continue
            class_scores = self._class_scores_from_logits(logits, classes, text_prompt)
            detections.append(
                Detection(
                    label=max(class_scores, key=class_scores.get),
                    confidence=float(score),
                    class_scores=class_scores,
                    box=box.tolist(),
                    mask=mask,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ── Detection ──────────────────────────────────────────────────────────────

    def _detect(self, image, box_threshold, text_threshold, text_prompt):
        inputs = self.gdino_proc(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.gdino(**inputs)

        h, w = image.height, image.width
        results = self.gdino_proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)],
        )[0]

        boxes = results["boxes"].cpu().numpy()   # (N, 4) x1y1x2y2 pixels
        scores = results["scores"].cpu().numpy() # (N,)
        labels = results["labels"]               # list[str]

        query_logits = self._match_query_logits(outputs, boxes, (h, w))
        return boxes, scores, labels, query_logits

    def _match_query_logits(self, outputs, boxes_px, image_size):
        """
        Map each post-processed (NMS'd) box back to its query in outputs.logits
        by nearest-neighbour matching in normalised cxcywh space.
        """
        if len(boxes_px) == 0:
            return []

        h, w = image_size
        pred_boxes = outputs.pred_boxes[0].cpu()  # (num_queries, 4) normalised cxcywh
        raw_logits = outputs.logits[0].cpu()       # (num_queries, text_len)

        boxes_t = torch.tensor(boxes_px, dtype=torch.float32)
        cx = (boxes_t[:, 0] + boxes_t[:, 2]) / 2 / w
        cy = (boxes_t[:, 1] + boxes_t[:, 3]) / 2 / h
        bw = (boxes_t[:, 2] - boxes_t[:, 0]) / w
        bh = (boxes_t[:, 3] - boxes_t[:, 1]) / h
        boxes_norm = torch.stack([cx, cy, bw, bh], dim=1)  # (N, 4)

        logits_per_box = []
        for box_norm in boxes_norm:
            dists = ((pred_boxes - box_norm) ** 2).sum(dim=1)
            best_query = dists.argmin().item()
            logits_per_box.append(raw_logits[best_query])
        return logits_per_box

    def _class_scores_from_logits(self, logits: torch.Tensor, classes: list[str], text_prompt: str) -> dict[str, float]:
        tokenizer = self.gdino_proc.tokenizer
        full_ids = tokenizer(text_prompt, add_special_tokens=True)["input_ids"]

        raw_scores = []
        for cls in classes:
            cls_ids = tokenizer(cls, add_special_tokens=False)["input_ids"]
            n = len(cls_ids)
            span_start = None
            for i in range(len(full_ids) - n + 1):
                if full_ids[i : i + n] == cls_ids:
                    span_start = i
                    break
            score = (
                logits[span_start : span_start + n].max().item()
                if span_start is not None
                else -10.0
            )
            raw_scores.append(score)

        probs = torch.softmax(torch.tensor(raw_scores, dtype=torch.float32), dim=0)
        return {cls: float(probs[i]) for i, cls in enumerate(classes)}

    def _canonical_label(self, label: str) -> str:
        label = label.lower().strip()
        if label in CLASSES:
            return label
        # Prefer more specific (longer) match to avoid "box" swallowing "cereal box"
        matches = [cls for cls in CLASSES if cls in label or label in cls]
        return max(matches, key=len) if matches else label

    # ── Segmentation ───────────────────────────────────────────────────────────

    def _segment(self, image: Image.Image, boxes: np.ndarray) -> list[np.ndarray]:
        masks = []
        for box in boxes:
            inputs = self.sam_proc(
                image,
                input_boxes=[[box.tolist()]],
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.sam(**inputs)

            processed = self.sam_proc.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
            # SAM returns 3 mask candidates per box; pick the one with highest IoU score
            iou = outputs.iou_scores.cpu().numpy()[0, 0]  # (3,)
            best = iou.argmax()
            masks.append(processed[0][0][best].numpy().astype(bool))
        return masks