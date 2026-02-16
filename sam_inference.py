"""
SAM Segmentation for Comparison
Course: CSc 8830 – Computer Vision

Description:
This script runs Meta's Segment Anything Model (SAM)
to generate segmentation masks for comparison
with the classical OpenCV-based method.

Execution:
python sam_inference.py --image images/image.jpg
"""

import os
import cv2
import numpy as np
import argparse
from segment_anything import sam_model_registry, SamPredictor

# --------------------------
# Argument Parser
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to input image")
args = parser.parse_args()

IMAGE_PATH = args.image
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_TYPE = "vit_h"
MODEL_CHECKPOINT = "sam_vit_h_4b8939.pth"  # Update if needed

# --------------------------
# Load Image
# --------------------------
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --------------------------
# Load SAM Model
# --------------------------
sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_CHECKPOINT)
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# --------------------------
# Run Inference
# --------------------------
masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=None,
    multimask_output=True
)

best_idx = np.argmax(scores)
mask = masks[best_idx]

# --------------------------
# Save Mask
# --------------------------
sam_mask = mask.astype(np.uint8) * 255
mask_path = os.path.join(RESULTS_DIR, "sam_mask.png")
cv2.imwrite(mask_path, sam_mask)

# --------------------------
# Overlay
# --------------------------
overlay = image.copy()
overlay[mask] = (0, 255, 0)
overlay_image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

overlay_path = os.path.join(RESULTS_DIR, "sam_overlay.png")
cv2.imwrite(overlay_path, overlay_image)

print("SAM segmentation complete.")
print("Results saved in 'results/' directory.")
