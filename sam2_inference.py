import os
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# ---------- PATHS ----------
IMAGE_PATH = "images/image.jpg"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_TYPE = "vit_h"  # options: vit_h, vit_l, vit_b
MODEL_CHECKPOINT = "sam_vit_h_4b8939.pth"  # update if needed

# ---------- LOAD IMAGE ----------
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------- LOAD SAM MODEL ----------
sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_CHECKPOINT)
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# ---------- RUN SAM INFERENCE ----------
# No points/box, output multiple masks
masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=None,
    multimask_output=True
)

# Choose best mask based on score
best_idx = np.argmax(scores)
mask = masks[best_idx]

# ---------- SAVE MASK ----------
mask_path = os.path.join(RESULTS_DIR, "sam2_mask.png")
cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)

# ---------- CREATE OVERLAY ----------
overlay = image.copy()
overlay[mask] = (0, 255, 0)  # green mask overlay
alpha = 0.5
overlay_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

overlay_path = os.path.join(RESULTS_DIR, "sam2_overlay.png")
cv2.imwrite(overlay_path, overlay_image)

print(f"SAM2 inference complete!\nMask saved: {mask_path}\nOverlay saved: {overlay_path}")
