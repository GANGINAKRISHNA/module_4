"""
Module 4 – Thermal Animal Boundary Detection
Course: CSc 8830 – Computer Vision

Description:
This script detects the exact boundary of an animal in a thermal image
using classical computer vision techniques only (NO deep learning).

Pipeline:
1. Gaussian Blur (noise reduction)
2. CLAHE (contrast enhancement)
3. Otsu Thresholding
4. Morphological Operations
5. Largest Contour Selection
6. Boundary Overlay + Mask Saving

Execution:
python thermal_segmentation.py --image images/image.jpg
"""

import cv2
import numpy as np
import os
import argparse

# --------------------------
# Argument Parser
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to input thermal image")
args = parser.parse_args()

image_path = args.image
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Load image
# --------------------------
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --------------------------
# Step 1: Noise reduction
# --------------------------
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# --------------------------
# Step 2: Contrast enhancement (CLAHE)
# --------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blur)

# --------------------------
# Step 3: Otsu Thresholding
# --------------------------
_, thresh = cv2.threshold(
    enhanced, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# --------------------------
# Step 4: Morphological Cleaning
# --------------------------
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# --------------------------
# Step 5: Contour Detection
# --------------------------
contours, _ = cv2.findContours(
    closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

output = image.copy()
mask = np.zeros_like(gray)

if len(contours) > 0:
    # Filter small noise contours
    min_area = 1000
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if len(valid_contours) > 0:
        largest_contour = max(valid_contours, key=cv2.contourArea)

        # Draw boundary
        cv2.drawContours(output, [largest_contour], -1, (0, 255, 0), 2)

        # Create binary mask
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    else:
        print("No valid contours found after filtering.")
else:
    print("No contours detected.")

# --------------------------
# Save results
# --------------------------
cv2.imwrite(f"{output_dir}/threshold.jpg", thresh)
cv2.imwrite(f"{output_dir}/cleaned.jpg", closing)
cv2.imwrite(f"{output_dir}/classical_mask.png", mask)
cv2.imwrite(f"{output_dir}/classical_output.jpg", output)

print("Classical segmentation complete.")
print("Results saved in 'results/' directory.")
