"""
Thermal animal boundary detection using classical CV.
Run:
python thermal_segmentation.py
"""

import cv2
import numpy as np
import os

# --------------------------
# Paths
# --------------------------
image_path = "images/image.jpg"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Load image
# --------------------------
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --------------------------
# Step 1: Noise reduction
# --------------------------
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# --------------------------
# Step 2: Contrast enhancement
# --------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blur)

# --------------------------
# Step 3: Thresholding
# --------------------------
_, thresh = cv2.threshold(
    enhanced, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# --------------------------
# Step 4: Morphology
# --------------------------
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# --------------------------
# Step 5: Contours
# --------------------------
contours, _ = cv2.findContours(
    closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# --------------------------
# Step 6: Draw boundaries
# --------------------------
output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

# --------------------------
# Save results
# --------------------------
cv2.imwrite(f"{output_dir}/threshold.jpg", thresh)
cv2.imwrite(f"{output_dir}/closing.jpg", closing)
cv2.imwrite(f"{output_dir}/classical_output.jpg", output)

print("Classical segmentation complete. Results saved in results/")
