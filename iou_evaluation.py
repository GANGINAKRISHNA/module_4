"""
IoU Evaluation Between Classical and SAM Segmentation
Course: CSc 8830 – Computer Vision

Execution:
python iou_evaluation.py
"""

import cv2
import numpy as np

classical = cv2.imread("results/classical_mask.png", 0)
sam = cv2.imread("results/sam_mask.png", 0)

if classical is None or sam is None:
    raise FileNotFoundError("Mask files not found in results/ directory.")

classical = classical > 0
sam = sam > 0

intersection = np.logical_and(classical, sam)
union = np.logical_or(classical, sam)

iou = np.sum(intersection) / np.sum(union)

print(f"IoU between Classical Method and SAM: {iou:.4f}")
