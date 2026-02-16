"""
CSC 8830 - Computer Vision
Module 3 Assignment

Image Blurring using:
1. Spatial-domain convolution
2. Fourier-domain multiplication

This script demonstrates that convolution in the spatial domain
is equivalent to multiplication in the frequency domain.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import os


# -------------------------------------------------------------
# Create results folder if it does not exist
# -------------------------------------------------------------
os.makedirs("results", exist_ok=True)


# -------------------------------------------------------------
# Load and preprocess the image
# -------------------------------------------------------------
# Make sure the image is inside: images/input.jpg
image = cv2.imread("images/input.webp", cv2.IMREAD_GRAYSCALE)


if image is None:
    raise FileNotFoundError("Place an image at images/input.jpg")

# Normalize to [0, 1]
image = image.astype(np.float32) / 255.0
H, W = image.shape


# -------------------------------------------------------------
# Create Gaussian blur kernel
# -------------------------------------------------------------
def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


kernel_size = 15
sigma = 3
kernel = gaussian_kernel(kernel_size, sigma)


# -------------------------------------------------------------
# Spatial-domain convolution
# -------------------------------------------------------------
spatial_blur = convolve2d(
    image,
    kernel,
    mode="same",
    boundary="symm"
)

cv2.imwrite("results/spatial_blur.png", (spatial_blur * 255).astype(np.uint8))


# -------------------------------------------------------------
# Prepare kernel for Fourier-domain filtering
# -------------------------------------------------------------
kernel_padded = np.zeros((H, W))
kh, kw = kernel.shape
kernel_padded[:kh, :kw] = kernel

# Shift kernel to correct FFT position
kernel_padded = np.fft.ifftshift(kernel_padded)


# -------------------------------------------------------------
# Fourier-domain filtering
# -------------------------------------------------------------
F = np.fft.fft2(image)
H_f = np.fft.fft2(kernel_padded)

G = F * H_f
fourier_blur = np.real(np.fft.ifft2(G))

cv2.imwrite("results/fourier_blur.png", (fourier_blur * 255).astype(np.uint8))


# -------------------------------------------------------------
# Compare results
# -------------------------------------------------------------
difference = np.abs(spatial_blur - fourier_blur)
mse = np.mean(difference ** 2)

print("Mean Squared Error (Spatial vs Fourier):", mse)

cv2.imwrite("results/difference.png", (difference * 255).astype(np.uint8))


# -------------------------------------------------------------
# Display results
# -------------------------------------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Spatial Domain Blur")
plt.imshow(spatial_blur, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Fourier Domain Blur")
plt.imshow(fourier_blur, cmap="gray")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Absolute Difference")
plt.imshow(difference, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
