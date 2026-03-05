# Hyperspectral Image (HSI) Mixed Noise Denoising



This repository contains a robust, generalized Python pipeline for simulating and removing complex mixed noise from Hyperspectral Images (HSI). The code applies a two-step algorithmic approach—Robust Principal Component Analysis (RPCA) followed by Principal Component Analysis (PCA) combined with Non-Local Means (NLM) filtering—to restore image quality while preserving spectral fidelity.

## Features

* **Universal File Loading:** Seamlessly loads both `.mat` and `.npy` hyperspectral data cubes.
* **Custom Mixed Noise Generator:** Accurately simulates real-world sensor degradation by adding customizable levels of:
  * Gaussian Noise
  * Salt & Pepper (Impulse) Noise
  * Structural Stripes / Dead Lines
* **Advanced Denoising Pipeline:** * **Step 1:** Fast Randomized RPCA to separate low-rank structural data from sparse corruptions.
  * **Step 2:** PCA-based dimensionality reduction followed by spatial NLM denoising to smooth out residual noise.
* **Quantitative Evaluation:** Automatically computes detailed metrics including PSNR, SSIM, and Global Mean Spectral Angle Distance (MSAD).
* **Visualization:** Built-in matplotlib visualization to compare the Original, Noisy, and Denoised bands side-by-side.

## Requirements

Ensure you have Python 3.7+ installed. You can install the required dependencies using pip:

```bash
pip install numpy scipy matplotlib scikit-image scikit-learn
