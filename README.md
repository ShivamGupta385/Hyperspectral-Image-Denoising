# Hyperspectral Image (HSI) Mixed Noise Denoising



This repository contains a robust, generalized Python pipeline for simulating and removing complex mixed noise from Hyperspectral Images (HSI). The code applies a two-step algorithmic approach—Robust Principal Component Analysis (RPCA) followed by Principal Component Analysis (PCA) combined with Non-Local Means (NLM) filtering—to restore image quality while preserving spectral fidelity.

## Features

* **Universal File Loading:** Seamlessly loads both `.mat` and `.npy` hyperspectral data cubes.
* **Custom Mixed Noise Generator:** Accurately simulates real-world sensor degradation by adding customizable levels of:
  * Gaussian Noise
  * Salt & Pepper (Impulse) Noise
  * Structural Stripes / Dead Lines
* **Advanced Denoising Pipeline:**
  * **Step 1:** Fast Randomized RPCA to separate low-rank structural data from sparse corruptions.
  * **Step 2:** PCA-based dimensionality reduction followed by spatial NLM denoising to smooth out residual noise.
* **Quantitative Evaluation:** Automatically computes detailed metrics including PSNR, SSIM, and Global Mean Spectral Angle Distance (MSAD).
* **Visualization:** Built-in matplotlib visualization to compare the Original, Noisy, and Denoised bands side-by-side.

## Requirements

Ensure you have Python 3.7+ installed. You can install the required dependencies using pip:

```
pip install numpy scipy matplotlib scikit-image scikit-learn
```

Usage:
* Clone this repository to your local machine.

* Place your hyperspectral image file (.mat or .npy) in the project directory.

* Open the script and scroll to the if __name__ == "__main__": block at the bottom.

* Update the input variables with your specific file path and desired noise parameters:
```
Python
# --- [INPUT REQUIRED] SET HSI FILE PATH ---
file_path = "your_image_file.npy"  

# --- [INPUT REQUIRED] SET NOISE LEVELS ---
gaussian_variance = 0.1   # Variance limit for Gaussian noise (G)
salt_pepper_prob = 0.2    # Probability limit for Salt & Pepper noise (P)
stripe_ratio = 0.2        # Fraction of spectral bands affected by stripes
```

Run the script:
```
python denoising_pipeline.py
```
Pipeline Overview:
* Normalization: The raw HSI cube is min-max normalized to a [0, 1] scale.

* Degradation: The add_mixed_noise function applies the defined noise tolerances across the spectral bands.

* RPCA: The noisy image is flattened and processed through a Fast RPCA algorithm to recover the low-rank component.

* PCA + NLM: The low-rank cube is projected into a lower-dimensional PCA space, where Non-Local Means filtering is applied to estimate and remove remaining noise before reconstructing the full spectral cube.

* Evaluation: The script outputs a detailed table of Mean, Median, Max, and Min values for PSNR and SSIM, alongside the Global MSAD score.
