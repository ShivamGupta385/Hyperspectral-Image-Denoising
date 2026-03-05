import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage.restoration import denoise_nl_means, estimate_sigma
from sklearn.utils.extmath import randomized_svd

def load_hsi_file(path):
    ext = os.path.splitext(path)[1].lower()
    base = os.path.splitext(path)[0]
    npy_path = base + ".npy"

    if ext == ".npy":
        return np.load(path)
    elif ext == ".mat":
        mat_data = loadmat(path)
        data = None
        for k, v in mat_data.items():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                data = v
                break
        if data is None:
            raise ValueError("No 3D hyperspectral array found in .mat file!")
        np.save(npy_path, data.astype(np.float32))
        return np.load(npy_path)
    else:
        raise ValueError("Unsupported file type.")

def mean_spectral_angle_distance(X, Xhat, eps=1e-12):
    H, W, B = X.shape
    Xv = X.reshape(-1, B)
    Yv = Xhat.reshape(-1, B)
    Xn = Xv / (np.linalg.norm(Xv, axis=1, keepdims=True) + eps)
    Yn = Yv / (np.linalg.norm(Yv, axis=1, keepdims=True) + eps)
    dot = np.sum(Xn * Yn, axis=1).clip(-1, 1)
    angles = np.degrees(np.arccos(dot))
    return np.mean(angles)

def detailed_evaluation(original, denoised):
    H, W, B = original.shape
    psnr_list = []
    ssim_list = []

    for b in range(B):
        p = psnr(original[:,:,b], denoised[:,:,b], data_range=1.0)
        s = ssim(original[:,:,b], denoised[:,:,b], data_range=1.0)
        psnr_list.append(p)
        ssim_list.append(s)

    psnr_arr = np.array(psnr_list)
    ssim_arr = np.array(ssim_list)
    msad = mean_spectral_angle_distance(original, denoised)

    print("\n" + "="*60)
    print(f"{'Metric':<10} | {'Mean':<10} | {'Median':<10} | {'Max':<10} | {'Min':<10}")
    print("-" * 60)
    print(f"{'PSNR (0-1)':<10} | {np.mean(psnr_arr):<10.4f} | {np.median(psnr_arr):<10.4f} | {np.max(psnr_arr):<10.4f} | {np.min(psnr_arr):<10.4f}")
    print(f"{'SSIM':<10} | {np.mean(ssim_arr):<10.4f} | {np.median(ssim_arr):<10.4f} | {np.max(ssim_arr):<10.4f} | {np.min(ssim_arr):<10.4f}")
    print("-" * 60)
    print(f"Global MSAD     : {msad:.4f}")
    print("="*60 + "\n")
    return np.mean(psnr_arr), np.mean(ssim_arr), msad

def add_mixed_noise(image, G, P, stripe_ratio, seed=0):
    rng = np.random.default_rng(seed)
    img = np.clip(image, 0, 1).astype(np.float32)
    H, W, B = img.shape
    noisy = img.copy()

    def add_sp_noise_band(band, p):
        if p <= 0: return band
        mask = rng.choice([0,1,2], size=band.shape, p=[p/2, 1-p, p/2])
        out = band.copy()
        out[mask==0] = 0.0
        out[mask==2] = 1.0
        return out

    for b in range(B):
        sigma = np.sqrt(rng.uniform(0, G))
        band = noisy[:,:,b] + rng.normal(0, sigma, (H,W)).astype(np.float32)
        band = add_sp_noise_band(band, rng.uniform(0, P))
        noisy[:,:,b] = band

    stripe_bands = rng.choice(B, size=max(1, int(stripe_ratio * B)), replace=False)
    for b in stripe_bands:
        for _ in range(rng.integers(2,8)):
            col = rng.integers(0,W)
            width = rng.integers(1,3)
            noisy[:, col:col+width, b] = np.clip(
                noisy[:, col:col+width, b] + rng.uniform(0.08,0.2), 0, 1)
            
    return np.clip(noisy, 0, 1)

def fast_rpca(M, rank_est=40, lam=None, mu=None, max_iter=100, tol=1e-6):
    M = M.astype(np.float64)
    m, n = M.shape
    if lam is None: lam = 1.0 / np.sqrt(max(m,n))

    Y = M / max(np.linalg.norm(M, 2), np.linalg.norm(M.ravel(), np.inf) / lam)
    S = np.zeros_like(M)
    L = np.zeros_like(M)
    if mu is None: mu = 1.25 / np.linalg.norm(M, 2)
    rho = 1.5

    for it in range(max_iter):
        temp = M - S + (1/mu)*Y
        U, s, Vt = randomized_svd(temp, n_components=rank_est, n_iter=5, random_state=42)
        s_thresh = np.maximum(s - 1/mu, 0)
        rank = np.sum(s_thresh > 0)
        L = (U[:, :rank] * s_thresh[:rank]) @ Vt[:rank, :]

        temp = M - L + (1/mu)*Y
        S = np.sign(temp) * np.maximum(np.abs(temp) - lam/mu, 0)

        Z = M - L - S
        err = np.linalg.norm(Z, 'fro') / np.linalg.norm(M, 'fro')
        Y += mu * Z
        mu *= rho

        if err < tol:
            break
    return L

def pca_nlm_denoising(hsi_data, n_components=15):
    H, W, B = hsi_data.shape
    X = hsi_data.reshape(-1, B)

    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec

    U, s, Vt = randomized_svd(X_centered, n_components=n_components, random_state=42)

    eig_images_flat = X_centered @ Vt.T
    eig_images = eig_images_flat.reshape(H, W, n_components)

    denoised_eig = np.zeros_like(eig_images)

    for k in range(n_components):
        img = eig_images[:,:,k]
        sigma_est = np.mean(estimate_sigma(img))

        denoised_eig[:,:,k] = denoise_nl_means(
            img,
            h=0.6 * sigma_est,
            sigma=sigma_est,
            fast_mode=True,
            patch_size=5,
            patch_distance=7
        )

    denoised_flat = denoised_eig.reshape(-1, n_components) @ Vt
    denoised_hsi = denoised_flat + mean_vec

    return denoised_hsi.reshape(H, W, B)

def ultimate_pipeline(noisy, rank=40, n_pca=16):
    H, W, B = noisy.shape
    M = noisy.reshape(-1, B)
    L_flat = fast_rpca(M, rank_est=rank, max_iter=100)
    low_rank_cube = L_flat.reshape(H, W, B)
    final_denoised = pca_nlm_denoising(low_rank_cube, n_components=n_pca)
    return np.clip(final_denoised, 0, 1)

if __name__ == "__main__":
    
    # --- [INPUT REQUIRED] SET HSI FILE PATH ---
    file_path = "Pavia_resized.npy"  
    
    # --- [INPUT REQUIRED] SET NOISE LEVELS ---
    gaussian_variance = 0.1   # Variance limit for Gaussian noise (G)
    salt_pepper_prob = 0.2    # Probability limit for Salt & Pepper noise (P)
    stripe_ratio = 0.2        # Fraction of spectral bands affected by stripes

    data = load_hsi_file(file_path).astype(np.float32)
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)

    noisy = add_mixed_noise(
        data_norm, 
        G=gaussian_variance, 
        P=salt_pepper_prob, 
        stripe_ratio=stripe_ratio, 
        seed=123
    )
    
    denoised = ultimate_pipeline(noisy, rank=75, n_pca=32)

    detailed_evaluation(data_norm, denoised)

    band_idx = 25
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(data_norm[:,:,band_idx], cmap='viridis'); plt.title("Original"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(noisy[:,:,band_idx], cmap='viridis'); plt.title("Noisy"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(denoised[:,:,band_idx], cmap='viridis'); plt.title("Denoised"); plt.axis('off')
    plt.tight_layout()
    plt.show()
