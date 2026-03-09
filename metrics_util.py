import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from cpbd import compute as cpbd_compute

def normalize_img(img):
    """Normalize image to [0, 1] range"""
    return img.astype(np.float32) / 255.0

def uiqm(img):
    """UIQM (Underwater Image Quality Metric)"""
    img = normalize_img(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)
    L = L * 100 / 255
    a = a - 128
    b = b - 128
    alpha_L = alpha_R = 0.1
    a_thresh = np.percentile(a, [alpha_L * 100, (1 - alpha_R) * 100])
    b_thresh = np.percentile(b, [alpha_L * 100, (1 - alpha_R) * 100])
    a_trimmed = a[(a >= a_thresh[0]) & (a <= a_thresh[1])]
    b_trimmed = b[(b >= b_thresh[0]) & (b <= b_thresh[1])]
    if len(a_trimmed) > 0 and len(b_trimmed) > 0:
        uicm = -0.0268 * np.sqrt(np.mean(a_trimmed)**2 + np.mean(b_trimmed)**2) + \
               0.1586 * np.sqrt(np.std(a_trimmed)**2 + np.std(b_trimmed)**2)
    else:
        uicm = 0
    sobel_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    uism = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
    window_size = 8
    h, w = L.shape
    num_windows_h = h // window_size
    num_windows_w = w // window_size
    contrast_values = []
    for i in range(num_windows_h):
        for j in range(num_windows_w):
            window = L[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
            window_min = np.min(window)
            window_max = np.max(window)
            denominator = window_max + window_min
            if denominator > 1e-4:
                contrast = (window_max - window_min) / denominator
                contrast_values.append(contrast)
    uiconm = np.mean(contrast_values) if contrast_values else 0
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uiqm_value = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)
    uiqm_value = (uiqm_value + 3) / 6
    uiqm_value = np.clip(uiqm_value, 0, 1)
    return uiqm_value

def uciqe(img):
    """UCIQE (Underwater Color Image Quality Evaluation)"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:,:,0]
    a = lab[:,:,1]
    b = lab[:,:,2]
    c = np.sqrt((a - 128)**2 + (b - 128)**2)
    sigma_c = np.std(c)
    con_l = np.max(L) - np.min(L)
    sat_l = np.mean(c)
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    uciqe_value = (c1 * sigma_c) + (c2 * con_l) + (c3 * sat_l)
    uciqe_value = uciqe_value / 120.0
    uciqe_value = np.clip(uciqe_value, 0, 1)
    return uciqe_value

def compute_psnr(gt_img, sr_img):
    """PSNR for uint8 images, HWC format"""
    return psnr(gt_img, sr_img, data_range=255)

def compute_ssim(gt_img, sr_img):
    """SSIM for uint8 images, HWC format"""
    return ssim(gt_img, sr_img, channel_axis=2, win_size=5, data_range=255)

def compute_cpbd(img):
    """CPBD expects grayscale or color image, HWC format"""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cpbd_compute(img)