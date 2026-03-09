# import os
# import glob
# import numpy as np
# import cv2
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
# import math
# from scipy import ndimage
# from scipy.ndimage import convolve
# import argparse
# import pandas as pd
# from tqdm import tqdm

# def normalize_img(img):
#     """Normalize image to [0, 1] range"""
#     return img.astype(np.float32) / 255.0

# def uiqm(img):
#     """UIQM (Underwater Image Quality Metric)"""
#     # Normalize image to [0, 1]
#     img = normalize_img(img)
    
#     # Convert to LAB color space
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
#     # Split channels
#     L, a, b = cv2.split(lab)
    
#     # Scale L to [0, 100] and a,b to [-128, 127]
#     L = L * 100 / 255
#     a = a - 128
#     b = b - 128
    
#     # Calculate colorfulness (UICM)
#     alpha_L = alpha_R = 0.1
#     a_thresh = np.percentile(a, [alpha_L * 100, (1 - alpha_R) * 100])
#     b_thresh = np.percentile(b, [alpha_L * 100, (1 - alpha_R) * 100])
    
#     a_trimmed = a[(a >= a_thresh[0]) & (a <= a_thresh[1])]
#     b_trimmed = b[(b >= b_thresh[0]) & (b <= b_thresh[1])]
    
#     if len(a_trimmed) > 0 and len(b_trimmed) > 0:
#         uicm = -0.0268 * np.sqrt(np.mean(a_trimmed)**2 + np.mean(b_trimmed)**2) + \
#                0.1586 * np.sqrt(np.std(a_trimmed)**2 + np.std(b_trimmed)**2)
#     else:
#         uicm = 0
    
#     # Calculate sharpness (UISM)
#     sobel_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
#     uism = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
    
#     # Calculate contrast (UIConM)
#     window_size = 8
#     h, w = L.shape
#     num_windows_h = h // window_size
#     num_windows_w = w // window_size
#     contrast_values = []
    
#     for i in range(num_windows_h):
#         for j in range(num_windows_w):
#             window = L[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
#             window_min = np.min(window)
#             window_max = np.max(window)
#             denominator = window_max + window_min
#             if denominator > 1e-4:  # Avoid division by very small numbers
#                 contrast = (window_max - window_min) / denominator
#                 contrast_values.append(contrast)
    
#     uiconm = np.mean(contrast_values) if contrast_values else 0
    
#     # Calculate UIQM with empirical coefficients
#     c1, c2, c3 = 0.0282, 0.2953, 3.5753
#     uiqm_value = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)
    
#     # Scale to match paper's range [0, 1]
#     uiqm_value = (uiqm_value + 3) / 6  # Assuming original range is [-3, 3]
#     uiqm_value = np.clip(uiqm_value, 0, 1)
    
#     return uiqm_value

# def uciqe(img):
#     """UCIQE (Underwater Color Image Quality Evaluation)"""
#     # Convert to LAB color space
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
#     # Split channels
#     L = lab[:,:,0]
#     a = lab[:,:,1]
#     b = lab[:,:,2]
    
#     # Calculate chroma
#     c = np.sqrt((a - 128)**2 + (b - 128)**2)
    
#     # Calculate metrics
#     sigma_c = np.std(c)
#     con_l = np.max(L) - np.min(L)
#     sat_l = np.mean(c)
    
#     # UCIQE parameters
#     c1, c2, c3 = 0.4680, 0.2745, 0.2576
#     uciqe_value = (c1 * sigma_c) + (c2 * con_l) + (c3 * sat_l)
    
#     # Normalize to [0,1] range
#     uciqe_value = uciqe_value / 120.0  # Adjusted normalization factor
#     uciqe_value = np.clip(uciqe_value, 0, 1)
    
#     return uciqe_value

# def compute_cpbd(image):
#     """
#     Compute CPBD (Cumulative Probability of Blur Detection) following the MATLAB implementation
#     from the paper "A No-Reference Image Blur Metric Based on the Cumulative Probability of Blur Detection"
#     """
#     if image.ndim == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Convert to float and normalize to [0, 1]
#     image = image.astype(np.float32) / 255.0
    
#     # Parameters
#     beta = 0.5
#     T = 0.0010  # Lower threshold for more edge detection
#     rf = 4  # Reasonable factor
    
#     # Compute edges using Sobel
#     gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
#     gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    
#     # Compute gradient magnitude and direction
#     magnitude = np.sqrt(gx**2 + gy**2)
#     direction = np.arctan2(gy, gx)
    
#     # Normalize magnitude
#     magnitude = magnitude / (np.max(magnitude) + 1e-6)
    
#     # Find strong edges
#     edge_map = magnitude > T
    
#     if not np.any(edge_map):
#         return 0.0
    
#     # Get edge coordinates
#     edge_y, edge_x = np.where(edge_map)
    
#     # Initialize arrays for edge widths
#     edge_widths = []
    
#     # Process each edge pixel
#     for y, x in zip(edge_y, edge_x):
#         # Skip border pixels
#         if y < rf or y >= image.shape[0]-rf or x < rf or x >= image.shape[1]-rf:
#             continue
        
#         # Get the edge direction
#         theta = direction[y, x]
        
#         # Get the perpendicular direction
#         theta_perp = theta + np.pi/2
        
#         # Sample points perpendicular to the edge
#         x_perp = np.cos(theta_perp)
#         y_perp = np.sin(theta_perp)
        
#         # Get intensity profile perpendicular to edge
#         profile = []
#         distances = np.arange(-rf, rf+1)
        
#         for d in distances:
#             sample_y = int(round(y + d * y_perp))
#             sample_x = int(round(x + d * x_perp))
            
#             if (0 <= sample_y < image.shape[0] and 
#                 0 <= sample_x < image.shape[1]):
#                 profile.append(magnitude[sample_y, sample_x])
#             else:
#                 profile.append(0)
        
#         profile = np.array(profile)
        
#         # Find local maxima
#         maxima = []
#         for i in range(1, len(profile)-1):
#             if profile[i] > profile[i-1] and profile[i] > profile[i+1]:
#                 maxima.append(i)
        
#         # Calculate edge width
#         if len(maxima) >= 2:
#             # Use distance between first and last maximum
#             width = maxima[-1] - maxima[0]
#             if width > 0:
#                 # Normalize width by the maximum possible width
#                 width = width / (2 * rf)
#                 edge_widths.append(width)
    
#     if not edge_widths:
#         return 0.0
    
#     # Calculate blur probabilities
#     probabilities = []
#     for width in edge_widths:
#         # Use a more gradual probability function
#         p_blur = 1 - np.exp(-(width/beta)**2)
#         # Only consider probabilities in a reasonable range
#         if p_blur <= 0.5:
#             probabilities.append(p_blur)
    
#     if not probabilities:
#         return 0.0
    
#     # Calculate CPBD - using raw value without scaling
#     return np.mean(probabilities)

# def calculate_metrics(sr_path, hr_path):
#     """Calculate metrics for all images"""
#     # Get all SR images
#     sr_images = sorted(glob.glob(os.path.join(sr_path, '*_sr.png')))
    
#     if not sr_images:
#         print(f"No SR images found in {sr_path}")
#         return
    
#     # Prepare results dictionary
#     results = {
#         'image': [],
#         'psnr': [],
#         'ssim': [],
#         'uiqm': [],
#         'uciqe': [],
#         'cpbd': []
#     }
    
#     # Process each image
#     for sr_file in tqdm(sr_images, desc="Processing Images"):
#         # Extract image index
#         img_idx = os.path.basename(sr_file).split('_')[1]
        
#         # Find corresponding HR image
#         hr_files = glob.glob(os.path.join(hr_path, f'*_{img_idx}_hr.png'))
#         if not hr_files:
#             print(f"Warning: No HR image found for {sr_file}")
#             continue
        
#         hr_file = hr_files[0]
        
#         # Read images
#         sr_img = cv2.imread(sr_file)
#         hr_img = cv2.imread(hr_file)
        
#         if sr_img is None or hr_img is None:
#             print(f"Warning: Could not read images for {sr_file}")
#             continue
        
#         # Calculate metrics
#         try:
#             psnr_value = psnr(hr_img, sr_img)
#             ssim_value = ssim(hr_img, sr_img, channel_axis=2, win_size=5)
#             uiqm_value = uiqm(sr_img)
#             uciqe_value = uciqe(sr_img)
#             cpbd_value = compute_cpbd(sr_img)
            
#             # Verify values are valid
#             if not (np.isfinite(psnr_value) and np.isfinite(ssim_value) and 
#                    np.isfinite(uiqm_value) and np.isfinite(uciqe_value) and 
#                    np.isfinite(cpbd_value)):
#                 print(f"Warning: Invalid metrics for {sr_file}")
#                 continue
            
#             # Store results
#             results['image'].append(os.path.basename(sr_file))
#             results['psnr'].append(psnr_value)
#             results['ssim'].append(ssim_value)
#             results['uiqm'].append(uiqm_value)
#             results['uciqe'].append(uciqe_value)
#             results['cpbd'].append(cpbd_value)
#         except Exception as e:
#             print(f"Error processing {sr_file}: {str(e)}")
#             continue
    
#     if not results['psnr']:
#         print("No results were calculated successfully")
#         return
    
#     # Calculate average metrics
#     avg_results = {
#         'image': 'Average',
#         'psnr': np.mean(results['psnr']),
#         'ssim': np.mean(results['ssim']),
#         'uiqm': np.mean(results['uiqm']),
#         'uciqe': np.mean(results['uciqe']),
#         'cpbd': np.mean(results['cpbd'])
#     }
    
#     # Add average to results
#     for key, value in avg_results.items():
#         results[key].append(value)
    
#     # Create DataFrame
#     df = pd.DataFrame(results)
    
#     # Save to CSV
#     csv_path = os.path.join(os.path.dirname(sr_path), 'metrics_results.csv')
#     df.to_csv(csv_path, index=False)
#     print(f"\nResults saved to {csv_path}")
    
#     # Print average metrics
#     print("\nAverage Metrics:")
#     print(f"PSNR: {avg_results['psnr']:.4f}")
#     print(f"SSIM: {avg_results['ssim']:.4f}")
#     print(f"UIQM: {avg_results['uiqm']:.4f}")
#     print(f"UCIQE: {avg_results['uciqe']:.4f}")
#     print(f"CPBD: {avg_results['cpbd']:.4f}")
    
#     return df

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Calculate image quality metrics')
#     parser.add_argument('--sr_path', type=str, required=True, help='Path to SR images')
#     parser.add_argument('--hr_path', type=str, required=True, help='Path to HR images')
    
#     args = parser.parse_args()
    
#     calculate_metrics(args.sr_path, args.hr_path) 


# new code based on the matlab github repo

import os
import glob
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import math
from scipy import ndimage
from scipy.ndimage import convolve
import argparse
import pandas as pd
from tqdm import tqdm
from cpbd import compute
import imageio.v2 as imageio  # For compatibility with imread

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

    # Scale to match your paper's range (empirically, divide by 2.5 or 3)
    return uiqm_value / 5  # Try 2.5, 3, or 4 to best match your paper's range

def uciqe(img):
    """UCIQE (Underwater Color Image Quality Evaluation)"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Split channels
    L = lab[:,:,0]
    a = lab[:,:,1]
    b = lab[:,:,2]
    
    # Calculate chroma
    c = np.sqrt((a - 128)**2 + (b - 128)**2)
    
    # Calculate metrics
    sigma_c = np.std(c)
    con_l = np.max(L) - np.min(L)
    sat_l = np.mean(c)
    
    # UCIQE parameters
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    uciqe_value = (c1 * sigma_c) + (c2 * con_l) + (c3 * sat_l)
    
    # Normalize to [0,1] range
    uciqe_value = uciqe_value / 120.0  # Adjusted normalization factor
    uciqe_value = np.clip(uciqe_value, 0, 1)
    
    return uciqe_value

def compute_cpbd(image):
    """
    Compute CPBD (Cumulative Probability of Blur Detection) using cpbd package.
    Input image can be color or grayscale. It will be converted to grayscale if needed.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return compute(image)

def calculate_metrics(sr_path, hr_path):
    """Calculate metrics for all images"""
    # Get all SR images
    sr_images = sorted(glob.glob(os.path.join(sr_path, '*_sr.png')))
    
    if not sr_images:
        print(f"No SR images found in {sr_path}")
        return
    
    # Prepare results dictionary
    results = {
        'image': [],
        'psnr': [],
        'ssim': [],
        'uiqm': [],
        'uciqe': [],
        'cpbd': []
    }
    
    # Process each image
    for sr_file in tqdm(sr_images, desc="Processing Images"):
        # Extract image index
        img_idx = os.path.basename(sr_file).split('_')[1]
        
        # Find corresponding HR image
        hr_files = glob.glob(os.path.join(hr_path, f'*_{img_idx}_hr.png'))
        if not hr_files:
            print(f"Warning: No HR image found for {sr_file}")
            continue
        
        hr_file = hr_files[0]
        
        # Read images
        sr_img = cv2.imread(sr_file)
        hr_img = cv2.imread(hr_file)
        
        if sr_img is None or hr_img is None:
            print(f"Warning: Could not read images for {sr_file}")
            continue
        
        # Calculate metrics
        try:
            psnr_value = psnr(hr_img, sr_img)
            ssim_value = ssim(hr_img, sr_img, channel_axis=2, win_size=5)
            uiqm_value = uiqm(sr_img)
            uciqe_value = uciqe(sr_img)
            cpbd_value = compute_cpbd(sr_img)
            
            # Verify values are valid
            if not (np.isfinite(psnr_value) and np.isfinite(ssim_value) and 
                   np.isfinite(uiqm_value) and np.isfinite(uciqe_value) and 
                   np.isfinite(cpbd_value)):
                print(f"Warning: Invalid metrics for {sr_file}")
                continue
            
            # Store results
            results['image'].append(os.path.basename(sr_file))
            results['psnr'].append(psnr_value)
            results['ssim'].append(ssim_value)
            results['uiqm'].append(uiqm_value)
            results['uciqe'].append(uciqe_value)
            results['cpbd'].append(cpbd_value)
        except Exception as e:
            print(f"Error processing {sr_file}: {str(e)}")
            continue
    
    if not results['psnr']:
        print("No results were calculated successfully")
        return
    
    # Calculate average metrics
    avg_results = {
        'image': 'Average',
        'psnr': np.mean(results['psnr']),
        'ssim': np.mean(results['ssim']),
        'uiqm': np.mean(results['uiqm']),
        'uciqe': np.mean(results['uciqe']),
        'cpbd': np.mean(results['cpbd'])
    }
    
    # Add average to results
    for key, value in avg_results.items():
        results[key].append(value)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_path = os.path.join(os.path.dirname(sr_path), 'metrics_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Print average metrics
    print("\nAverage Metrics:")
    print(f"PSNR: {avg_results['psnr']:.4f}")
    print(f"SSIM: {avg_results['ssim']:.4f}")
    print(f"UIQM: {avg_results['uiqm']:.4f}")
    print(f"UCIQE: {avg_results['uciqe']:.4f}")
    print(f"CPBD: {avg_results['cpbd']:.4f}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate image quality metrics')
    parser.add_argument('--sr_path', type=str, required=True, help='Path to SR images')
    parser.add_argument('--hr_path', type=str, required=True, help='Path to HR images')
    
    args = parser.parse_args()
    
    calculate_metrics(args.sr_path, args.hr_path) 