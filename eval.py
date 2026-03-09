import argparse
import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob
from metrics_util import compute_psnr, compute_ssim, uiqm, uciqe, compute_cpbd
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='experiments/basic_sr_ffhq_210809_142238/results')
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*_hr.png'.format(args.path)))
    fake_names = list(glob.glob('{}/*_sr.png'.format(args.path)))

    real_names.sort()
    fake_names.sort()

    writer = SummaryWriter(logdir='eval_tb_logs')  # You can change logdir as needed

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_uiqm = 0.0
    avg_uciqe = 0.0
    avg_cpbd = 0.0
    idx = 0
    for rname, fname in zip(real_names, fake_names):
        idx += 1
        ridx = rname.rsplit("_hr")[0]
        fidx = fname.rsplit("_sr")[0]
        assert ridx == fidx, f'Image ridx:{ridx}!=fidx:{fidx}'

        hr_img = np.array(Image.open(rname).convert('RGB'))
        sr_img = np.array(Image.open(fname).convert('RGB'))

        psnr = compute_psnr(hr_img, sr_img)
        ssim = compute_ssim(hr_img, sr_img)
        uiqm_val = uiqm(sr_img)
        uciqe_val = uciqe(sr_img)
        cpbd_val = compute_cpbd(sr_img)

        avg_psnr += psnr
        avg_ssim += ssim
        avg_uiqm += uiqm_val
        avg_uciqe += uciqe_val
        avg_cpbd += cpbd_val

        if idx % 20 == 0:
            print(f'Image:{idx}, PSNR:{psnr:.4f}, SSIM:{ssim:.4f}, UIQM:{uiqm_val:.4f}, UCIQE:{uciqe_val:.4f}, CPBD:{cpbd_val:.4f}')

    avg_psnr /= idx
    avg_ssim /= idx
    avg_uiqm /= idx
    avg_uciqe /= idx
    avg_cpbd /= idx

    # log to TensorBoard
    writer.add_scalar('Validation/PSNR', avg_psnr, 0)
    writer.add_scalar('Validation/SSIM', avg_ssim, 0)
    writer.add_scalar('Validation/UIQM', avg_uiqm, 0)
    writer.add_scalar('Validation/UCIQE', avg_uciqe, 0)
    writer.add_scalar('Validation/CPBD', avg_cpbd, 0)
    writer.close()

    # log
    print('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    print('# Validation # SSIM: {:.4e}'.format(avg_ssim))
    print('# Validation # UIQM: {:.4e}'.format(avg_uiqm))
    print('# Validation # UCIQE: {:.4e}'.format(avg_uciqe))
    print('# Validation # CPBD: {:.4e}'.format(avg_cpbd))