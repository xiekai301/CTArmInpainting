import numpy as np
import argparse
import matplotlib.pyplot as plt

from glob import glob
from ntpath import basename
# from scipy.misc import imread
from imageio import imread
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
import os
from skimage.color import rgb2gray
import cv2
from scipy.io import loadmat


def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', default='/home/czey/01-CTimpainting/z1_showandoutput_img/ori/',  help='Path to ground truth data', type=str)
    # parser.add_argument('--data-path', default='/home/czey/01-MRinpainting/generative_inpainting_MRMAR/training_data/MR_MAR/validation',  help='Path to ground truth data', type=str)
    # parser.add_argument('--output-path', default='/home/czey/01-MRinpainting/z1_showandoutput_img/unet/', help='Path to output data', type=str)
    # parser.add_argument('--output-path', default='/home/czey/01-MRinpainting/z1_showandoutput_img/pix2pix/', help='Path to output data', type=str)
    # parser.add_argument('--output-path', default='/home/czey/01-MRinpainting/z1_showandoutput_img/pconv/', help='Path to output data', type=str)
    # parser.add_argument('--output-path', default='/home/czey/01-MRinpainting/generative_inpainting_MRMAR/training_data/MR_MAR/validation_output/', help='Path to output data', type=str)
    # parser.add_argument('--output-path', default='/home/czey/01-MRinpainting/generative_inpainting_MRMAR/training_data/MR_MAR/validation_toothmasktraining_output/', help='Path to output data', type=str)
    parser.add_argument('--output-path', default='/home/czey/01-CTimpainting/z1_showandoutput_img/gatedconv/', help='Path to output data', type=str)
    # parser.add_argument('--output-path', default='/home/czey/01-MRinpainting/generative_inpainting_MRMAR/training_data/MR_MAR/validation_nogateandCA_output/', help='Path to output data', type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args


def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    # return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)
    return np.sum(np.abs(img_true - img_test)) / img_true.size

def compare_mae_mask(img_true, img_test,mask):
    img_true = (img_true*mask).astype(np.float32)
    img_test = (img_test*mask).astype(np.float32)
    # return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)
    return np.sum(np.abs(img_true - img_test)) / np.sum(mask)


def mean_squared_error_mask(img_true, img_test, mask):
    img_true = (img_true*mask).astype(np.float32)
    img_test = (img_test*mask).astype(np.float32)
    return np.sum((img_true - img_test) ** 2) / np.sum(mask)

def compare_psnr_mask(image_true, image_test, mask, data_range=None):
    err = mean_squared_error_mask(image_true, image_test, mask)
    return 10 * np.log10((data_range ** 2) / err)


args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

path_true = args.data_path
path_pred = args.output_path

psnr = []
psnr_mask = []
ssim = []
mae = []
mae_mask = []
names = []
index = 1

# files = list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))
files = list(glob(path_true + '/*/*.npy'))
for i, fn in enumerate(sorted(files)):
    name = basename(str(fn))
    names.append(name)

    img_gt = np.load(fn)
    img_gt[img_gt>2500] = 2500
    # img_pred = (imread(path_pred + '/' + basename(str(fn))) / 255.0).astype(np.float32)
    # img_pred = np.load(path_pred + '/' + basename(str(fn)))
    img_pred = np.load(fn.replace(path_true, path_pred))
    mask_path = os.path.dirname(fn) + '_mask.png'
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask = mask.astype(np.bool)
    mask = mask * (img_gt > 0)

    psnr.append(compare_psnr(img_gt, img_pred, data_range=2500))
    psnr_mask.append(compare_psnr_mask(img_gt, img_pred, mask, data_range=2500))
    ssim.append(compare_ssim(img_gt, img_pred, data_range=2500, win_size=51) * 100)
    mae.append(compare_mae(img_gt, img_pred))
    mae_mask.append(compare_mae_mask(img_gt, img_pred, mask))

print(
    "PSNR: %.4f" % round(np.mean(psnr), 4),
    "PSNR Variance: %.4f" % round(np.std(psnr), 4),
    "PSNR_mask: %.4f" % round(np.mean(psnr_mask), 4),
    "PSNR_mask Variance: %.4f" % round(np.std(psnr_mask), 4),
    "SSIM: %.4f" % round(np.mean(ssim), 4),
    "SSIM Variance: %.4f" % round(np.std(ssim), 4),
    "MAE: %.4f" % round(np.mean(mae), 4),
    "MAE Variance: %.4f" % round(np.std(mae), 4),
    "MAE_mask: %.4f" % round(np.mean(mae_mask), 4),
    "MAE_mask Variance: %.4f" % round(np.std(mae_mask), 4)
)
