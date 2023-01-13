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
from medpy.metric.binary import dc, hd95
# from hausdorff.hausdorff import hausdorff_distance


def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', default='/home/czey/01-CTinpainting/z1_showandoutput_img/ori/',  help='Path to ground truth data', type=str)
    # parser.add_argument('--data-path', default='/home/czey/01-CTinpainting/generative_inpainting_MRMAR/training_data/MR_MAR/validation',  help='Path to ground truth data', type=str)
    # parser.add_argument('--output-path', default='/home/czey/01-CTinpainting/z1_showandoutput_img/unet/', help='Path to output data', type=str)
    # parser.add_argument('--output-path', default='/home/czey/01-CTinpainting/z1_showandoutput_img/pix2pix/', help='Path to output data', type=str)
    parser.add_argument('--output-path', default='/home/czey/01-CTinpainting/z1_showandoutput_img/pconv/', help='Path to output data', type=str)
    # parser.add_argument('--output-path', default='/home/czey/01-CTinpainting/generative_inpainting_MRMAR/training_data/MR_MAR/validation_output/', help='Path to output data', type=str)
    # parser.add_argument('--output-path', default='/home/czey/01-CTinpainting/generative_inpainting_MRMAR/training_data/MR_MAR/validation_toothmasktraining_output/', help='Path to output data', type=str)
    # parser.add_argument('--output-path', default='/home/czey/01-CTinpainting/z1_showandoutput_img/gatedconv/', help='Path to output data', type=str)
    # parser.add_argument('--output-path', default='/home/czey/01-CTinpainting/generative_inpainting_MRMAR/training_data/MR_MAR/validation_nogateandCA_output/', help='Path to output data', type=str)
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
dice = []
names = []
index = 1

# files = list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))
p_paths = [os.path.join(path_true, f) for f in sorted(os.listdir(path_true)) if os.path.isdir(os.path.join(path_true, f))]

for i, p_path in enumerate(p_paths):
    npy_num = len(os.listdir(p_path))
    imgs_gt = np.zeros((512,512, npy_num))
    imgs_pred = np.zeros((512,512, npy_num))
    masks = np.zeros((512,512, npy_num))
    for j, npy_name in enumerate(sorted(os.listdir(p_path))):
        npy_path = os.path.join(path_true, p_path, npy_name)
        img_gt = np.load(npy_path)
        img_gt[img_gt > 2500] = 2500
        img_pred = np.load(npy_path.replace(path_true, path_pred))

        mask_path = os.path.dirname(npy_path) + '_mask.png'
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.bool)
        mask = mask * (img_gt > 0)

        imgs_gt[:, :, j] = img_gt
        imgs_pred[:, :, j] = img_pred
        masks[:, :, j] = mask

    psnr.append(compare_psnr(imgs_gt, imgs_pred, data_range=2500))
    psnr_mask.append(compare_psnr_mask(imgs_gt, imgs_pred, masks, data_range=2500))
    ssim.append(compare_ssim(imgs_gt, imgs_pred, data_range=2500, win_size=51, multichannel=True) * 100)
    mae.append(compare_mae(imgs_gt, imgs_pred))
    mae_mask.append(compare_mae_mask(imgs_gt, imgs_pred, masks))
    dice.append(dc(imgs_gt*masks, imgs_pred*masks))


print(
    "PSNR: %.4f" % round(float(np.mean(psnr)), 4),
    "PSNR Variance: %.4f" % round(float(np.std(psnr)), 4),
    "PSNR_mask: %.4f" % round(float(np.mean(psnr_mask)), 4),
    "PSNR_mask Variance: %.4f" % round(float(np.std(psnr_mask)), 4),
    "SSIM: %.4f" % round(float(np.mean(ssim)), 4),
    "SSIM Variance: %.4f" % round(float(np.std(ssim)), 4),
    "MAE: %.4f" % round(float(np.mean(mae)), 4),
    "MAE Variance: %.4f" % round(float(np.std(mae)), 4),
    "MAE_mask: %.4f" % round(float(np.mean(mae_mask)), 4),
    "MAE_mask Variance: %.4f" % round(float(np.std(mae_mask)), 4),
    "dice_mask: %.4f" % round(float(np.mean(dice)), 4),
    "dice_mask Variance: %.4f" % round(float(np.std(dice)), 4)
)
