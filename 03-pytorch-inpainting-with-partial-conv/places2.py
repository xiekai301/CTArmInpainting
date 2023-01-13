import random
import torch
from PIL import Image
from glob import glob
from scipy.io import loadmat
import numpy as np
import os
import cv2
from util.create_mask import random_bbox, generate_mask
from util.config import Config


class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # use about 8M images in the challenge dataset
        # if split == 'train':
        #     self.paths = glob('{:s}/*.mat'.format(img_root),
        #                       recursive=True)
        # else:
        #     self.paths = glob('{:s}/*'.format(img_root, split))
        self.paths = []
        for root, _, fnames in sorted(os.walk(img_root)):
            for fname in fnames:
                # if is_image_file(fname):
                if '.npy' in fname:
                    path = os.path.join(root, fname)
                    self.paths.append(path)
        self.paths.sort()

        self.mask_paths = glob('{:s}/*.png'.format(mask_root))
        self.mask_paths.sort()
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        # gt_img = Image.open(self.paths[index])
        # gt_img = self.img_transform(gt_img.convert('RGB'))

        img = np.load(self.paths[index])
        img[img > 2500] = 2500
        gt_img = np.sqrt(img/2500) * 2 - 1
        gt_img = np.float32(np.expand_dims(gt_img, axis=0))
        gt_img = torch.from_numpy(gt_img)

        # mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        # mask_path = '/home/czey/01-MRinpainting/00-pytorch-CycleGAN-xk_MRMAR/masks_forvalidation/' + str(index) + '.png'

        mask_path = os.path.dirname(self.paths[index]) + '_mask.png'
        # mask_path = self.paths[index][:-4] + '_mask.png'
        mask = 1 - cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) / 255
        mask = np.float32(np.expand_dims(mask, axis=0))

        # mask = generate_mask()  # 矩形mask, 在上一句基础上再随机內缩，最大32/2
        # mask = 1 - mask.astype(np.bool)
        # mask = np.float32(np.expand_dims(mask, axis=0))
        # mask = torch.from_numpy(mask)
        # return gt_img * mask + torch.ones_like(gt_img) * (1-mask) * (-1), mask, gt_img
        return gt_img * mask + torch.ones_like(gt_img) * (1-mask) * (-1), mask, gt_img, self.paths[index]

    def __len__(self):
        return len(self.paths)
