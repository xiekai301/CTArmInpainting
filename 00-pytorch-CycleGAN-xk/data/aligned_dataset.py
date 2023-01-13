import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np
import cv2


import math
from PIL import Image, ImageDraw
from util.config import Config
from util.create_mask import random_bbox, generate_hand_mask, generate_rect_mask, generate_stroke_mask


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.dir_A = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # get the image directory
        self.B_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths
        # self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        # self.FLAGS = Config('util/mask_parameters.yaml')


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index

        # A_path = self.A_paths[index]
        # B_path = A_path.replace('A.npy', 'B.npy')
        # FLAGS = self.FLAGS
        # bbox = random_bbox(FLAGS)  # 生成mask矩阵的左上角坐标和高宽
        # regular_mask = generate_rect_mask(bbox, FLAGS)  # 矩形mask, 在上一句基础上再随机內缩，最大32/2
        # irregular_mask = generate_stroke_mask(FLAGS)  # 类似笔刷, 设置随机角, 再用椭圆平滑


        B_path = self.B_paths[index]
        # mask = generate_hand_mask()
        mask_path = os.path.dirname(B_path) + '_mask.png'
        # mask_path = B_path[:-4] + '_mask.png'
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) / 255
        B = np.load(B_path)
        B = np.float32(np.expand_dims(B, axis=0))
        mask = np.float32(np.expand_dims(mask, axis=0))
        B[B > 2500] = 2500
        B = B / 2500 * 2 - 1
        A = B * (1 - mask) + np.ones_like(B) * mask * (-1)
        A = torch.from_numpy(A)
        B = torch.from_numpy(B)
        mask = torch.from_numpy(mask)

        return {'A': A, 'B': B, 'mask': mask, 'A_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        # return len(self.AB_paths)
        return len(self.B_paths)
