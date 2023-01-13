import argparse
import torch
from torchvision import transforms

import opt
from places2 import Places2
from evaluation import evaluate
from net import PConvUNet
from util.io import load_ckpt
from torch.utils import data
import numpy as np
import os
import time

parser = argparse.ArgumentParser()
# training options
# parser.add_argument('--valroot', type=str, default='/home/czey/01-MRinpainting/generative_inpainting_MRMAR/training_data/MR_MAR/validation')
# parser.add_argument('--valroot', type=str, default='/home/czey/01-CTinpainting/03-pytorch-inpainting-with-partial-conv/datasets/test_truepatients')
parser.add_argument('--valroot', type=str, default='/home/czey/01-CTinpainting/03-pytorch-inpainting-with-partial-conv/datasets/test_forTPS')
# parser.add_argument('--valroot', type=str, default='/home/czey/01-CTinpainting/03-pytorch-inpainting-with-partial-conv/datasets/validation_fortrun0.4')
parser.add_argument('--mask_root', type=str, default='')
parser.add_argument('--snapshot', type=str, default='logs/100000.pth')
parser.add_argument('--image_size', type=int, default=256)
args = parser.parse_args()

device = torch.device('cuda')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

dataset_val = Places2(args.valroot, args.mask_root, img_transform, mask_transform, 'val')
iterator_val = data.DataLoader(
    dataset_val, batch_size=1, num_workers=0)

# model = PConvUNet().to(device)
model = PConvUNet(input_channels=1).to(device)
start_iter = load_ckpt(args.snapshot, [('model', model)])

model.eval()
# evaluate(model, dataset_val, device, 'result.jpg')
# image, mask, gt, path = [x.to(device) for x in next(iterator_val)]
iter_start_time = time.time()
with torch.no_grad():
    for i, (image, mask, gt, path) in enumerate(iterator_val):

        output, _ = model(image.to(device), mask.to(device))
        output = output.to(torch.device('cpu'))
        output_comp = mask * image + (1 - mask) * output

        img_name = path[0].split('/')[-1]
        patient_name = path[0].split('/')[-2]
        out_patient_path = os.path.join('output', patient_name)
        if not os.path.exists(out_patient_path):
            os.mkdir(out_patient_path)
        # print('processing (%04d)-th image... %s' % (i, path))
        np.save(os.path.join(out_patient_path, img_name[:-4] + '.npy'), np.uint16(np.around(((output_comp[0, 0, :, :]+1)/2) ** 2 * 2500)))
t_comp = (time.time() - iter_start_time) / len(iterator_val)
print(t_comp)