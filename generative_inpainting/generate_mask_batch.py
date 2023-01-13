import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

for img_name in sorted(os.listdir('./validation_data_10_imgs')):
    print(img_name)
    img = cv2.imread('./validation_data_10_imgs/' + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img[100:170, 100:170, :] = 255
    cv2.imwrite('./validation_data_10_imgs/' + img_name.split('.')[0] + '_after.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#
    img_mask = cv2.imread('10_mask.png', cv2.IMREAD_UNCHANGED)
    img_mask = cv2.resize(img_mask, (256,256))
    img_mask[:, :, :] = 0
    img_mask[100:170, 100:170, :] = 255
    cv2.imwrite('./validation_data_10_imgs/' + img_name.split('.')[0] + '_mask.png', img_mask)

