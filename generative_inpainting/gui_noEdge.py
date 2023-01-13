import os
import sys
from pathlib import Path

import cv2
# import imutils
import numpy as np
import torch
import tqdm

import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel
import matplotlib.pyplot as plt
from temp3 import generate_hand_mask

parser = argparse.ArgumentParser()
parser.add_argument(
    '--flist', default='/home/czey/generative_inpainting/data/CTArmNpy/validation_static_view.flist', type=str,
    help='The filenames of image to be processed: input, mask, output.')
parser.add_argument(
    '--image_height', default=512, type=int,
    help='The height of images should be defined, otherwise batch mode is not'
         ' supported.')
parser.add_argument(
    '--image_width', default=512, type=int,
    help='The width of images should be defined, otherwise batch mode is not'
         ' supported.')
parser.add_argument(
    '--checkpoint_dir', default='/home/czey/generative_inpainting/logs/CTArmNpy512_noEdge_v4/', type=str,
    help='The directory of tensorflow checkpoint.')

size = 512
img_path = '/home/czey/generative_inpainting/training_data/CTArmNpy/validation/Y180620/1070.npy'

print("press 'esc' to exit, 'i' to remask, 'Enter to inpainting'!")
drawing = False  # true if mouse is pressed
ix, iy = -1, -1
radius = 20


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing:
            cv2.circle(mask[0,:,:,0], (x, y), radius, 1, -1)
    else:
        pass


if __name__ == '__main__':

    tf.reset_default_graph()
    args = parser.parse_args()
    FLAGS = ng.Config('inpaint_CTimg.yml')
    FLAGS.guided = False

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, args.image_height, args.image_width * 2, 1))
    xin, x1, x2, output = model.build_server_graph(FLAGS, input_image_ph)
    # output = (output + 1.) * 1250
    output = tf.reverse(output, [-1])
    # output = tf.saturate_cast(output, tf.uint16)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    # saver = tf.train.Saver()
    # saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            args.checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    image = np.load(img_path)
    image[image > 2500] = 2500
    image = np.sqrt(image / 2500) * 2 - 1
    h, w = image.shape
    mask = np.zeros((h, w), np.uint8)
    image = np.expand_dims(image, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    image = np.expand_dims(image, axis=0)
    mask = np.expand_dims(mask, axis=0)

    x_pos, y_pos = 100, 300
    cv2.namedWindow('image_masked', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('image_masked', 1024, 1024)
    cv2.moveWindow("image_masked", x_pos, y_pos)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('result', 1024, 1024)
    cv2.moveWindow("result", x_pos + 1200, y_pos)
    cv2.imshow('result', np.zeros((512, 512)))
    cv2.createTrackbar('radius', 'image_masked', 10, 100, lambda x: None)
    cv2.setMouseCallback('image_masked', draw_circle)

    while (1):
        image_masked = (image[0, :, :, 0] * (1-mask[0, :, :, 0]) + 1) / 2
        cv2.imshow('image_masked', image_masked)
        radius = cv2.getTrackbarPos('radius', 'image_masked')
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Esc
            break
        elif k == ord('i'):
            image = image.copy()
            mask = np.zeros((1, h, w, 1), np.uint8)
        elif k == 13:  # Enter

            input_image = np.concatenate([image, mask * 255], axis=2)
            result = sess.run(output, feed_dict={input_image_ph: input_image})

            cv2.imshow('result', (result[0, :, :, 0] + 1) / 2)
            # k = cv2.waitKey(1) & 0xFF
            # if k == 27:
            #     break
            # elif k == ord('i'):
            #     image = image.copy()
            #     mask = np.zeros((1, h, w, 1), np.uint8)
            # else:
            #     mask = np.zeros((1, h, w, 1), np.uint8)
    cv2.destroyAllWindows()
