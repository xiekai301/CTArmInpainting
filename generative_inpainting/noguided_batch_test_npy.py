import time
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
    # '--flist', default='/home/czey/01-CTinpainting/generative_inpainting/data/CTArmNpy/validation_static_view.flist', type=str,
    # '--flist', default='/home/czey/01-CTinpainting/generative_inpainting/data/CTArmNpy/test_static_view_truepatients.flist', type=str,
    '--flist', default='/home/czey/01-CTinpainting/generative_inpainting/data/CTArmNpy/test_static_view_forTPS.flist', type=str,
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
    '--checkpoint_dir', default='/home/czey/01-CTinpainting/generative_inpainting/logs/CTArmNpy512_noEdge_211211', type=str,
    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    # ng.get_gpus(0)
    # os.environ['CUDA_VISIBLE_DEVICES'] =''
    tf.reset_default_graph()
    args = parser.parse_args()
    FLAGS = ng.Config('inpaint_CTimg.yml')

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, args.image_height, args.image_width*2, 1))
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
        # print(from_name)
        var_value = tf.contrib.framework.load_variable(
            args.checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    with open(args.flist, 'r') as f:
        lines = f.read().splitlines()
    t = time.time()
    for i, line in enumerate(lines):
        if i > -1:
            image_path = line
            # out = image[:-4] + '_completed.npy'
            image = np.load(image_path)
            image[image > 2500] = 2500
            image = np.sqrt(image / 2500) * 2 - 1
            mask_path = os.path.dirname(image_path) + '_mask.png'
            # mask_path = image_path[:-4] + '_mask.png'
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            image = np.expand_dims(image, axis=-1)
            mask = np.expand_dims(mask, axis=-1)

            assert image.shape == mask.shape

            h, w, _ = image.shape
            grid = 4
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid, :]
            # print('Shape of image: {} : {}'.format(image.shape, i))

            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)


            # load pretrained model
            # xin_out = sess.run(xin, feed_dict={input_image_ph: input_image})
            # x1_out = sess.run(x1, feed_dict={input_image_ph: input_image})
            # x2_out = sess.run(x2, feed_dict={input_image_ph: input_image})
            result = sess.run(output, feed_dict={input_image_ph: input_image})

            # print('Processed: {}'.format(image_path))
            # npy_path = image_path.replace('validation_fortrun0.4', 'validation_fortrun0.4_output')
            # npy_path = image_path.replace('test_truepatients', 'test_truepatients_output')
            npy_path = image_path.replace('test_forTPS', 'test_forTPS_output')
            if not os.path.exists(os.path.dirname(npy_path)):
                os.mkdir(os.path.dirname(npy_path))

            result_transform_back = np.uint16(np.around(((result[0, :, :, 0]+1)/2) ** 2 * 2500))
            np.save(npy_path, result_transform_back)

    print('Time : {}'.format((time.time() - t)/len(lines)))
