import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    FLAGS = ng.Config('inpaint_CTimg.yml')
    # ng.get_gpus(1)
    args, unknown = parser.parse_known_args()

    model = InpaintCAModel()
    # image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)  # change, add cv2.IMREAD_UNCHANGED
    # mask = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)  # BGR change, add cv2.IMREAD_UNCHANGED
    image = np.load(args.image)  # change, add cv2.IMREAD_UNCHANGED  change
    mask = np.load(args.mask)  # BGR change, add cv2.IMREAD_UNCHANGED
    image = np.expand_dims(image, -1)  # change ,add np.expand_dims(image, -1)
    mask = np.expand_dims(mask, -1)
    # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)

    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    # 自己补充： 裁剪成256*256
    # image = cv2.resize(image, (256, 256))  # if use this line ,shape changes
    # mask = cv2.resize(mask, (256, 256))

    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image)
        # output = (output + 1.) * 127.5  # (1,1024,1024,3)
        output = output * FLAGS.std_value + FLAGS.mean_value # (1,1024,1024,3)  # change
        output = tf.clip_by_value(output, 0, 4095)
        output = tf.reverse(output, [-1])  # RGB
        # output = tf.saturate_cast(output, tf.uint8)
        output = tf.saturate_cast(output, tf.uint16)  # change
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:  # 通过这种val形式恢复参数
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        # cv2.imwrite(args.output, result[0][:, :, ::-1])  # BGR
        np.save(args.output, result[0][:, :, 0])  # change