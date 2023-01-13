import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from ui.ui import Ui_Form
from ui.mouse_event import GraphicsScene
import cv2
import numpy as np
import os
import time
import argparse

import cv2
import tensorflow as tf
import neuralgym as ng
from PIL import Image

from inpaint_model import InpaintCAModel
parser = argparse.ArgumentParser()
parser.add_argument(
    '--image_height', default=512, type=int,
    help='The height of images should be defined, otherwise batch mode is not'
         ' supported.')
parser.add_argument(
    '--image_width', default=512, type=int,
    help='The width of images should be defined, otherwise batch mode is not'
         ' supported.')
parser.add_argument(
    '--checkpoint_dir', default='/home/czey/generative_inpainting/logs/CTArmNpy512_v4/', type=str,
    help='The directory of tensorflow checkpoint.')


class Ex(QWidget, Ui_Form):
    def __init__(self, model, config):
        super().__init__()
        self.setupUi(self)
        self.show()
        self.model = model
        self.config = config
        # self.model.load_demo_graph(config)

        self.output_img = None

        self.mat_img = None

        self.ld_mask = None
        self.ld_sk = None

        self.modes = [0,0,0]
        self.mouse_clicked = False
        self.scene = GraphicsScene(self.modes)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.result_scene = QGraphicsScene()
        self.graphicsView_2.setScene(self.result_scene)
        self.graphicsView_2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.dlg = QColorDialog(self.graphicsView)
        self.color = None
        self.lineWidth = 10
        self.shown_ori = True

    def mode_select(self, mode):
        for i in range(len(self.modes)):
            self.modes[i] = 0
        self.modes[mode] = 1

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:
            image = np.load(fileName)
            image[image > 2500] = 2500
            image_show = (image / 2500 * 255).astype('uint8')
            image = np.sqrt(image / 2500) * 2 - 1
            if image is None:
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            self.image = Image.fromarray(image_show).toqpixmap()
            self.mat_img = image[np.newaxis, :, :, np.newaxis]
            self.scene.reset()
            if len(self.scene.items())>0:
                self.scene.reset_items()
            self.scene.addPixmap(self.image)
            if len(self.result_scene.items())>0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(self.image)

    def mask_mode(self):
        self.mode_select(0)

    def sketch_mode(self):
        self.mode_select(1)

    def open_model(self):
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                                                  QDir.currentPath())
        if fileName is not None:
            saver.restore(sess, fileName.split('.')[0])

    def valChange(self):
        self.lineWidth = self.splider.value()
        self.scene.get_lineWidth(self.lineWidth)
        self.label.setNum(self.lineWidth)

    def change_ori_res(self):
        self.shown_ori = not self.shown_ori
        if len(self.result_scene.items()) > 0:
            self.result_scene.removeItem(self.result_scene.items()[-1])
        if not self.shown_ori:
            self.result_scene.addPixmap(self.image)
        else:
            try:
                self.result_scene.addPixmap(self.result)
            except AttributeError:
                self.result_scene.addPixmap(self.image)

    def complete(self):
        sketch = self.make_sketch(self.scene.sketch_points)
        # stroke = self.make_stroke(self.scene.stroke_points)
        mask = self.make_mask(self.scene.mask_points)
        if not type(self.ld_mask)==type(None):
            ld_mask = np.expand_dims(self.ld_mask[:,:,0:1],axis=0)
            ld_mask[ld_mask>0] = 1
            ld_mask[ld_mask<1] = 0
            mask = mask+ld_mask
            mask[mask>0] = 1
            mask[mask<1] = 0
            mask = np.asarray(mask,dtype=np.uint8)
            print(mask.shape)

        if not type(self.ld_sk)==type(None):
            sketch = sketch+self.ld_sk
            sketch[sketch>0]=1 

        # noise = self.make_noise()

        sketch = sketch*mask
        start_t = time.time()

        input_image = np.concatenate([self.mat_img, sketch * 255, mask * 255], axis=2)

        result = sess.run(output, feed_dict={input_image_ph: input_image})
        # result = self.model.demo(self.config, batch)
        end_t = time.time()
        print('inference time : {}'.format(end_t-start_t))
        result_show = (result[0, :, :, 0] + 1) ** 2 / 4 * 255
        self.output_img = ((result[0, :, :, 0] + 1) ** 2 / 4 * 2500).astype('int16')
        self.result = Image.fromarray(result_show.astype('uint8')).toqpixmap()
        self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(self.result)

    def make_mask(self, pts):
        if len(pts)>0:
            mask = np.zeros((512,512,3))
            for pt in pts:
                cv2.line(mask,pt['prev'],pt['curr'],(255,255,255),self.lineWidth)
            mask = np.asarray(mask[:,:,0]/255,dtype=np.uint8)
            mask = np.expand_dims(mask,axis=2)
            mask = np.expand_dims(mask,axis=0)
        else:
            mask = np.zeros((512,512,3))
            mask = np.asarray(mask[:,:,0]/255,dtype=np.uint8)
            mask = np.expand_dims(mask,axis=2)
            mask = np.expand_dims(mask,axis=0)
        return mask

    def make_sketch(self, pts):
        if len(pts)>0:
            sketch = np.zeros((512,512,3))
            for pt in pts:
                cv2.line(sketch,pt['prev'],pt['curr'],(255,255,255),1)
            sketch = np.asarray(sketch[:,:,0]/255,dtype=np.uint8)
            sketch = np.expand_dims(sketch,axis=2)
            sketch = np.expand_dims(sketch,axis=0)
        else:
            sketch = np.zeros((512,512,3))
            # sketch = 255*sketch
            sketch = np.asarray(sketch[:,:,0]/255,dtype=np.uint8)
            sketch = np.expand_dims(sketch,axis=2)
            sketch = np.expand_dims(sketch,axis=0)
        return sketch

    def save_img(self):
        if type(self.output_img):
            fileName, _ = QFileDialog.getSaveFileName(self, "Save File",
                    QDir.currentPath())
            # cv2.imwrite(fileName+'.jpg',self.output_img)
            np.save(fileName+'.npy',self.output_img)

    def undo(self):
        self.scene.undo()

    def clear(self):
        self.scene.reset_items()
        self.scene.reset()
        if type(self.image):
            self.scene.addPixmap(self.image)


if __name__ == '__main__':
    tf.reset_default_graph()
    args = parser.parse_args()
    FLAGS = ng.Config('inpaint_CTimg.yml')
    FLAGS.guided = True
    config = FLAGS

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, args.image_height, args.image_width * 3, 1))
    xin, x1, x2, output = model.build_server_graph(FLAGS, input_image_ph)
    output = tf.reverse(output, [-1])
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    # saver = tf.train.Saver()
    # saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
    print('Model loaded.')

    app = QApplication(sys.argv)
    ex = Ex(model, config)
    sys.exit(app.exec_())


