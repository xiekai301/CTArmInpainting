""" common model for DCGAN """
import logging

import cv2
import neuralgym as ng
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from neuralgym.models import Model
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.ops.summary_ops import gradients_summary
from neuralgym.ops.layers import flatten, resize
from neuralgym.ops.gan_ops import gan_hinge_loss
from neuralgym.ops.gan_ops import random_interpolates

from inpaint_ops import gen_conv, gen_deconv, dis_conv
from inpaint_ops import random_bbox, bbox2mask, local_patch, brush_stroke_mask, rectcirc_hand_mask
from inpaint_ops import resize_mask_like, contextual_attention


logger = logging.getLogger()


class InpaintCAModel(Model):
    def __init__(self):
        super().__init__('InpaintCAModel')

    def build_inpaint_net(self, x, mask, reuse=False,
                          training=True, padding='SAME', name='inpaint_net'):  # change pading to 'SYMMETRIC'
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        xin = x  # (6, 384, 384, 3)
        offset_flow = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1]    # 这里为何添加一个ones_x(单通道全1)?  # change, add 9.27
        # x = tf.concat([x, ones_x, ones_x*mask], axis=3)
        x = tf.concat([x, ones_x*mask], axis=3)
        # x = tf.concat([ones_x, x, ones_x, ones_x*mask], axis=3)  # 原文修改
        # x = tf.concat([tf.tile(x, [1, 1, 1, 3]), ones_x, ones_x*mask], axis=3)

        # two stage network
        cnum = 48
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([gen_conv, gen_deconv],
                          training=training, padding=padding):
            # stage1
            x = gen_conv(x, cnum, 5, 1, name='conv1')  # (6, 384, 384, 24)  split, 所以输出只有cnum/2
            x = gen_conv(x, 2*cnum, 3, 2, name='conv2_downsample')  # (6, 192, 192, 48)
            x = gen_conv(x, 2*cnum, 3, 1, name='conv3')  # (6, 192, 192, 48)
            x = gen_conv(x, 4*cnum, 3, 2, name='conv4_downsample')  # (6, 96, 96, 96)
            x = gen_conv(x, 4*cnum, 3, 1, name='conv5')  # (6, 96, 96, 96)
            x = gen_conv(x, 4*cnum, 3, 1, name='conv6')  # (6, 96, 96, 96)
            mask_s = resize_mask_like(mask, x)  # mask_s: (1,96,96,1), mask: (1,384,384,1)
            # x = gen_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous')  # (6, 96, 96, 96)
            # x = gen_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous')  # (6, 96, 96, 96)
            # x = gen_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous')  # (6, 96, 96, 96)
            # x = gen_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous')  # (6, 96, 96, 96)
            x = gen_conv(x, 4 * cnum, 3, rate=1, name='conv7_atrous')  # (6, 96, 96, 96)
            x = gen_conv(x, 4 * cnum, 3, rate=2, name='conv8_atrous')  # (6, 96, 96, 96)
            x = gen_conv(x, 4 * cnum, 3, rate=2, name='conv9_atrous')  # (6, 96, 96, 96)
            x = gen_conv(x, 4 * cnum, 3, rate=2, name='conv10_atrous')  # (6, 96, 96, 96)
            x = gen_conv(x, 4*cnum, 3, 1, name='conv11')  # (6, 96, 96, 96)
            x = gen_conv(x, 4*cnum, 3, 1, name='conv12')  # (6, 96, 96, 96)
            x = gen_deconv(x, 2*cnum, name='conv13_upsample')  # (6, 192, 192, 48)
            x = gen_conv(x, 2*cnum, 3, 1, name='conv14')  # (6, 192, 192, 48)
            x = gen_deconv(x, cnum, name='conv15_upsample')  # (6, 384, 384, 24)
            x = gen_conv(x, cnum//2, 3, 1, name='conv16')  # (6, 384, 384, 12)
            x = gen_conv(x, 1, 3, 1, activation=None, name='conv17')  # (6, 384, 384, 3)  # change 3 to 1
            x = tf.nn.tanh(x)  # (6, 384, 384, 3)
            x_stage1 = x  # x_stage1: (6, 384, 384, 3)

            # stage2, paste result as input
            x = x*mask + xin[:, :, :, 0:3]*(1.-mask)  # (6, 384, 384, 3)
            x.set_shape(xin[:, :, :, 0:3].get_shape().as_list())  # (6, 384, 384, 3) shape不是已经一致了吗？防止中间出现偏差吗？
            # conv branch
            # xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)
            xnow = x  # (6, 384, 384, 3)
            x = gen_conv(xnow, cnum, 5, 1, name='xconv1')  # (6, 384, 384, 24)
            x = gen_conv(x, cnum, 3, 2, name='xconv2_downsample')  # (6, 192, 192, 24)
            x = gen_conv(x, 2*cnum, 3, 1, name='xconv3')  # (6, 192, 192, 48)
            x = gen_conv(x, 2*cnum, 3, 2, name='xconv4_downsample')  # (6, 96, 96, 48)
            x = gen_conv(x, 4*cnum, 3, 1, name='xconv5')  # (6, 96, 96, 96)
            x = gen_conv(x, 4*cnum, 3, 1, name='xconv6')  # (6, 96, 96, 96)
            # x = gen_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous')  # (6, 96, 96, 96)
            # x = gen_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous')  # (6, 96, 96, 96)
            # x = gen_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous')  # (6, 96, 96, 96)
            # x = gen_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous')  # (6, 96, 96, 96)
            x = gen_conv(x, 4 * cnum, 3, rate=1, name='xconv7_atrous')  # (6, 96, 96, 96)
            x = gen_conv(x, 4 * cnum, 3, rate=2, name='xconv8_atrous')  # (6, 96, 96, 96)
            x = gen_conv(x, 4 * cnum, 3, rate=2, name='xconv9_atrous')  # (6, 96, 96, 96)
            x = gen_conv(x, 4 * cnum, 3, rate=2, name='xconv10_atrous')  # (6, 96, 96, 96)
            x_hallu = x  # (6, 96, 96, 96)
            # attention branch
            x = gen_conv(xnow, cnum, 5, 1, name='pmconv1')  # (6, 384, 384, 24)
            x = gen_conv(x, cnum, 3, 2, name='pmconv2_downsample')  # (6, 192, 192, 24)
            x = gen_conv(x, 2*cnum, 3, 1, name='pmconv3')  # (6, 192, 192, 48)
            x = gen_conv(x, 4*cnum, 3, 2, name='pmconv4_downsample')  # (6, 96, 96, 96)
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv5')  # (6, 96, 96, 96)
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv6',
                                activation=tf.nn.relu)  # (6, 96, 96, 96)
            x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)  # (6, 96, 96, 96), (6, 96, 96, 3)
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv9')  # (6, 96, 96, 96)
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv10')  # (6, 96, 96, 96)
            pm = x  # (6, 96, 96, 96)
            x = tf.concat([x_hallu, pm], axis=3)  # (6, 96, 96, 192)

            x = gen_conv(x, 4*cnum, 3, 1, name='allconv11')  # (6, 96, 96, 96)
            x = gen_conv(x, 4*cnum, 3, 1, name='allconv12')  # (6, 96, 96, 96)
            x = gen_deconv(x, 2*cnum, name='allconv13_upsample')  # (6, 192, 192, 48)
            x = gen_conv(x, 2*cnum, 3, 1, name='allconv14')  # (6, 192, 192, 48)
            x = gen_deconv(x, cnum, name='allconv15_upsample')  # (6, 384, 384, 24)
            x = gen_conv(x, cnum//2, 3, 1, name='allconv16')  # (6, 384, 384, 12)
            x = gen_conv(x, 1, 3, 1, activation=None, name='allconv17')  # (6, 384, 384, 3), 这里改维度 change 3 to 1
            x = tf.nn.tanh(x)  # (6, 384, 384, 3)
            x_stage2 = x  # (6, 384, 384, 3)
        return x_stage1, x_stage2, offset_flow  # (6, 384, 384, 3), (6, 384, 384, 3), (6, 96, 96, 3)

    def build_sn_patch_gan_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope('sn_patch_gan', reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, name='conv1', training=training)  # (12,192,192,64)
            x = dis_conv(x, cnum*2, name='conv2', training=training)  # (12,96,96,128)
            x = dis_conv(x, cnum*4, name='conv3', training=training)  # (12,48,48,256)
            x = dis_conv(x, cnum*4, name='conv4', training=training)  # (12,24,24,256)
            x = dis_conv(x, cnum*4, name='conv5', training=training)  # (12,12,12,256)
            x = dis_conv(x, cnum*4, name='conv6', training=training)  # (12,6,6,256)
            x = flatten(x, name='flatten')  # 这里与文章内不同, 注意？  # (12,9216)
            return x

    def build_gan_discriminator(
            self, batch, reuse=False, training=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            d = self.build_sn_patch_gan_discriminator(
                batch, reuse=reuse, training=training)
            return d

    def build_graph_with_losses(
            self, FLAGS, batch_data, training=True, summary=False,
            reuse=False):
        if FLAGS.guided:
            batch_data, edge = batch_data
            edge = edge[:, :, :, 0:1] / 255.
            edge = tf.cast(edge > FLAGS.edge_threshold, tf.float32)
        # batch_pos = batch_data / 127.5 - 1.
        batch_pos = batch_data
        # batch_pos = (batch_data - FLAGS.mean_value) / FLAGS.std_value  # change [-1,1] to [-0.5,77](may)
        # generate mask, 1 represents masked point
        bbox = random_bbox(FLAGS)  # 生成mask矩阵的左上角坐标和高宽
        # regular_mask = bbox2mask(FLAGS, bbox, name='mask_c')  # 矩形mask, 在上一句基础上再随机內缩，最大32/2
        # irregular_mask = brush_stroke_mask(FLAGS, name='mask_c')  # 类似笔刷, 设置随机角, 再用椭圆平滑
        hand_mask = rectcirc_hand_mask(FLAGS, name='mask_c')
        # mask_old = tf.cast(
        #     tf.logical_or(
        #         tf.cast(irregular_mask, tf.bool),
        #         tf.cast(regular_mask, tf.bool),
        #     ),
        #     tf.float32
        # )
        # mask = tf.cast(
        #     tf.logical_or(
        #         tf.cast(mask_old, tf.bool),
        #         tf.cast(hand_mask, tf.bool),
        #     ),
        #     tf.float32
        # )
        mask = hand_mask

        # batch_incomplete = batch_pos*(1.-mask)  # mask部分为0
        batch_incomplete = batch_pos*(1.-mask) + tf.ones_like(batch_pos)*mask*(-1.)  # mask部分为0
        if FLAGS.guided:
            edge = edge * mask
            xin = tf.concat([batch_incomplete, edge], axis=3)
        else:
            xin = batch_incomplete
        x1, x2, offset_flow = self.build_inpaint_net(
            xin, mask, reuse=reuse, training=training,
            padding=FLAGS.padding)  # batch_incomplete输入网络, 生成输出图, offset_flow还不清楚, offset_flow (16,64,64,3)
        batch_predicted = x2
        losses = {}
        # apply mask and complete image
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)  # 生成合成图
        # local patches
        # losses['ae_loss'] = FLAGS.l1_loss_alpha * tf.reduce_mean(tf.abs(batch_pos - x1))  # ae损失函数
        # losses['ae_loss'] += FLAGS.l1_loss_alpha * tf.reduce_mean(tf.abs(batch_pos - x2))
        # if summary:
        #     scalar_summary('losses/ae_loss', losses['ae_loss'])
        losses['ae_loss1'] = FLAGS.l1_loss_alphax1 * tf.reduce_mean(tf.abs(batch_pos - x1))  # ae损失函数
        losses['ae_loss2'] = FLAGS.l1_loss_alphax2 * tf.reduce_mean(tf.abs(batch_pos - x2))
        losses['ae_loss'] = losses['ae_loss1'] + losses['ae_loss2']
        if summary:
            scalar_summary('losses/ae_loss1', losses['ae_loss1'])
            scalar_summary('losses/ae_loss2', losses['ae_loss2'])
            scalar_summary('losses/ae_loss', losses['ae_loss'])
            if FLAGS.guided:
                # viz_img = [
                #     batch_pos,
                #     batch_incomplete + edge,
                #     batch_complete]
                viz_img = [tf.tile(batch_pos, [1, 1, 1, 3]), tf.tile(batch_incomplete + edge, [1, 1, 1, 3]),
                           tf.tile(x1, [1, 1, 1, 3]), tf.tile(x2, [1, 1, 1, 3]),
                           tf.tile(batch_complete, [1, 1, 1, 3])]  # change
            else:
                # viz_img = [batch_pos, batch_incomplete, batch_complete]
                viz_img = [tf.tile(batch_pos, [1,1,1,3]), tf.tile(batch_incomplete, [1,1,1,3]),
                           tf.tile(x1, [1,1,1,3]), tf.tile(x2, [1,1,1,3]),
                           tf.tile(batch_complete, [1,1,1,3])]  # change
            if offset_flow is not None:
                viz_img.append(
                    resize(offset_flow, scale=4,
                           func=tf.image.resize_bilinear))
            # images_summary(  # TODO 查看打印的输出图像
            #     tf.concat(viz_img, axis=2),
            #     'raw_incomplete_predicted_complete', FLAGS.viz_max_out)  # 对应上tensorboard中的输出, 但这个有两个(g和d调用两次)
            images_summary(  # TODO 查看打印的输出图像
                tf.concat(viz_img, axis=2),
                'raw_incomplete_predicted_complete', FLAGS.viz_max_out, image_width=FLAGS.img_shapes[0])  # 对应上tensorboard中的输出, 但这个有两个(g和d调用两次)

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)  # (12,384,384,3)
        if FLAGS.gan_with_mask:
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(mask, [FLAGS.batch_size*2, 1, 1, 1])], axis=3)
            # (32,256,256,4)  # (12,384,384,4)
        if FLAGS.guided:
            # conditional GANs
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(edge, [2, 1, 1, 1])], axis=3)
        # wgan with gradient penalty
        if FLAGS.gan == 'sngan':
            pos_neg = self.build_gan_discriminator(batch_pos_neg, training=training, reuse=reuse)  # 输入判别网络
            # (32, 4096)  # (12,9216)
            pos, neg = tf.split(pos_neg, 2)  # 2个(16, 4096)  # (6,9216)
            g_loss, d_loss = gan_hinge_loss(pos, neg)  # 能理解了,将pos和neg打包送进dis再拆分,计算dis和gen的loss
            losses['g_loss'] = g_loss  # d和g损失函数
            losses['d_loss'] = d_loss
        else:
            raise NotImplementedError('{} not implemented.'.format(FLAGS.gan))
        if summary:
            # summary the magnitude of gradients from different losses w.r.t. predicted image
            # gradients_summary(losses['g_loss'], batch_predicted, name='g_loss')  # 这里batch_predicted==x2啊?
            gradients_summary(losses['g_loss'], batch_predicted,
                              name='gradient_loss/g_loss_to_batch_predicted')  # 这里batch_predicted==x2啊?
            gradients_summary(losses['g_loss'], x2, name='gradient_loss/g_loss_to_x2')  # 保存梯度损失函数
            # gradients_summary(losses['ae_loss'], x1, name='ae_loss_to_x1')  # TODO 这里没有运行,后面查看输出维度
            gradients_summary(losses['ae_loss'], x2, name='gradient_loss/ae_loss_to_x2')
        losses['g_loss'] = FLAGS.gan_loss_alpha * losses['g_loss']
        if FLAGS.ae_loss:  # True
            losses['g_loss'] += losses['ae_loss']
        losses['g_losswithoutae'] = losses['g_loss'] - losses['ae_loss']
        losses['all_loss'] = losses['g_loss']+losses['d_loss']
        if summary:
            scalar_summary('losses/g_losswithoutae', losses['g_losswithoutae'])
            scalar_summary('losses/d_loss', losses['d_loss'])
            scalar_summary('losses/all_loss', losses['all_loss'])
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')  # 应该是保留inpaint_net和discriminator的训练变量
        return g_vars, d_vars, losses

    def build_infer_graph(self, FLAGS, batch_data, bbox=None, name='val'):
        """
        """
        if FLAGS.guided:
            batch_data, edge = batch_data
            edge = edge[:, :, :, 0:1] / 255.
            edge = tf.cast(edge > FLAGS.edge_threshold, tf.float32)
        # regular_mask = bbox2mask(FLAGS, bbox, name='mask_c')  # (1,384,384,1)
        # irregular_mask = brush_stroke_mask(FLAGS, name='mask_c')
        hand_mask = rectcirc_hand_mask(FLAGS, name='mask_c')
        # mask_old = tf.cast(
        #     tf.logical_or(
        #         tf.cast(irregular_mask, tf.bool),
        #         tf.cast(regular_mask, tf.bool),
        #     ),
        #     tf.float32
        # )
        # mask = tf.cast(
        #     tf.logical_or(
        #         tf.cast(mask_old, tf.bool),
        #         tf.cast(hand_mask, tf.bool),
        #     ),
        #     tf.float32
        # )
        mask = hand_mask

        # batch_pos = batch_data / 127.5 - 1.
        batch_pos = batch_data
        # batch_pos = (batch_data - FLAGS.mean_value) / FLAGS.std_value  # change [-1,1] to [-0.5,77](may)
        # batch_incomplete = batch_pos*(1.-mask)
        batch_incomplete = batch_pos*(1.-mask) + tf.ones_like(batch_pos)*mask*(-1.)  # mask部分为0
        if FLAGS.guided:
            edge = edge * mask
            xin = tf.concat([batch_incomplete, edge], axis=3)
        else:
            xin = batch_incomplete
        # inpaint
        x1, x2, offset_flow = self.build_inpaint_net(  # 同之前,注意offset_flow已经是图像(意味着光流已经可视化)
            xin, mask, reuse=True,
            training=False, padding=FLAGS.padding)
        batch_predicted = x2
        # apply mask and reconstruct
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        # global image visualization
        if FLAGS.guided:
            # viz_img = [
            #     batch_pos,
            #     batch_incomplete + edge,
            #     batch_complete]
            viz_img = [tf.tile(batch_pos, [1, 1, 1, 3]), tf.tile(batch_incomplete + edge, [1, 1, 1, 3]),
                       tf.tile(x1, [1, 1, 1, 3]), tf.tile(x2, [1, 1, 1, 3]),
                       tf.tile(batch_complete, [1, 1, 1, 3])]  # 'list', 三个(1,384,384,3) change
        else:
            # viz_img = [batch_pos, batch_incomplete, batch_complete]  # 'list', 三个(1,384,384,3)
            viz_img = [tf.tile(batch_pos, [1,1,1,3]), tf.tile(batch_incomplete, [1,1,1,3]),
                       tf.tile(x1, [1,1,1,3]), tf.tile(x2, [1,1,1,3]),
                       tf.tile(batch_complete, [1,1,1,3])]  # 'list', 三个(1,384,384,3) change
        if offset_flow is not None:
            viz_img.append(
                resize(offset_flow, scale=4,
                       func=tf.image.resize_bilinear))
        # images_summary(
        #     tf.concat(viz_img, axis=2),
        #     name+'_raw_incomplete_complete', FLAGS.viz_max_out)
        images_summary(
            tf.concat(viz_img, axis=2), name + '_raw_incomplete_complete', FLAGS.viz_max_out,
            image_width=FLAGS.img_shapes[0])
        return batch_complete

    def build_static_infer_graph(self, FLAGS, batch_data, name):
        """
        """
        # generate mask, 1 represents masked point
        bbox = (tf.constant(FLAGS.height//2), tf.constant(FLAGS.width//2),
                tf.constant(FLAGS.height), tf.constant(FLAGS.width))
        return self.build_infer_graph(FLAGS, batch_data, bbox, name)


    def build_server_graph(self, FLAGS, batch_data, reuse=False, is_training=False):  # 只用于test
        """
        """
        # generate mask, 1 represents masked point
        if FLAGS.guided:
            batch_raw, edge, masks_raw = tf.split(batch_data, 3, axis=2)
            edge = edge[:, :, :, 0:1] / 255.
            edge = tf.cast(edge > FLAGS.edge_threshold, tf.float32)
        else:
            batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
        masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

        # batch_pos = batch_raw / 127.5 - 1.
        batch_pos = batch_raw
        # batch_pos = (batch_raw - FLAGS.mean_value) / FLAGS.std_value  # change [-1,1] to [-0.5,77](may)
        # batch_incomplete = batch_pos * (1. - masks)
        batch_incomplete = batch_pos * (1. - masks) + tf.ones_like(batch_pos) * masks * (-1.)  # mask部分为0

        if FLAGS.guided:
            edge = edge * masks[:, :, :, 0:1]
            xin = tf.concat([batch_incomplete, edge], axis=3)
        else:
            xin = batch_incomplete
        # inpaint
        x1, x2, flow = self.build_inpaint_net(
            xin, masks, reuse=reuse, training=is_training)
        batch_predict = x2
        # apply mask and reconstruct
        batch_complete = batch_predict*masks + batch_incomplete*(1-masks)
        return xin, x1, x2, batch_complete

    # def build_demo_graph(self, config):
    #     incom_imgs = self.images*(1-self.masks)
    #     batch_data = tf.concat([incom_imgs,self.sketches,\
    #                     self.color,self.masks,self.noises],axis=3)
    #     gen_img, output_mask =  build_server_graph(FLAGS, batch_data,self.masks)
    #     self.demo_output= gen_img*self.masks + incom_imgs


    # def demo(self, config, batch):
    #     demo_output = self.sess.run(self.demo_output,
    #         feed_dict={
    #             self.images: batch[:,:,:,:3],
    #             self.sketches: batch[:,:,:,3:4],
    #             self.color: batch[:,:,:,4:7],
    #             self.masks: batch[:,:,:,7:8],
    #             self.noises: batch[:,:,:,8:9]})
    #     return demo_output
