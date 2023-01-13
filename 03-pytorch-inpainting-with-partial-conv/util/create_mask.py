import numpy as np
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat

def random_bbox(FLAGS):
    """Generate a random tlhw.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = FLAGS.img_shapes
    img_height = img_shape[0]
    img_width = img_shape[1]
    maxt = img_height - FLAGS.vertical_margin - FLAGS.height  # 256 - 0 - 128
    maxl = img_width - FLAGS.horizontal_margin - FLAGS.width
    t = np.random.randint(FLAGS.vertical_margin, maxt)
    l = np.random.randint(FLAGS.horizontal_margin, maxl)
    h = FLAGS.height
    w = FLAGS.width
    return (t, l, h, w)


def generate_rect_mask(bbox, FLAGS):
    mask = np.zeros((512, 512), np.float32)
    delta_h = FLAGS.max_delta_height
    delta_w = FLAGS.max_delta_width
    h = np.random.randint(delta_h//2+1)
    w = np.random.randint(delta_w//2+1)
    mask[bbox[0]+h:bbox[0]+bbox[2]-h,
         bbox[1]+w:bbox[1]+bbox[3]-w] = 1.
    return mask


def generate_stroke_mask(FLAGS):
    H = W = 512
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    min_num_vertex = 1
    max_num_vertex = 3
    mean_angle = 2 * math.pi / 5
    angle_range = 2 * math.pi / 15
    min_width = 25
    max_width = 50

    for _ in range(np.random.randint(1, 4)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        # vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        vertex.append((int(np.random.randint(0+2*FLAGS.vertical_margin, w-2*FLAGS.horizontal_margin)),
                       int(np.random.randint(0+2*FLAGS.vertical_margin, h-2*FLAGS.horizontal_margin))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.float32)
    mask = np.reshape(mask, (H, W))
    return mask


def generate_mask():
    H = W = 512
    mask = np.zeros((512,512), np.float32)
    # randnum_L = np.random.uniform(0, 1)
    # if randnum_L < 0.5:
    #     randnum_vstart = int(np.random.uniform(0.45, 0.6) * H)
    #     randnum_vheight = int(np.random.uniform(0.05, 0.15) * H)
    #     randnum_hwidth = int(np.random.uniform(0.15, 0.25) * W)
    #     mask[randnum_vstart:randnum_vstart + randnum_vheight, :randnum_hwidth] = 1
    # else:
    #     randnum_vstart = int(np.random.uniform(0.4, 0.6) * H)
    #     randnum_vheight = int(np.random.uniform(0.1, 0.2) * H)
    #     randnum_hstart = int(np.random.uniform(0.05, 0.15) * W)
    #     randnum_hwidth = int(np.random.uniform(0.05, 0.15) * W)
    #     mask[randnum_vstart:randnum_vstart + randnum_vheight,
    #          randnum_hstart:randnum_hstart + randnum_hwidth] = 1
    #
    # randnum_R = np.random.uniform(0, 1)
    # if randnum_R < 0.5:
    #     randnum_vstart = int(np.random.uniform(0.45, 0.6) * H)
    #     randnum_vheight = int(np.random.uniform(0.05, 0.15) * H)
    #     randnum_hwidth = int(np.random.uniform(0.15, 0.25) * W)
    #     mask[randnum_vstart:randnum_vstart + randnum_vheight, W - randnum_hwidth:] = 1
    # else:
    #     randnum_vstart = int(np.random.uniform(0.4, 0.6) * H)
    #     randnum_vheight = int(np.random.uniform(0.1, 0.2) * H)
    #     randnum_hstart = int(np.random.uniform(0.05, 0.15) * W)
    #     randnum_hwidth = int(np.random.uniform(0.05, 0.15) * W)
    #     mask[randnum_vstart:randnum_vstart + randnum_vheight,
    #          W - (randnum_hstart + randnum_hwidth):H - randnum_hstart] = 1

    x, y = np.ogrid[:H, :W]
    cx, cy = H/2, W/2
    radius = int(np.random.uniform(0.7, 0.8) * H / 2)
    # radius = int(np.random.uniform(0.8, 0.8) * H / 2)
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    circmask = r2 >= radius * radius
    # circmask[:int(H / 4), :] = 0
    # circmask[int(H * 3 / 4):, :] = 0
    mask[circmask] = 1
    return mask


if __name__=='__main__':
    # FLAGS = Config('mask_parameters.yaml')
    # # FLAGS = Config('util/mask_parameters.yaml')
    # mat = loadmat('/home/czey/00-pytorch-CycleGAN-xk_MRMAR/datasets/MRMAR_pix2pix/test/CMR136491_1000.mat')
    mat = loadmat('/home/czey/generative_inpainting_MRMAR/training_data/MR_MAR/validation/CMR127608/1001.mat')
    img = np.sqrt(mat['img']) * 255

    mask = np.zeros((512, 512), np.float32)
    x, y = np.ogrid[:512, :512]
    cx, cy = 110, 310
    # cx, cy = 120, 210
    radius = 30
    # radius = int(np.random.uniform(0.8, 0.8) * H / 2)
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    circmask = r2 <= radius * radius
    # r1 = (x - 120) * (x - 120) + (y - 210) * (y - 210)
    # circmask1 = r1 <= radius * radius
    # circmask[:int(H / 4), :] = 0
    # circmask[int(H * 3 / 4):, :] = 0
    mask[circmask] = 1
    # mask[circmask1] = 1

    img_withmask = img + mask *255
    plt.imshow(img_withmask, cmap='gray')
    plt.show()

    mask_path = '/home/czey/00-pytorch-CycleGAN-xk_MRMAR/mask_circle2.png'
    cv2.imwrite(mask_path, mask * 255)

    # for i in range(1):
    #     bbox = random_bbox(FLAGS)  # 生成mask矩阵的左上角坐标和高宽
    #     regular_mask = generate_rect_mask(bbox, FLAGS, )  # 矩形mask, 在上一句基础上再随机內缩，最大32/2
    #     irregular_mask = generate_stroke_mask(FLAGS)  # 类似笔刷, 设置随机角, 再用椭圆平滑
    #
    #     hand_mask = (regular_mask + irregular_mask).astype(np.bool) * 255
    #     img_withmask = img + hand_mask
    #     plt.imshow(img_withmask, cmap='gray')
    #     plt.show()
    #     mask_path = os.path.join('mask', str(i)+'.png')
        # cv2.imwrite(mask_path, hand_mask)
    # plt.imshow(hand_mask, cmap='gray')
    # plt.show()

