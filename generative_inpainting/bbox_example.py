import numpy as np
import neuralgym as ng
import cv2

def random_bbox(FLAGS):
    """Generate a random tlhw.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = FLAGS.img_shapes
    img_height = img_shape[0]*2
    img_width = img_shape[1]*2
    maxt = img_height - FLAGS.vertical_margin - FLAGS.height  # 256 - 0 - 128
    maxl = img_width - FLAGS.horizontal_margin - FLAGS.width
    t = np.uint16(np.random.uniform(
        low=FLAGS.vertical_margin, high=maxt))
    l = np.uint16(np.random.uniform(
        low=FLAGS.horizontal_margin, high=maxl))
    h = FLAGS.height
    w = FLAGS.width
    return t, l, h, w


def npmask(bbox, height, width, delta_h, delta_w):
    mask = np.ones((height, width), np.float32) * 0.5
    h = np.random.randint(delta_h//2+1)
    w = np.random.randint(delta_w//2+1)
    mask[bbox[0]+h:bbox[0]+bbox[2]-h,
         bbox[1]+w:bbox[1]+bbox[3]-w] = 1.
    return mask


if __name__ == '__main__':
    FLAGS = ng.Config('inpaint_CTimg.yml')
    img_shape = FLAGS.img_shapes
    height = img_shape[0]*2
    width = img_shape[1]*2

    for i in range(100):
        bbox = random_bbox(FLAGS)
        img = npmask(bbox, height, width, FLAGS.max_delta_height, FLAGS.max_delta_width)
        # cv2.imshow('bbox', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        cv2.imwrite('bbox_mask/' + str(i) + '.jpg', img * 255)
