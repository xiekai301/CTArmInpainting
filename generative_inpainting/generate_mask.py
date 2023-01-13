import cv2
import matplotlib.pyplot as plt
import numpy as np

img = np.load('1001.npy')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (384, 384))
img[100:170, 100:170] = 4095
plt.imshow(img, vmin=500, vmax=1500)
plt.show()
np.save('1001_after.npy', img)

img_mask = cv2.imread('10_mask.png', cv2.IMREAD_UNCHANGED)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_mask = cv2.resize(img_mask, (384, 384))
img_mask = img_mask[:, :, 0]
img_mask[:, :] = 0
img_mask[100:170, 100:170] = 255
plt.imshow(img_mask)
plt.show()

np.save('1001_mask.npy', img_mask)

# img = cv2.imread('case1_mask.png', cv2.IMREAD_UNCHANGED)
# img = cv2.resize(img, (1024,1024))
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # img[:, :, :] = 0
# img[:, :, :] = 0
# img[400:500, 400:500, :] = 255
# plt.imshow(img)
# plt.show()
# cv2.imwrite('10_mask2.png', img)

# img = cv2.imread('case1_mask.png')
# img2 = cv2.imread('10_mask.png')
#
# img2_max = np.array(img2)
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()
