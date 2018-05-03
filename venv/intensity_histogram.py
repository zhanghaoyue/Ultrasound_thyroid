import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import data, segmentation, color
from skimage.future import graph

path = '/home/zhanghaoyue/Documents/ultrasound_images/ML_PILOT_BATCH_2'
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
center = cv2.KMEANS_RANDOM_CENTERS


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.PNG')]


imlist = get_imlist(path)


def get_img_intensity(input_image):
    img = cv2.imread(input_image, 0)
    histg = cv2.calcHist([img], [0], None, [256], [0, 256])
    return histg


'''
for file in imlist:
    img = cv2.imread(file)
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    compactness, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, center)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    res2 = res.reshape(img.shape)
    print(file)
    plt.imshow(res2, interpolation='nearest')
    plt.show()
'''

img = cv2.imread(imlist[0])
labels1 = segmentation.slic(img, compactness=30, n_segments=400)
out1 = color.label2rgb(labels1, img, kind='avg')

g = graph.rag_mean_color(img, labels1, mode='similarity')
labels2 = graph.cut_normalized(labels1, g)
out2 = color.label2rgb(labels2, img, kind='avg')

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))



ax[0].imshow(out1)
ax[1].imshow(out2)

for a in ax:
    a.axis('off')


plt.tight_layout()
plt.show()