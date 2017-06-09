import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import img_as_float
from scipy.ndimage import distance_transform_edt

IMG_RAW_PATH = 'C:\\Users\\zkalf\\Desktop\\Result\\image.pgm'
IMG_ORI_PATH = 'C:\\Users\\zkalf\\Desktop\\Result\\image_out(srad1).pgm'
IMG_COMP_PATH = 'C:\\Users\\zkalf\\Desktop\\Result\\image_out_mod(srad1).pgm'


DEFAULT_ALPHA = 1.0 / 9

def fom(edges_img, edges_gold, alpha=DEFAULT_ALPHA):
    dist = distance_transform_edt(np.invert(edges_gold))
    fom = 1.0 / np.maximum(np.count_nonzero(edges_img), np.count_nonzero(edges_gold))
    N, M = edges_img.shape
    for i in range(0, N):
        for j in range(0, M):
            if edges_img[i, j]:
                fom += 1.0 / (1.0 + dist[i, j] * dist[i, j] * alpha)
    fom /= np.maximum(
        np.count_nonzero(edges_img),
        np.count_nonzero(edges_gold))
    return fom

# Original image
imgRaw = cv2.imread(IMG_RAW_PATH)
mean = np.mean(imgRaw)
canny_edgeRaw = cv2.Canny(imgRaw, 0.66 * mean, 1.33 * mean)

# first image
imgOri = cv2.imread(IMG_ORI_PATH)
canny_edgeOri = cv2.Canny(imgOri, 10, 25)

# second image
imgComp = cv2.imread(IMG_COMP_PATH)
mean = np.mean(imgComp)
canny_edgeComp = cv2.Canny(imgComp, 10, 25)

fig, axes = plt.subplots(nrows=2, ncols=3,
                         sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
axes[0, 0].imshow(img_as_float(imgRaw), cmap=plt.cm.gray, vmin=0, vmax=1)
axes[0, 0].axis('off')
axes[1, 0].imshow(img_as_float(canny_edgeRaw), cmap=plt.cm.gray, vmin=0, vmax=1)
axes[1, 0].axis('off')
axes[0, 0].set_title('Raw Image')

axes[0, 1].imshow(img_as_float(imgOri), cmap=plt.cm.gray, vmin=0, vmax=1)
axes[0, 1].axis('off')
axes[1, 1].imshow(img_as_float(canny_edgeOri), cmap=plt.cm.gray, vmin=0, vmax=1)
axes[1, 1].axis('off')
axes[0, 1].set_title('Ori (PFOM=' + "{:0.3f}".format(fom(canny_edgeOri, canny_edgeRaw, alpha=DEFAULT_ALPHA)) + ')')

axes[0, 2].imshow(img_as_float(imgComp), cmap=plt.cm.gray, vmin=0, vmax=1)
axes[0, 2].axis('off')
axes[1, 2].imshow(img_as_float(canny_edgeComp), cmap=plt.cm.gray, vmin=0, vmax=1)
axes[1, 2].axis('off')
axes[0, 2].set_title('Comp (PFOM=' + "{:0.3f}".format(fom(canny_edgeComp, canny_edgeRaw, alpha=DEFAULT_ALPHA)) + ')')

plt.tight_layout()
plt.show()


"""
def nothing(x):
    pass
    
cv2.createTrackbar('min_value', 'canny_edge', 0, 300, nothing)
cv2.createTrackbar('max_value', 'canny_edge', 0, 500, nothing)

while True:
    cv2.imshow('image', imgOri)
    cv2.imshow('canny_edge', canny_edgeOri)

    min_value = cv2.getTrackbarPos('min_value', 'canny_edge')
    max_value = cv2.getTrackbarPos('max_value', 'canny_edge')

    canny_edgeOri = cv2.Canny(imgOri, min_value, max_value)

    k = cv2.waitKey(37)
    if k == 27:
        break

"""