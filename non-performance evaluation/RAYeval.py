import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim

IMAGE_ORI_PATH = 'C:\\Users\\zkalf\\Desktop\\Result\\output.bmp'
IMAGE_COMP_PATH = 'C:\\Users\\zkalf\\Desktop\\Result\\output_mod.bmp'

imgOri = img_as_float(data.imread(IMAGE_ORI_PATH))
imgComp = img_as_float(data.imread(IMAGE_COMP_PATH))
rows, cols, _ = imgOri.shape

def mse(x, y):
    return np.linalg.norm(x - y)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                         sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

mseOri = mse(imgOri, imgOri)
ssimOri = ssim(imgOri, imgOri, data_range=imgOri.max() - imgOri.min(), multichannel=True)

mseComp = mse(imgOri, imgComp)
ssimComp = ssim(imgOri, imgComp, data_range=imgComp.max() - imgComp.min(), multichannel=True)

label = 'MSE: {:.4f}, SSIM: {:.4f}'

ax[0].imshow(imgOri, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[0].set_xlabel(label.format(mseOri, ssimOri))
ax[0].set_title('Original image')

ax[1].imshow(imgComp, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_xlabel(label.format(mseComp, ssimComp))
ax[1].set_title('Compare Image')

plt.tight_layout()
plt.show()
