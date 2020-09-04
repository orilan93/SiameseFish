from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from config import IMG_SIZE
from data import resize_padding

image_path = r"../data/dataset_condensed/cropped_head/420208_1.jpg"

pil_img = Image.open(image_path)
pil_img = resize_padding(pil_img, IMG_SIZE)

img = np.array(pil_img)

saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(img)

saliencyMap_u8 = (saliencyMap * 255).astype('uint8')
threshMap = cv2.threshold(saliencyMap_u8, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

ret, thresh1 = cv2.threshold(saliencyMap_u8, 32, 255, cv2.THRESH_BINARY)

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0][0].imshow(img)
ax[0][1].imshow(saliencyMap)
ax[1][0].imshow(saliencyMap_u8)
ax[1][1].imshow(thresh1)

plt.show()