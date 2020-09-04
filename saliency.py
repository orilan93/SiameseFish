"""
Extracts the pattern into a binary image.
"""

import cv2

image_path = r"../data/dataset_condensed/cropped_head/420208_1.jpg"
image = cv2.imread(image_path)

saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(image)

saliencyMap_u8 = (saliencyMap * 255).astype('uint8')
threshMap = cv2.threshold(saliencyMap_u8, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#threshMap = cv2.threshold(saliencyMap.astype("uint8"), 32, 255, cv2.THRESH_BINARY_INV)[1]
th3 = cv2.adaptiveThreshold(saliencyMap_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 2)

ret,thresh1 = cv2.threshold(saliencyMap_u8,32,255,cv2.THRESH_BINARY)
cv2.imshow("Thresh1", thresh1)

cv2.imshow("Image", image)
cv2.imshow("Output", saliencyMap)
cv2.imshow("Thresh", threshMap)
cv2.waitKey(0)
