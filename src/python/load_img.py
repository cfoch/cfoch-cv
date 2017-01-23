import cv2
import os
from matplotlib import pyplot as plt

img = cv2.imread("resources/background1.jpg", 0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])
plt.show()
