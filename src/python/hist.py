import cv2
import numpy as np
from matplotlib import pyplot as plt
from IPython import embed

img = cv2.imread('../data/face1.png', 0)
embed()
plt.hist(img.ravel(), 256, [0, 256])
plt.show()
