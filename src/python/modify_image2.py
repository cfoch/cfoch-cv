import sys
import numpy as np
from matplotlib import pyplot as plt
import cv2

filename = "../resources/people.png"

image = cv2.imread(filename, cv2.IMREAD_COLOR)

if len(image) == 0:
    print("Image %s is empty" % filename)
    sys.exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()
