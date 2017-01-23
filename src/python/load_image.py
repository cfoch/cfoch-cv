import sys
import cv2

filename = "../resources/people.png"

image = cv2.imread(filename, cv2.IMREAD_COLOR)

if len(image) == 0:
    print("Image %s is empty" % filename)
    sys.exit()

cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Display Window", image)
cv2.waitKey(0)
