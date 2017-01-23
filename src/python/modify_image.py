import sys
import cv2

filename = "resources/people.png"

image = cv2.imread(filename, cv2.IMREAD_COLOR)

if len(image) == 0:
    print("Image %s is empty" % filename)
    sys.exit()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("out/people-gray.png", gray_image)



cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Gray", cv2.WINDOW_AUTOSIZE)

cv2.imshow("Original", image)
cv2.imshow("Gray", gray_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
