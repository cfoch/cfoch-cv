import sys
import os
import dlib
import glob
from skimage import io

print(dlib)

predictor_path = "/home/fabian/Documents/git/cfoch-cv/src/resources/data"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
window = dlib.image_window()

print("hey")
img = io.imread("/home/fabian/Documents/git/cfoch-cv/src/resources/trump.jpg")
print(img)
window.clear_overlay()
window.set_image(img)

dets = detector(img, 1)
for k, d in enumerate(dets):
    print(k, d.left(), d.top(), d.right, d.bottom)
    shape = predictor(img, d)
    window.add_overlay(shape)
window.add_overlay(dets)
dlib.hit_enter_to_continue()
