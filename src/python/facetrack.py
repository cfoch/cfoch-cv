import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("../data/haarcascade_frontalface_default.xml")

colors = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255)
)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        # flags=cv2.CV_HAAR_SCALE_IMAGE
    )
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors[i % 3], 2)
    cv2.imshow('Video', frame)


    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
