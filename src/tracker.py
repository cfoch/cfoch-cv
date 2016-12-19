import sys
import numpy as np
import cv2


class Tracker:
    CASCADE_FILENAME = 'data/haarcascade_frontalface_default.xml'
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.tracking = False
        self.ROIs = []

    def updateROIs(self, frame):
        face_cascade = cv2.CascadeClassifier(self.CASCADE_FILENAME)
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1,
            minNeighbors=5, minSize=(120,120))
        for (x, y, w, h) in faces:
           self.ROIs.append((x, y, w, h))
        self.tracking = bool(self.ROIs)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading from webcam")
                sys.exit()

            if not self.tracking:
                self.updateROIs(frame)
            else:
                for x, y, w, h in self.ROIs:
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            cv2.imshow("CAM", frame)
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

tracker = Tracker()
tracker.run()
