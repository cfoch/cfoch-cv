import sys
import numpy as np
import cv2
from IPython import embed


class Tracker:
    CASCADE_FILENAME = 'data/haarcascade_frontalface_default.xml'
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.tracking = False
        self.ROIs_pts = []
        # self.ROIs_hsv = []
        self.ROIs_hist = []
        self.termination =\
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def updateROIs(self, frame):
        face_cascade = cv2.CascadeClassifier(self.CASCADE_FILENAME)
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1,
            minNeighbors=5, minSize=(120,120))
        for (x, y, w, h) in faces:
            roi = frame[y : y + h, x : x + w]
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi_hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
            roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            self.ROIs_hist.append(roi_hist)
            self.ROIs_pts.append((x, y, w, h))
            # self.ROIs.append((x, y, w, h))
        self.tracking = bool(self.ROIs_pts)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading from webcam")
                sys.exit()

            if not self.tracking:
                self.updateROIs(frame)
            else:
                for i in range(len(self.ROIs_pts)):
                    roi_pts = self.ROIs_pts[i]
                    roi_hist = self.ROIs_hist[i]
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    back_proj = cv2.calcBackProject([hsv], [0], roi_hist,
                        [0, 180], 1)

                    print(self.ROIs_pts)
                    ret, self.ROIs_pts[i] = cv2.CamShift(back_proj,
                        self.ROIs_pts[i], self.termination)
                    print(self.ROIs_pts)

                    pts = cv2.boxPoints(ret)
                    pts = np.int0(pts)
                    
                    box = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            cv2.imshow("CAM", frame)
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

tracker = Tracker()
tracker.run()
