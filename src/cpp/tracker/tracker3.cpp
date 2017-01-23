#include <string>
#include <iostream>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>

#define PREDICTOR_PATH "data"
#define SKIP_N_FRAMES 0

using namespace dlib;
using namespace cv;
using namespace std;

static void
draw_landmarks (Mat & frame, std::vector<std::vector<cv::Point>> & landmarks,
    double ratio, const cv::Scalar & color)
{
  int i, j;
  for (i = 0; i < landmarks.size (); i++)
    for (j = 0; j < landmarks[i].size (); j++)
      putText (frame, to_string (j), landmarks[i][j],
          FONT_HERSHEY_SCRIPT_SIMPLEX, ratio, color);
}

int
main (int argc, char ** argv)
{
  VideoCapture cap(0);
  frontal_face_detector detector;
  std::vector<dlib::rectangle> dets;
  shape_predictor predictor;
  std::vector<std::vector<cv::Point>> landmarks, old_landmarks;
  Mat frame, old_gray, gray;
  bool detected = false;
  int frame_count, real_frame_count;

  if (!cap.isOpened ()) {
    cout << "Could not find webcam device." << endl;
    return 1;
  }

  namedWindow ("window", 1);
  deserialize (PREDICTOR_PATH) >> predictor;
  detector = get_frontal_face_detector ();

  for (frame_count = 1, real_frame_count = 1;; frame_count++, real_frame_count++) {
    int i, j;
    array2d<unsigned char> img;
    // const vector<rectangle> rects;

    cap >> frame;
    cvtColor (frame, gray, CV_BGR2GRAY);


    if (!detected) {
      assign_image (img, cv_image<rgb_pixel> (frame));


      dets = detector (img);

      detected = dets.size() > 0;

      if (detected) {
        cout << "Faces were detected at frame number: " << real_frame_count <<
            endl;

        // Remember landmarks.
        for (i = 0; i < dets.size(); i++) {
          std::vector<cv::Point> landmark;

          full_object_detection shape = predictor (img, dets[i]);
          for (j = 0; j < shape.num_parts (); j++) {
            cv::Point pt;
            pt = cv::Point (shape.part(j).x(), shape.part(j).y());
            landmark.push_back (pt);
          }
          landmarks.push_back (landmark);
        }
        old_landmarks = landmarks;
        draw_landmarks (frame, landmarks, 0.5, cv::Scalar (0, 255, 0));
        gray.copyTo (old_gray);
      }
    } else {
      cout << "Frame number: " << real_frame_count << endl;

      draw_landmarks (frame, landmarks, 0.5, cv::Scalar (0, 255, 0));

      if (frame_count % (SKIP_N_FRAMES + 1) == 0) {
        std::vector<std::vector<cv::Point>> new_landmarks;

        // TODO: DUmmy. Delete me.
        imshow ("window_current_gray", gray);
        imshow ("window_old_gray", old_gray);

        cout << "Frame number (since detected): " << frame_count << endl;

        // Tracking.
        for (i = 0; i < old_landmarks.size (); i++) {
          std::vector<cv::Point> new_landmark;
          std::vector<cv::Point2f> features_prev, features;
          std::vector<uchar> status;
          std::vector<float> err;

          for (j = 0; j < old_landmarks[i].size (); j++) {
            cv::Point pt;

            pt = old_landmarks[i][j];
            features_prev.push_back (pt);
          }
          calcOpticalFlowPyrLK (old_gray, gray, features_prev, features,
              status, err);

          // Draw predicted points.
          // cout << "Previous Features size: " << features_prev.size () << endl;
          // cout << "Features size: " << features.size () << endl;
          for (j = 0; j < features.size (); j++) {
            cv::Point pt;

            if (status[j] == 0) {
              cout << "Status is 0." << endl;
              continue;
            }
            // if (err[j] == 0)
            //   cout << "Error is 0." << endl;

            pt = features[j];
            new_landmark.push_back (pt);
          }
          new_landmarks.push_back (new_landmark);
        }
        draw_landmarks (frame, new_landmarks, 0.4, cv::Scalar (0, 0, 255));
        old_landmarks = new_landmarks;

        gray.copyTo (old_gray);
        
      }
    }

    imshow ("window_frame", frame);

    if (waitKey (30) >= 0)
      break;
  }

  return 0;
}
