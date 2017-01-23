#include <string>
#include <iostream>
#include <algorithm>

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

static const int MAX_CONTOUR_POINTS = 27;
static const int CONTOUR_INDICES[MAX_CONTOUR_POINTS] = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  26, 25, 24, 23, 22,
  21, 20, 19, 18, 17};

/*
struct {
  bool operator
  () (cv::Point pt1, cv::Point pt2)
  {
    return pt1.x < pt2.x;
  }
} sort_by_x;

struct {
  bool operator
  () (cv::Point pt1, cv::Point pt2)
  {
    return pt1.y < pt2.y;
  }
} sort_by_y;
*/

/*

bool
comparator_x (const cv::Point & pt1, const cv::Point & pt2)
{
  return pt1.x () < pt2.x ();
}

bool
comparator_y (const cv::Point & pt1, const cv::Point & pt2)
{
  return pt1.y () < pt2.y ();
}
*/

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

/*

static void
circumbscribed_rectangle (std::vector<cv::Point> pts, Point & pt1, Point & pt2)
{
  Point px_min, px_max, py_min, py_max;
  std::vector pts_copy (pts);
  px_min = std::min_element (pts_copy.begin (), pts_copy.end (), comparator_x);
  px_max = std::max_element (pts_copy.begin (), pts_copy.end (), comparator_x);
  py_min = std::min_element (pts_copy.begin (), pts_copy.end (), comparator_y);
  py_max = std::max_element (pts_copy.begin (), pts_copy.end (), comparator_y);

  pt1 = cv::Point (px_min.x (), py_max.y ());
  pt2 = cv::Point (px_max.x (), py_min.y ());
}

*/


int
main (int argc, char ** argv)
{
  VideoCapture cap(0);
  frontal_face_detector detector;
  std::vector<dlib::rectangle> dets;
  shape_predictor predictor;
  std::vector<std::vector<cv::Point>> old_landmarks;
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

  for (frame_count = 1, real_frame_count = 1; ; frame_count++, real_frame_count++) {
    cv::Size sz;
    std::vector<Mat> masks;
    int i, j;
    std::vector<std::vector<cv::Point>> new_landmarks;
    array2d<unsigned char> img;
    // const vector<rectangle> rects;

    cap >> frame;
    sz = frame.size ();
    cvtColor (frame, gray, CV_BGR2GRAY);



    assign_image (img, cv_image<rgb_pixel> (frame));


    dets = detector (img);

    detected = dets.size() > 0;

    if (detected) {
      cout << "Faces were detected at frame number: " << real_frame_count <<
          endl;

      // Remember landmarks.
      for (i = 0; i < dets.size(); i++) {
        Mat mask;
        cv::Point contour_pts[MAX_CONTOUR_POINTS];
        std::vector<cv::Point> new_landmark;

        full_object_detection shape = predictor (img, dets[i]);
        for (j = 0; j < shape.num_parts (); j++) {
          cv::Point pt;
          pt = cv::Point (shape.part(j).x(), shape.part(j).y());
          new_landmark.push_back (pt);
        }
        new_landmarks.push_back (new_landmark);

        // Init masks with black background.
        mask = Mat::zeros (frame.size (), CV_8UC1);
        masks.push_back (mask);
        // Create the polygon closing the contour of the face.
        for (j = 0; j < MAX_CONTOUR_POINTS; j++)
          contour_pts[j] = new_landmark[CONTOUR_INDICES[j]];

        const Point *contours_pts[1] = { contour_pts };

        if (i == 0)
          fillPoly (masks[i], contours_pts, &MAX_CONTOUR_POINTS, 1,
              cv::Scalar (255, 255, 255));
      }

      Mat masked;
      // cv::Point rectangle_pt1, rectangle_pt1;

      // circumbscribed_rectangle (old_landmarks, rectangle_pt1, rectangle_pt2);

      gray.copyTo (masked, masks[0]);

      imshow ("mask", masks[0]);
      imshow ("masked", masked);
      draw_landmarks (frame, new_landmarks, 0.5, cv::Scalar (0, 255, 0));
      old_landmarks = new_landmarks;
    }


    imshow ("window", frame);

    if (waitKey (30) >= 0)
      break;
  }

  return 0;
}
