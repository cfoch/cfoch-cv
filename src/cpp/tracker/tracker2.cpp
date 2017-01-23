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

#define DEFAULT_VIDEO_PATH "conferencia.avi"
#define PREDICTOR_PATH "data"
#define MAX_COLORS 20

using namespace dlib;
using namespace cv;
using namespace std;



void
set_random_colors (cv::Scalar * colors, int n)
{
  RNG rng(12345);
  int i;
  for (i = 0; i < MAX_COLORS; i++)
    colors[i] = cv::Scalar (rng.uniform (0, 255), rng.uniform (0, 255),
        rng.uniform (0, 255));
}

int
main (int argc, char ** argv)
{

  VideoCapture cap;
  frontal_face_detector detector;
  std::vector<dlib::rectangle> dets;
  shape_predictor predictor;
  cv::Scalar colors[MAX_COLORS];

  if (argc == 2)
    cap = VideoCapture (argv[1]);
  else
    cap = VideoCapture (0);

  if (!cap.isOpened ()) {
    cout << "Could not find webcam device." << endl;
    return 1;
  }

  namedWindow ("window", 1);
  deserialize (PREDICTOR_PATH) >> predictor;
  detector = get_frontal_face_detector ();

  
  set_random_colors (colors, MAX_COLORS);

  for (;;) {
    int i, j;
    Mat frame;
    array2d<unsigned char> img;
    // const vector<rectangle> rects;

    cap >> frame;

    assign_image (img, cv_image<rgb_pixel> (frame));


    dets = detector (img);

    // frame = dlib::toMat (img);

    for (i = 0; i < dets.size(); i++) {
      cv::Scalar landmark_color;

      landmark_color = colors[i % MAX_COLORS];
      full_object_detection shape = predictor (img, dets[i]);


      for (j = 0; j < shape.num_parts (); j++) {
        cv::Point pt;

        pt = cv::Point (shape.part(j).x(), shape.part(j).y());

        putText (frame, to_string (j), pt, FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5,
            landmark_color);
      }
    }

    imshow ("window", frame);

    if (waitKey (30) >= 0)
      break;
  }

  return 0;
}
