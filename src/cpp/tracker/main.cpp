#include <iostream>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <dlib/opencv.h>

#include <opencv2/opencv.hpp>

using namespace dlib;
using namespace cv;
using namespace std;

int
main (int argc, char ** argv)
{
  VideoCapture cap(0);
  frontal_face_detector detector;
  std::vector<dlib::rectangle> dets;
  image_window win;

  if (!cap.isOpened ()) {
    cout << "Could not find webcam device." << endl;
    return 1;
  }

  namedWindow ("window", 1);

  detector = get_frontal_face_detector ();

  for (;;) {
    int i;
    Mat frame;
    array2d<unsigned char> img;
    // const vector<rectangle> rects;

    cap >> frame;

    assign_image (img, cv_image<rgb_pixel> (frame));


    dets = detector (img);

    // frame = dlib::toMat (img);

    for (i = 0; i < dets.size(); i++) {
      cv::Point pt1 (dets[i].left(), dets[i].top());
      cv::Point pt2 (dets[i].right(), dets[i].bottom());

      cv::rectangle (frame, pt1, pt2, cv::Scalar (0, 255, 0));
    }

    imshow ("window", frame);

    if (waitKey (30) >= 0)
      break;
  }

  return 0;
}
