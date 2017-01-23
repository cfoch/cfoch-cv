#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int
main (int argc, char ** argv)
{
  VideoCapture cap(0);

  if (!cap.isOpened ()) {
    cout << "Could not find webcam device." << endl;
    return 1;
  }

  namedWindow ("edges", 1);
  for (;;) {
    Mat frame;
    cap >> frame;
    // cvtColor (frame, edges, COLOR_BGR2GRAY);
    imshow ("edges", frame);

    if (waitKey (30) >= 0)
      break;
  }

  return 0;
}
