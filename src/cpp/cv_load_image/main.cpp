#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int
main (int argc, char ** argv)
{
  Mat image;
  image = imread (argv[1], CV_LOAD_IMAGE_COLOR);

  if (!image.data) {
    cout << "Could not open image file." << endl;
    return 1;
  }

  namedWindow ("Display Window", WINDOW_AUTOSIZE);
  imshow ("Display Window", image);

  waitKey (0);

  return 0;
}
