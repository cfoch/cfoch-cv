#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>

using namespace dlib;
using namespace std;

int
main (int argc, char ** argv)
{
  array2d<rgb_pixel> img;
  frontal_face_detector detector;
  std::vector<rectangle> dets;
  image_window win;


  detector = get_frontal_face_detector ();
  load_image (img, argv[1]);

  // Make the image twice bigger.
  pyramid_up (img);
  dets = detector (img);

  win.clear_overlay ();
  win.set_image (img);
  win.add_overlay (dets, rgb_pixel (255, 0, 0));

  cout << "Press enter to exit." << endl;
  cin.get();

  return 0;
}
