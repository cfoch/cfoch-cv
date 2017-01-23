#include "tracker8.hpp"

int
main (int argc, char ** argv)
{
  if (argc == 2) {
    FaceTracker tracker (argv[1]);
    tracker.start_tracking ();
  } else {
    FaceTracker tracker;
    tracker.start_tracking ();
  }
  
  return 0;
}
