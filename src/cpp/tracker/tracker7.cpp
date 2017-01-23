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
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/video/tracking.hpp"

#define DEFAULT_VIDEO_PATH "conferencia.avi"
#define PREDICTOR_PATH "data"
#define SKIP_N_FRAMES 0
#define MAX_COLORS 20

using namespace dlib;
using namespace cv;
using namespace std;

// Histogram.
static const int N_HIST_CHANNELS = 1;

static const float HIST_H_RANGE[] = { 0, 180 };
static const int HIST_H_BINS = 180;

static const int HIST_CHANNELS[] = { 0 };
static const float *HIST_RANGES[] = { HIST_H_RANGE };
static const int HIST_SIZE[] = { HIST_H_BINS };


// Contours.
static const int MAX_CONTOUR_POINTS = 27;
static const int CONTOUR_INDICES[MAX_CONTOUR_POINTS] = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  26, 25, 24, 23, 22,
  21, 20, 19, 18, 17};


void
set_random_colors (cv::Scalar * colors, int n)
{
  RNG rng(12345);
  int i;
  for (i = 0; i < MAX_COLORS; i++)
    colors[i] = cv::Scalar (rng.uniform (0, 255), rng.uniform (0, 255),
        rng.uniform (0, 255));
}

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

static void
draw_landmarks (Mat & frame, std::vector<std::vector<cv::Point>> & landmarks,
    double ratio, std::vector<cv::Scalar> & colors)
{
  int i, j;
  for (i = 0; i < landmarks.size (); i++)
    for (j = 0; j < landmarks[i].size (); j++)
      putText (frame, to_string (j), landmarks[i][j],
          FONT_HERSHEY_SCRIPT_SIMPLEX, ratio, colors[i]);
}

static void
draw_track_windows (Mat & frame, std::vector<cv::Rect> & track_windows,
    std::vector<cv::Scalar> & colors)
{
  int i, j;
  for (i = 0; i < track_windows.size (); i++)
    cv::rectangle (frame, track_windows[i], colors[i]);
}

bool
comparator_x (const cv::Point & pt1, const cv::Point & pt2)
{
  return pt1.x < pt2.x;
}

bool
comparator_y (const cv::Point & pt1, const cv::Point & pt2)
{
  return pt1.y < pt2.y;
}

cv::Rect
get_circumbscribed_rectangle (std::vector<cv::Point> pts)
{
  Point px_min, px_max, py_min, py_max, pt1, pt2;
  std::vector<cv::Point> pts_copy (pts);

  px_min = *std::min_element (pts_copy.begin (), pts_copy.end (), comparator_x);
  px_max = *std::max_element (pts_copy.begin (), pts_copy.end (), comparator_x);
  py_min = *std::min_element (pts_copy.begin (), pts_copy.end (), comparator_y);
  py_max = *std::max_element (pts_copy.begin (), pts_copy.end (), comparator_y);

  pt1 = cv::Point (px_min.x, py_max.y);
  pt2 = cv::Point (px_max.x, py_min.y);

  return Rect (pt1, pt2);
}

std::vector<cv::Point>
get_centroids (std::vector<std::vector<cv::Point>> & polygons)
{
  int i, j, sum_x, sum_y, pt_x, pt_y;
  std::vector<cv::Point> points;

  for (i = 0; i < polygons.size (); i++) {
    sum_x = sum_y = 0;
    for (j = 0; j < polygons[i].size (); j++) {
      sum_x += polygons[i][j].x;
      sum_y += polygons[i][j].y;
    }
    pt_x = sum_x / polygons[i].size();
    pt_y = sum_y / polygons[i].size();
    points.push_back (cv::Point (pt_x, pt_y));
  }
  return points;
}

std::vector<cv::Point>
get_centroids (std::vector<cv::Rect> & rects)
{
  int i;
  std::vector<cv::Point> points;

  for (i = 0; i < rects.size (); i++) {
    Point centroid;
    centroid = (rects[i].tl () + rects[i].br ()) * 0.5;
    points.push_back (cv::Point (centroid));
  }
  return points;
}

std::vector<std::vector<cv::Point>>
contour_from_landmark (std::vector<std::vector<cv::Point>> & landmarks)
{
  int i, j;
  std::vector<std::vector<cv::Point>> contours;

  // Create the polygon closing the contour of the face.
  for (i = 0; i < landmarks.size (); i++) {
    std::vector<cv::Point> contour;
    for (j = 0; j < MAX_CONTOUR_POINTS; j++)
      contour.push_back (landmarks[i][CONTOUR_INDICES[j]]);
    contours.push_back (contour);
  }
  return contours;
}


int
_have_to_ignore (int i_track_window, int i_landmark, int ignore[][2], int len)
{
  int i;
  for (i = 0; i < len; i++)
    if (ignore[i][0] == i_track_window || ignore[i][1] == i_landmark)
      return 1;
  return 0;
}

void
_find_minimum_pair (double ** distance_matrix, int n_track_windows,
    int n_landmarks, int ignore[][2], int n_ignore)
{
  int i_track_window, i_landmark, i_track_window_min, i_landmark_min;
  double minimum_distance = 100000;

  i_track_window_min = i_landmark_min = -1;
  for (i_track_window = 0; i_track_window < n_track_windows; i_track_window++) {
    for (i_landmark = 0; i_landmark < n_landmarks; i_landmark++) {
      if (!_have_to_ignore (i_track_window, i_landmark, ignore, n_ignore)) {
        double distance;
        distance = distance_matrix[i_track_window][i_landmark];
        if (distance < minimum_distance) {
          minimum_distance = distance;
          i_track_window_min = i_track_window;
          i_landmark_min = i_landmark;
        }
      }
    }
  }

  if (i_track_window_min != -1 && i_landmark_min != -1) {
    ignore[n_ignore][0] = i_track_window_min;
    ignore[n_ignore][1] = i_landmark_min;
  }
}

int
_ignore_comparator (const void * a, const void * b)
{
  // Sort by the first element [0].
  return *(int *) a - *(int *) b;
}


void
__print_matrix (double ** matrix, int n, int m)
{
  int i, j;
  cout << "-----------------------" << endl;
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      cout << setw (10) << matrix[i][j];
    }
    cout << endl;
  }
  cout << "-----------------------" << endl;
}

void
__print_matrix (int matrix[][2], int n, int m)
{
  int i, j;
  cout << "-----------------------" << endl;
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      cout << setw (10) << matrix[i][j];
    }
    cout << endl;
  }
  cout << "-----------------------" << endl;
}

void
__print_centroids (std::vector<cv::Point> & centroids)
{
  int i;
  for (i = 0; i < centroids.size (); i++)
    cout << i << ") " << centroids[i] << endl;
  cout << endl;
}

void
reorder_landmarks (std::vector<std::vector<cv::Point>> & new_landmarks,
    std::vector<cv::Rect> & track_windows)
{
  int n_landmarks = new_landmarks.size ();
  int n_track_windows = track_windows.size ();
  int n_ignore = min (n_landmarks, n_track_windows);

  double **distance_matrix;
  int ignore[n_ignore][2];

  int i, j, i_landmark, i_track_window, t, k;

  std::vector<std::vector<cv::Point>> landmark_contours;
  std::vector<cv::Point> landmark_centroids, track_windows_centroids;
  std::vector<std::vector<cv::Point>> tmp_landmarks (n_landmarks);

  

  landmark_contours = contour_from_landmark (new_landmarks);
  landmark_centroids = get_centroids (landmark_contours);
  track_windows_centroids = get_centroids (track_windows);

  // Init a matrix of distances.
  distance_matrix = new double * [n_track_windows];
  for (i_track_window = 0; i_track_window < n_track_windows; i_track_window++) {
    distance_matrix[i_track_window] = new double [n_landmarks];
    for (i_landmark = 0; i_landmark < n_landmarks; i_landmark++) {
      distance_matrix[i_track_window][i_landmark] =\
          cv::norm (track_windows_centroids[i_track_window] - landmark_centroids[i_landmark]);
    }
  }

  cout << "Landmark centroids: " << endl;
  __print_centroids (landmark_centroids);
  cout << "Track Window centroids: " << endl;
  __print_centroids (track_windows_centroids);

  __print_matrix (distance_matrix, n_track_windows, n_landmarks);




  // Populate ignore variable. "Ignore" are the closest pairs.
  for (t = 0; t < n_track_windows; t++)
    _find_minimum_pair (distance_matrix, n_ignore, n_landmarks, ignore, t);

  cout << "Pairs bef: " << endl;
  __print_matrix (ignore, n_ignore, 2);
  cout << " -o-o-o-o- " << endl;

  // FIXME
  // Reorder_new landmarks.
  for (i = 0; i < n_ignore; i++)
    tmp_landmarks[ignore[i][0]] = new_landmarks[ignore[i][1]];

  for (i = j = 0; i < n_landmarks; i++) {
    int found = 0;
    for (t = 0; t < n_ignore && !found; t++)
      if (ignore[t][1] == i)
        found = 1;
    if (!found)
      tmp_landmarks[k++] = new_landmarks[i];
  }
  new_landmarks = tmp_landmarks;

  landmark_contours = contour_from_landmark (new_landmarks);
  landmark_centroids = get_centroids (landmark_contours);
  cout << "New Landmark centroids: " << endl;
  __print_centroids (landmark_centroids);
}

void
line_landmarks2windows (Mat & frame,
    std::vector<std::vector<cv::Point>> & new_landmarks,
    std::vector<cv::Rect> & track_windows)
{
  int i;
  std::vector<std::vector<cv::Point>> landmark_contours;
  std::vector<cv::Point> landmark_centroids, track_windows_centroids;

  landmark_contours = contour_from_landmark (new_landmarks);
  landmark_centroids = get_centroids (landmark_contours);
  track_windows_centroids = get_centroids (track_windows);


  circle (frame, landmark_centroids[0], 1, cv::Scalar (0, 0, 255), 10);
  circle (frame, landmark_centroids[1], 1, cv::Scalar (0, 255, 0), 10);

  for (i = 0; i < track_windows_centroids.size (); i++)
  {
    line(frame, landmark_centroids[i], track_windows_centroids[i],
        cv::Scalar(0, 0, 255));
    // circle (frame, track_windows[i].tl (), 1, cv::Scalar (0, 255, 0), 10);
    //circle (frame, track_windows[i].br (), 1, cv::Scalar (0, 255, 0), 10);

    // cout << "tl: " << track_windows[i].tl () << " | br: " << track_windows[i].br () << endl;

    // centroid
    // circle (frame, track_windows_centroids[i], 1, cv::Scalar (0, 0, 255), 10);

    // cout << "centroid: " << track_windows_centroids[i] << endl;
  }
}


void
apply_camshift (Mat & frame, std::vector<cv::Rect> & track_windows,
    std::vector<MatND> & ROIs_hist, 
    std::vector<cv::RotatedRect> & rotated_rects)
{
  int i;
  // Apply camshfit.
  for (i = 0; i < track_windows.size (); i++) {
    RotatedRect rotated_rect;
    MatND back_projection;
    Mat hsv;
    cvtColor (frame, hsv, CV_BGR2HSV);
    calcBackProject (&hsv, 1, HIST_CHANNELS, ROIs_hist[i], back_projection,
         HIST_RANGES, 1, true);
    rotated_rect = CamShift (back_projection, track_windows[i],
        TermCriteria (TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
    rotated_rects.push_back (rotated_rect);

    // imshow ("back_projection-" + to_string (i), back_projection);
  }
}

int
main (int argc, char ** argv)
{
  VideoCapture cap;
  frontal_face_detector detector;
  std::vector<dlib::rectangle> dets;
  shape_predictor predictor;

  std::vector<std::vector<cv::Point>> old_landmarks;
  std::vector<Mat> ROIs;
  std::vector<MatND> ROIs_hist;

  std::vector<cv::Rect> track_windows;
  std::vector<cv::RotatedRect> rotated_rects;

  Mat frame, old_gray, gray;


  bool detected = false;
  int frame_count, real_frame_count;

  cv::Scalar available_colors[MAX_COLORS];
  std::vector<cv::Scalar> colors;

  if (argc == 2)
    cap = VideoCapture (0);
  else
    cap = VideoCapture (DEFAULT_VIDEO_PATH);

  if (!cap.isOpened ()) {
    cout << "Could not find webcam device." << endl;
    return 1;
  }

  namedWindow ("window", 1);
  deserialize (PREDICTOR_PATH) >> predictor;
  detector = get_frontal_face_detector ();


  set_random_colors (available_colors, MAX_COLORS);

  for (frame_count = 1, real_frame_count = 1; ; frame_count++, real_frame_count++) {
    cv::Size sz;
    std::vector<Mat> masks, maskeds;
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

    if (detected && old_landmarks.empty ()) {
      cout << "Faces were detected at frame number: " << real_frame_count <<
          endl;

      // For each detected face.
      for (i = 0; i < dets.size(); i++) {
        MatND ROI_hist;
        Mat mask, masked, ROI, ROI_hsv, ROI_mask;
        Rect circumbscribed_rectangle;
        cv::Point contour_pts[MAX_CONTOUR_POINTS];
        std::vector<cv::Point> contour_pts_vector;
        std::vector<cv::Point> new_landmark;

        // TODO. This should go to a function.
        // Generate landmarks.
        full_object_detection shape = predictor (img, dets[i]);
        for (j = 0; j < shape.num_parts (); j++) {
          cv::Point pt;
          pt = cv::Point (shape.part(j).x(), shape.part(j).y());
          new_landmark.push_back (pt);
        }
        new_landmarks.push_back (new_landmark);

        // Assign a random color.
        colors.push_back (available_colors[i % MAX_COLORS]);

        // TODO: Move to a function.
        // Create the polygon closing the contour of the face.
        for (j = 0; j < MAX_CONTOUR_POINTS; j++) {
          contour_pts[j] = new_landmark[CONTOUR_INDICES[j]];
          contour_pts_vector.push_back (contour_pts[j]);
        }

        // Calculate circumbscribed rectangle
        // circumbscribed_rectangle =\
        //     get_circumbscribed_rectangle (contour_pts_vector);
        circumbscribed_rectangle = boundingRect (contour_pts_vector);
        circumbscribed_rectangle = circumbscribed_rectangle &
            Rect (0, 0, frame.cols, frame.rows);
        track_windows.push_back (circumbscribed_rectangle);

        // Init masks with black background.
        mask = Mat::zeros (frame.size (), CV_8UC1);
        masks.push_back (mask);

        const Point *contours_pts[1] = { contour_pts };

        fillPoly (mask, contours_pts, &MAX_CONTOUR_POINTS, 1,
                      cv::Scalar (255, 255, 255));

        gray.copyTo (masked, mask);
        maskeds.push_back (masked);

        // Assign ROI.
        masked (circumbscribed_rectangle).copyTo (ROI);
        ROIs.push_back (ROI);

        // Generate ROI hist.
        cvtColor (frame, ROI_hsv, CV_BGR2HSV);
        ROI_mask = Mat ();
        calcHist (&ROI_hsv, 1, HIST_CHANNELS, mask, ROI_hist,
            N_HIST_CHANNELS, HIST_SIZE, HIST_RANGES, true, false);

        normalize (ROI_hist, ROI_hist, 0, 255, NORM_MINMAX);
        ROIs_hist.push_back (ROI_hist);


        // Remember the past landmark.
        old_landmarks = new_landmarks;
      }
      draw_track_windows (frame,track_windows, colors);
      draw_landmarks (frame, new_landmarks, 0.5, colors);

      // imshow ("ROI", ROIs[0]);
      // imshow ("masked", maskeds[0]);
      // imshow ("mask", masks[0]);
      // imshow ("gray", gray);

      putText (frame, "DETECTED", cv::Point (50, 50),
          FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar (0, 255, 0), 10);
    } else if (detected && !old_landmarks.empty ()) {

      for (i = 0; i < dets.size(); i++) {
        std::vector<cv::Point> new_landmark;
        full_object_detection shape = predictor (img, dets[i]);


        // TODO. This should go to a function.
        // Generate landmarks.
        for (j = 0; j < shape.num_parts (); j++) {
          cv::Point pt;
          pt = cv::Point (shape.part(j).x(), shape.part(j).y());
          new_landmark.push_back (pt);
        }
        new_landmarks.push_back (new_landmark);
      }


      // Apply camshift.
      apply_camshift (frame, track_windows, ROIs_hist, rotated_rects);


      // Reorder according the shortest distances between the new_landmarks and track_windows.
      reorder_landmarks (new_landmarks, track_windows);


      // Draw.
      line_landmarks2windows (frame, new_landmarks, track_windows);
      draw_landmarks (frame, new_landmarks, 0.5, colors);
      draw_track_windows (frame, track_windows, colors);

      putText (frame, "DETECTED", cv::Point (50, 50),
          FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar (0, 255, 0), 10);
    } else if (!detected) {
      cout << "Not detected" << endl;
      putText (frame, "NOT DETECTED", cv::Point (50, 50),
          FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar (0, 0, 255), 10);
    }


    imshow ("window", frame);

    if (waitKey (30) >= 0)
      break;
  }

  return 0;
}
