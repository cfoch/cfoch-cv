#include "tracker8.hpp"


const int FaceTracker::CONTOUR_INDICES[MAX_CONTOUR_POINTS] = {
  // Contour of the face.
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  // Left eyebrow.
  26, 25, 24, 23, 22,
  // Right eyebrow.
  21, 20, 19, 18, 17
};
const float FaceTracker::HIST_H_RANGE[] = { 0, 180 };
const int FaceTracker::HIST_CHANNELS[] = { 0 };
const float *FaceTracker::HIST_RANGES[] = { HIST_H_RANGE };
const int FaceTracker::HIST_SIZE[] = { HIST_H_BINS };


FaceTracker::FaceTracker ()
{
  set_capture_source ();
  init_variables ();
}

FaceTracker::FaceTracker (const char * video_path)
{
  set_capture_source (video_path);
  init_variables ();
}


FaceTracker::FaceTracker (int device)
{
  set_capture_source (device);
  init_variables ();
}

void
FaceTracker::set_capture_source (const char * video_path)
{
  cap = VideoCapture (video_path);
  if (!cap.isOpened ())
    cout << "Could not find webcam device." << endl;
}

void
FaceTracker::set_capture_source (int device)
{
  cap = VideoCapture (device);
  if (!cap.isOpened ())
    cout << "Could not find webcam device." << endl;
}

void
FaceTracker::init_variables ()
{
  detected = false;
  set_random_colors (available_colors, N_AVAILABLE_COLORS);
  detector = get_frontal_face_detector ();
  // TODO
  set_predictor ("data");
}

void
FaceTracker::set_predictor (const char * predictor_path)
{
  deserialize (predictor_path) >> predictor;

}

void
FaceTracker::set_random_colors (cv::Scalar * colors, int n)
{
  RNG rng(12345);
  int i;
  for (i = 0; i < N_AVAILABLE_COLORS; i++)
    colors[i] = cv::Scalar (rng.uniform (0, 255),
        rng.uniform (0, 255), rng.uniform (0, 255));
}


void
FaceTracker::start_tracking ()
{
  while (process_frame ());
}

std::vector<cv::Point>
FaceTracker::landmark_from_shape (full_object_detection & shape)
{
  int i;
  std::vector<cv::Point> landmark;
  for (i = 0; i < shape.num_parts (); i++) {
    cv::Point pt;
    pt = cv::Point (shape.part(i).x(), shape.part(i).y());
    landmark.push_back (pt);
  }
  return landmark;
}

void
FaceTracker::contour_from_landmark (std::vector<cv::Point> & landmark,
    cv::Point contour[MAX_CONTOUR_POINTS])
{
  int i;
  // Create the polygon closing the contour of the face.
  for (i = 0; i < MAX_CONTOUR_POINTS; i++)
    contour[i] = landmark[CONTOUR_INDICES[i]];
}

std::vector<std::vector<cv::Point>>
FaceTracker::contour_from_landmark (std::vector<std::vector<cv::Point>> &
    landmarks)
{
  int i, j;
  std::vector<std::vector<cv::Point>> contours;

  // Create the polygon closing the contour of the face.
  for (i = 0; i < landmarks.size (); i++) {
    std::vector<cv::Point> contour;
    cv::Point contour_array[MAX_CONTOUR_POINTS];

    contour_from_landmark (landmarks[i], contour_array);
    contour.assign (contour_array, contour_array + MAX_CONTOUR_POINTS);

    contours.push_back (contour);
  }
  return contours;
}

void
FaceTracker::apply_mask (cv::Point contour[], Mat & mask,
    Mat & masked)
{
  const cv::Point *contours_pts[1] = { contour };
  const int contour_pts_lens[1] = { MAX_CONTOUR_POINTS };

  // Init masks with black background.
  mask = Mat::zeros (frame.size (), CV_8UC1);

  fillPoly (mask, contours_pts, contour_pts_lens, 1,
      cv::Scalar (255, 255, 255));

  gray.copyTo (masked, mask);

  masks.push_back (mask);
  maskeds.push_back (masked);
}


void
FaceTracker::set_ROI (Rect & rectangle, Mat & frame, Mat & mask, Mat & masked,
    Mat & ROI, Mat & ROI_hist)
{
  Mat ROI_hsv;

  // Assign ROI.
  masked (rectangle).copyTo (ROI);

  // Generate ROI hist.
  cvtColor (frame, ROI_hsv, CV_BGR2HSV);
  calcHist (&ROI_hsv, 1, HIST_CHANNELS, mask, ROI_hist,
      N_HIST_CHANNELS, HIST_SIZE, HIST_RANGES, true, false);

  normalize (ROI_hist, ROI_hist, 0, 255, NORM_MINMAX);
}

void
FaceTracker::apply_camshift (Mat & frame,
    std::vector<cv::RotatedRect> & rotated_rects)
{
  int i;
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
FaceTracker::have_to_ignore (int i_track_window, int i_landmark,
    int ignore[][2], int len)
{
  int i;
  for (i = 0; i < len; i++)
    if (ignore[i][0] == i_track_window || ignore[i][1] == i_landmark)
      return 1;
  return 0;
}

void
FaceTracker::add_next_pair_to_ignore (double ** distance_matrix,
    int n_track_windows, int n_landmarks, int ignore[][2], int n_ignore)
{
  int i_track_window, i_landmark, i_track_window_min, i_landmark_min;
  double minimum_distance = 100000;

  i_track_window_min = i_landmark_min = -1;
  for (i_track_window = 0; i_track_window < n_track_windows; i_track_window++) {
    for (i_landmark = 0; i_landmark < n_landmarks; i_landmark++) {
      if (!have_to_ignore (i_track_window, i_landmark, ignore, n_ignore)) {
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

std::vector<cv::Point>
FaceTracker::get_centroids (std::vector<std::vector<cv::Point>> & polygons)
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
FaceTracker::get_centroids (std::vector<cv::Rect> & rects)
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

void
FaceTracker::reorder_landmarks (
    std::vector<std::vector<cv::Point>> & new_landmarks,
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
      cv::Point diff;
      diff = track_windows_centroids[i_track_window] -
          landmark_centroids[i_landmark];
      distance_matrix[i_track_window][i_landmark] = cv::norm (diff);
    }
  }

  // cout << "Landmark centroids: " << endl;
  // __print_centroids (landmark_centroids);
  // cout << "Track Window centroids: " << endl;
  // __print_centroids (track_windows_centroids);

  // __print_matrix (distance_matrix, n_track_windows, n_landmarks);

  // Populate ignore variable. "Ignore" are the closest pairs.
  for (t = 0; t < n_track_windows; t++)
    add_next_pair_to_ignore (distance_matrix, n_ignore, n_landmarks, ignore, t);


  // cout << "Pairs bef: " << endl;
  // __print_matrix (ignore, n_ignore, 2);
  // cout << " -o-o-o-o- " << endl;


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

  //landmark_contours = contour_from_landmark (new_landmarks);
  //landmark_centroids = get_centroids (landmark_contours);
  //cout << "New Landmark centroids: " << endl;
  //__print_centroids (landmark_centroids);
}

void
FaceTracker::draw_detected_state (Mat & frame, bool detected, cv::Point pt)
{
  string text;
  cv::Scalar color;

  text = detected ? "DETECTED" : "NOT DETECTED";
  color = detected ? cv::Scalar (0, 255, 0) : cv::Scalar (0, 0, 255);

  putText (frame, text, pt, FONT_HERSHEY_SCRIPT_SIMPLEX, 1, color, 10);
}

void
FaceTracker::draw_landmarks (Mat & frame,
    std::vector<std::vector<cv::Point>> & landmarks, double ratio,
    std::vector<cv::Scalar> & colors)
{
  int i, j;
  for (i = 0; i < landmarks.size (); i++)
    for (j = 0; j < landmarks[i].size (); j++)
      putText (frame, to_string (j), landmarks[i][j],
          FONT_HERSHEY_SCRIPT_SIMPLEX, ratio, colors[i]);
}

void
FaceTracker::draw_track_windows (Mat & frame,
    std::vector<cv::Rect> & track_windows, std::vector<cv::Scalar> & colors)
{
  int i, j;
  for (i = 0; i < track_windows.size (); i++)
    cv::rectangle (frame, track_windows[i], colors[i]);
}

void
FaceTracker::draw_line_landmarks2windows (Mat & frame,
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
  }
}

void
FaceTracker::filter_track_windows (std::vector<cv::Rect> & old_track_windows)
{
  int i;
  std::vector<cv::Rect> new_track_windows;
  std::vector<cv::Scalar> new_colors;

  for (i = 0; i < old_track_windows.size (); i++) {
    Rect intersection;
    intersection = old_track_windows[i] & Rect (0, 0, frame.cols, frame.rows);
    if (intersection.area () >= MINIMUM_TRACK_WINDOW_AREA) {
      new_track_windows.push_back (old_track_windows[i]);
      new_colors.push_back (colors[i]);
    }
  }
  old_track_windows = new_track_windows;
  colors = new_colors;
}

Rect
FaceTracker::get_rect_resized (Rect & rect)
{
  int d = 20;
  Rect resized (rect.x + d, rect.y + d, rect.width - d, rect.height - d);
  return resized;
}

int
FaceTracker::process_frame ()
{
  int i;
  std::vector<std::vector<cv::Point>> new_landmarks;
  std::vector<dlib::rectangle> dets;
  array2d<unsigned char> img;

  cap >> frame;
  sz = frame.size ();
  cvtColor (frame, gray, CV_BGR2GRAY);

  assign_image (img, cv_image<rgb_pixel> (frame));
  dets = detector (img);

  detected = dets.size () > 0;

  // TODO. Delete.
  Mat frame2;
  frame.copyTo (frame2);

  if (detected && old_landmarks.empty ()) {
    for (i = 0; i < dets.size (); i++) {
      std::vector<cv::Point> new_landmark;
      full_object_detection shape = predictor (img, dets[i]);
      Rect circumbscribed_rectangle;
      cv::Point contour_pts[MAX_CONTOUR_POINTS];
      Mat mask, masked, ROI, ROI_hist;

      // Generate landmarks.
      new_landmark = landmark_from_shape (shape);
      new_landmarks.push_back (new_landmark);

      // Asign a random color.
      colors.push_back (available_colors[i % N_AVAILABLE_COLORS]);

      // Create a polygon closing the contour of the face.
      contour_from_landmark (new_landmark, contour_pts);

      // TODO: Check if we just can get the dets[i]'s coordinates.
      // Calculate circumbscribed rectangle to the landmark.
      circumbscribed_rectangle =
          boundingRect (std::vector<cv::Point> (contour_pts,
              contour_pts + MAX_CONTOUR_POINTS));
      circumbscribed_rectangle = circumbscribed_rectangle &
          Rect (0, 0, frame.cols, frame.rows);
      circumbscribed_rectangle = get_rect_resized (circumbscribed_rectangle);

      track_windows.push_back (circumbscribed_rectangle);

      // Init masks with black background.
      apply_mask (contour_pts, mask, masked);

      // Genrate ROI & ROI_hist. Add them to their respective lists.
      set_ROI (circumbscribed_rectangle, frame, mask, masked, ROI, ROI_hist);
      ROIs.push_back (ROI);
      ROIs_hist.push_back (ROI_hist);

      old_landmarks = new_landmarks;
    }
    draw_track_windows (frame,track_windows, colors);
    draw_landmarks (frame, new_landmarks, 0.5, colors);
    draw_detected_state (frame, true);

  } else if (detected && !old_landmarks.empty ()) {
    int old_track_windows_size;
    std::vector<cv::RotatedRect> rotated_rects;

    // Assume that track_windows entering to this zone are always correct.
    for (i = 0; i < dets.size (); i++) {
      std::vector<cv::Point> new_landmark;
      full_object_detection shape = predictor (img, dets[i]);
      new_landmark = landmark_from_shape (shape);
      new_landmarks.push_back (new_landmark);
    }

    // Apply camshift for each track_window & update track_windows.
    apply_camshift (frame, rotated_rects);

    // Exclude track_windows outside the captured area.
    filter_track_windows (track_windows);
    // Assume that track_windows entering to this zone are always correct.
    old_track_windows_size = track_windows.size ();

    // Reorder  according the shortest distances between landmarks and windows.
    reorder_landmarks (new_landmarks, track_windows);

    // TODO. Delete.
    draw_line_landmarks2windows (frame2, new_landmarks, track_windows);
    draw_landmarks (frame2, new_landmarks, 0.5, colors);
    draw_track_windows (frame2, track_windows, colors);
    draw_detected_state (frame2, true);


    // Recalculate ROIs
    for (i = 0; i < new_landmarks.size (); i++) {
      Rect circumbscribed_rectangle;
      cv::Point contour_pts[MAX_CONTOUR_POINTS];
      Mat mask, masked, ROI, ROI_hist;

      // Create a polygon closing the contour of the face.
      contour_from_landmark (new_landmarks[i], contour_pts);

      // Calculate circumbscribed rectangle to the landmark.
      circumbscribed_rectangle =
          boundingRect (std::vector<cv::Point> (contour_pts,
              contour_pts + MAX_CONTOUR_POINTS));
      circumbscribed_rectangle = circumbscribed_rectangle &
          Rect (0, 0, frame.cols, frame.rows);
      circumbscribed_rectangle = get_rect_resized (circumbscribed_rectangle);

      if (i < old_track_windows_size)
        track_windows[i] = circumbscribed_rectangle;
      else
        track_windows.push_back(circumbscribed_rectangle);

      // Init masks with black background.
      apply_mask (contour_pts, mask, masked);

      // Genrate ROI & ROI_hist. Add them to their respective lists.
      set_ROI (circumbscribed_rectangle, frame, mask, masked, ROI, ROI_hist);
    }

    // Draw.
    draw_line_landmarks2windows (frame, new_landmarks, track_windows);
    draw_landmarks (frame, new_landmarks, 0.5, colors);
    draw_track_windows (frame, track_windows, colors);
    draw_detected_state (frame, true);

  } else if (!detected) {
    std::vector<RotatedRect> rotated_rects;
    // Apply camshift for each track_window & update track_windows.
    apply_camshift (frame, rotated_rects);

    // Exclude track_windows outside the captured area.
    filter_track_windows (track_windows);

    draw_detected_state (frame, false);
    draw_track_windows (frame, track_windows, colors);
  }

  imshow ("window", frame);
  imshow ("window2", frame2);
  if (waitKey (30) >= 0)
    return 0;
  return 1;
}



