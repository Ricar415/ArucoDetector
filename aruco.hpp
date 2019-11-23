#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <opencv/cv.hpp>

using namespace std;
using namespace cv;

class aruco {
public:
	vector<Point2f> corners;
	int id = -1;
	int nframes = 0;
	int vframes = 1;
	Mat img;
	Mat homography;
	Mat rvec;
	Mat tvec;

	aruco();
	~aruco();

	void point_rot(int t);

	bool perspective(Mat img, int frame);
	void solvePnPs(Mat cameraMatrix, Mat distCoeffs);

	vector<Point2f> project(vector<Point3f> points, Mat cameraMatrix, Mat distCoeffs);
	Point2f project(Point3f points, Mat cameraMatrix, Mat distCoeffs);

};

