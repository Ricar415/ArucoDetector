#pragma once
#include "aruco.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <opencv/cv.hpp>
#include <filesystem>
#include <fstream>

struct lib{
	Mat img;
	int mean;
	Scalar color;
};

class aruco_handler {
public:
	vector<aruco> arucos;
	vector<aruco> old_arucos;
	vector<lib> library;
	vector<vector<Point3f>> shapes;

	Mat cameraMatrix;
	Mat distCoeffs;
	
	aruco_handler();

	void clear_arucos();
	void clear_all();
	void next();
	void add(aruco a);
	void remove(int a);
	void camera_matrix(Mat cam);
	void dist_coeffs(Mat dist);
	vector<Point2f> points();
	bool checkcorners(vector<Point> check);

	int load_library(string folder, int ARUCO_SIZE);
	int load_shapes(string folder);

	Mat preprocess(Mat img);
	void opticalflow(Mat &gray, Mat &prev);
	void find_rectangles(Mat img);
	void perspective_correction(Mat img, int aruco_frame);
	int id(int ARUCO_SIZE, int ARUCO_RES);
	void solvePnPs();

	Mat loop(Mat &img, Mat &prev, int ARUCO_SIZE, int ARUCO_RES);

	void paint(Mat &img);
	void paintIds(Mat &img);
	void shapes3d(Mat &img);
	void paint3d(Mat &img, vector<Point3f> points, int a);

};

