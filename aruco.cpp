#include "aruco.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <opencv/cv.hpp>

using namespace std;
using namespace cv;

aruco::aruco(){ Mat rvec(3, 1, DataType<float>::type),tvec(3, 1, DataType<float>::type);}
aruco::~aruco(){}

void aruco::point_rot(int t) {
	if (corners.size() != 4) { return; } //this should never happen
	vector<Point2f> aux(4);
	for (int i = 0; i < t; i++) {
		aux = corners;
		corners[0] = aux[1];
		corners[1] = aux[2];
		corners[2] = aux[3];
		corners[3] = aux[0];
	}
}

bool aruco::perspective(Mat imge, int aruco_frame) {
	if (corners.size() != 4) return 0;
	vector<Point2f> ina = corners;
	vector<Point2f> outa(4);
	outa[0] = Point(0, 0);
	outa[1] = Point(0, aruco_frame);
	outa[2] = Point(aruco_frame, aruco_frame);
	outa[3] = Point(aruco_frame, 0);

	Mat a(2, 4, CV_32FC1);
	a = Mat::zeros(imge.rows, imge.cols, imge.type());
	a = findHomography(ina, outa);
	a.copyTo(homography);
	Mat x(aruco_frame, aruco_frame, imge.type());

	warpPerspective(imge, x, a, Size(aruco_frame, aruco_frame), INTER_AREA);
	x.copyTo(img);
	
	return 1;
}

void aruco::solvePnPs(Mat cameraMatrix, Mat distCoeffs) {
	vector<Point3f> points;
	float x, y, z;
	x = -100; y = 100; z = 0;
	points.push_back(Point3f(x, y, z));
	x = -100; y = -100; z = 0;
	points.push_back(Point3f(x, y, z));
	x = 100; y = -100; z = 0;
	points.push_back(Point3f(x, y, z));
	x = 100; y = 100; z = 0;
	points.push_back(Point3f(x, y, z));

	vector<Point2f> points2;
	x = corners[0].x;
	y = corners[0].y;
	points2.push_back(Point2f(x, y));
	x = corners[3].x;
	y = corners[3].y;
	points2.push_back(Point2f(x, y));
	x = corners[2].x;
	y = corners[2].y;
	points2.push_back(Point2f(x, y));
	x = corners[1].x;
	y = corners[1].y;
	points2.push_back(Point2f(x, y));


	solvePnP(points, points2, cameraMatrix, distCoeffs, rvec, tvec);
}

vector<Point2f> aruco::project(vector<Point3f> points, Mat cameraMatrix, Mat distCoeffs) {
	vector<Point2f> projectedPoints;
	projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
	return projectedPoints;
}

Point2f aruco::project(Point3f point, Mat cameraMatrix, Mat distCoeffs) {
	vector<Point2f> projectedPoints;
	vector<Point3f> points;
	points.push_back(point);
	projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
	return projectedPoints[0];
}