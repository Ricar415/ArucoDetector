#include <opencv/cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>

#include "aruco.hpp"
#include "aruco_handler.hpp"

using namespace std;
using namespace cv;


Mat cameraMatrix(3, 3, DataType<float>::type);
Mat distCoeffs(5, 1, DataType<float>::type);

bool load_matrices(string matricesdoc);

int main() {
	//--definitions
	double t0 = 0, fps = 0, tx = 0;
	int x = 0, ARUCO_SIZE = 4, ARUCO_RES = 5;
	char key = 0;
	string current_folder = experimental::filesystem::current_path().u8string();
	Mat org, prev;
	VideoCapture vid(0);
	aruco_handler handler;
	//--

	//load data from files (library, shapes and matrices for the camera)
	if (load_matrices("matrices.txt") == 1) { cout << "Matrices loaded" << endl; } //default matrices file = matrices.txt
	else { cout << "No matrices loaded" << endl; }
	handler.load_library(current_folder, ARUCO_SIZE); //load files from library folder
	handler.load_shapes(current_folder); //load files from shapes folder
	handler.camera_matrix(cameraMatrix);
	handler.dist_coeffs(distCoeffs);
	//--

	if (!vid.isOpened()) { cerr << "Camera not found!" << endl; return 0; }
	vid.read(org); //get first frame
	org.copyTo(prev);

	while (key != 27) {
		vid.read(org); //get image
		handler.loop(org, prev, ARUCO_SIZE, ARUCO_RES); //full image processing loop
		org.copyTo(prev); //save image as previous (for optical flow)
		handler.paint(org); //paint axis and borders
		handler.shapes3d(org); //paint corresponding 3D shapes
		handler.paintIds(org); //paint IDs
		handler.next(); //save vector of points as previous

		//--fps show
		x++;
		if ((((double)getTickCount() - tx) / getTickFrequency()) >= 1) { //frames showed in the last second
			fps = x;
			x = 0;
			tx = (double)getTickCount();
		}
		putText(org, to_string(fps), Point(5, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 143), 1);
		//--

		imshow("Aruco_Detector", org); //show result

		key = waitKey(1);
	}
	return 1;
}

bool load_matrices(string matricesdoc) { //load existing matrices
	ifstream a(matricesdoc);
	vector<float> nums;
	while (!a.fail()) {
		float aux;
		a >> aux;
		nums.push_back(aux);
	}
	a.close();
	if (nums.size() < 12) { return 0; }
	for (int i = 0; i < 3; i++) {
		cameraMatrix.at<float>(i, 0) = nums[i * 3];
		cameraMatrix.at<float>(i, 1) = nums[i * 3 + 1];
		cameraMatrix.at<float>(i, 2) = nums[i * 3 + 2];
	}
	for (int i = 0; i < nums.size() - 10; i++) {
		distCoeffs.at<float>(i) = nums[i + 9];
	}
	return 1;
}