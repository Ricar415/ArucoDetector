#include "aruco_handler.hpp"

aruco_handler::aruco_handler(){}

void aruco_handler::clear_arucos() { arucos.clear(); }
void aruco_handler::clear_all() { old_arucos.clear(); arucos.clear(); }
void aruco_handler::next() { old_arucos = arucos; arucos.clear(); }
void aruco_handler::add(aruco a) { arucos.push_back(a); }
void aruco_handler::remove(int a) { arucos.erase(arucos.begin() + a); }
void aruco_handler::camera_matrix(Mat cam) { cameraMatrix = cam; }
void aruco_handler::dist_coeffs(Mat dist) { distCoeffs = dist; }

vector<Point2f> aruco_handler::points() {
	vector<Point2f> aux;
	for (int i = 0; i < old_arucos.size(); i++) {
		for (int j = 0; j < old_arucos[i].corners.size(); j++) {
			aux.push_back(old_arucos[i].corners[j]);
		}
	}
	return aux;
}

bool aruco_handler::checkcorners(vector<Point> check) {
	bool a = true;
	for (int i = 0; i < arucos.size(); i++) { //for each aruco already saved
		for (int w = 0; w < 4; w++) { //for each corner saved
			for (int j = 0; j < check.size(); j++) { //for each corner given
				if (arucos[i].corners[w].x - check[j].x < 15 && arucos[i].corners[w].y - check[j].y < 15 && arucos[i].corners[w].x - check[j].x > -15 && arucos[i].corners[w].y - check[j].y > -15) {
					arucos[i].corners[w] = check[j];
					a = false;
				}
			}
		}
	}
	return a;
}

int aruco_handler::load_library(string folder, int ARUCO_SIZE) {
	library.clear();
	for (const auto& dirEntry : experimental::filesystem::directory_iterator(folder + "\\aruco_library")){
		Mat it;
		it = imread(dirEntry.path().u8string());
		if (it.data != NULL) {
			cvtColor(it, it, CV_RGB2GRAY);
			resize(it, it, Size(ARUCO_SIZE + 2, ARUCO_SIZE + 2), INTER_AREA);
			threshold(it, it, 200, 255, THRESH_BINARY);
			lib a; 
			it.copyTo(a.img);
			a.mean = mean(it)[0];
			library.push_back(a);
		}
	}
	return library.size();
}

int aruco_handler::load_shapes(string folder) {
	shapes.clear();
	int i = 0;
	for (const auto& dirEntry : experimental::filesystem::directory_iterator(folder + "\\shapes_library")) {
		vector<Point3f> vecs;
		ifstream a(dirEntry.path().u8string());
		double x=0, y=0, z=0;
		vector<float> nums;
		if (!a.fail()) { a >> x; }
		if (!a.fail()) { a >> y; }
		if (!a.fail()) { a >> z; }
		if (i < library.size()) { library[i].color = Scalar(x, y, z); }
		while (!a.fail()) {
			float aux;
			a >> aux;
			nums.push_back(aux);
		}
		for (int j = 0; j < nums.size()/3; j++) {
			vecs.push_back(Point3f(nums[j * 3], nums[j * 3 + 1], nums[j * 3 + 2]));
		}
		shapes.push_back(vecs);
		a.close();
		i++;
	}
	return shapes.size();
}

Mat aruco_handler::preprocess(Mat img) {
	Mat aux;
	pyrDown(img, aux, Size(img.cols / 2, img.rows / 2));
	pyrUp(aux, aux, img.size());
	cvtColor(aux, aux, CV_BGR2GRAY);
	threshold(aux, aux, mean(aux)[0]-20, 255, THRESH_BINARY);
	return aux;
}

void aruco_handler::opticalflow(Mat &gray, Mat &prev) {
	vector<Point2f>  pointso;
	vector<uchar> status;
	vector<float> err;
	vector<Point2f> pointsi = this->points();

	if (pointsi.size() > 0) {
		calcOpticalFlowPyrLK(prev, gray, pointsi, pointso, status, err, Size(15, 15));
	}
	for (int i = 0; i < pointso.size() / 4; i++) {  //for each set of 4 points
		if (status[i * 4] != 0 && status[i * 4 + 1] != 0 && status[i * 4 + 2] != 0 && status[i * 4 + 3] != 0 && this->old_arucos[i].vframes / (this->old_arucos[i].nframes + 1) > 0.1) { //if all corners are found and that aruco has been checked right for at least 10% of the frames
			aruco aux;
			aux = this->old_arucos[i];
			aux.corners[0] = pointso[i * 4];
			aux.corners[1] = pointso[i * 4 + 1];
			aux.corners[2] = pointso[i * 4 + 2];
			aux.corners[3] = pointso[i * 4 + 3];
			aux.nframes++; //number of frames this aruco has been without check
			this->arucos.push_back(aux);
		}
	}
}

void aruco_handler::find_rectangles(Mat img) {
	vector<vector<Point>> contours;
	vector<Point> approx;
	Mat auxi;
	Canny(img, auxi, 10, 50, 3);
	findContours(auxi, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);
		if (fabs(contourArea(contours[i])) < 100 || !isContourConvex(approx)) continue;

		if (approx.size() == 4) {
			if (this->checkcorners(approx)) { //if its not already saved
				aruco aux;
				aux.corners.push_back(approx[0]);
				aux.corners.push_back(approx[1]);
				aux.corners.push_back(approx[2]);
				aux.corners.push_back(approx[3]);
				this->add(aux);
			}
		}
	}
}

void aruco_handler::perspective_correction(Mat img, int aruco_frame) {
	for (int i = 0; i < arucos.size(); i++) {
		arucos[i].perspective(img, aruco_frame);
	}
}

int aruco_handler::id(int ARUCO_SIZE, int ARUCO_RES) {
	int aruco_frame = (ARUCO_SIZE + 2)*ARUCO_RES;
	int n_pixels = (ARUCO_SIZE + 2)*(ARUCO_SIZE + 2);

	for (int i = 0; i < arucos.size(); i++) {
		int id = -1;
		Mat aux, clear;
		if (arucos[i].nframes > 1) { id = arucos[i].id; }
		//threshold making the frame black
		resize(arucos[i].img, arucos[i].img, Size(ARUCO_SIZE + 2, ARUCO_SIZE + 2), 0, 0, INTER_AREA);
		for (int j = 0; j < arucos[i].img.cols; j++) {
			for (int g = 0; g < arucos[i].img.rows; g++) {
				if (j == 0 || j == arucos[i].img.cols - 1 || g == 0 || g == arucos[i].img.rows - 1) arucos[i].img.at<char>(g, j) = 0;
			}
		}
		threshold(arucos[i].img, arucos[i].img, 200, 255, THRESH_BINARY);
		//check id
		for (int j = 0; j < library.size(); j++) {
			if ((mean(arucos[i].img)[0] > (library[j].mean - 255 / n_pixels)) && (mean(arucos[i].img)[0] < (library[j].mean + 255 / n_pixels))) {
				for (int a = 0; a < 4; a++) {
					Mat aux(ARUCO_SIZE + 2, ARUCO_SIZE + 2, arucos[i].img.type());
					Mat rot(ARUCO_SIZE + 2, ARUCO_SIZE + 2, arucos[i].img.type());
					arucos[i].img.copyTo(rot);
					for (int t = 0; t < a; t++) {
						rotate(rot, rot, ROTATE_90_CLOCKWISE);
					}
					bitwise_xor(rot, library[j].img, aux);
					if (countNonZero(aux) == 0) {
						id = j;
						arucos[i].vframes++;
						arucos[i].point_rot(a); //change position of corners to align x,y of the library aruco
					}
				}
			}
		}
		arucos[i].id = id;
		
	}
	for (int i = 0; i < arucos.size(); i++) {
		if (arucos[i].id == -1) {
			remove(i);
			i--;
		}
	}
	return arucos.size();
}

void aruco_handler::solvePnPs() {
	for (int i = 0; i < arucos.size(); i++) {
		arucos[i].solvePnPs(cameraMatrix, distCoeffs);
	}
}

void aruco_handler::paint(Mat &im) {
	for (int i = 0; i < arucos.size(); i++) {
		Point y_right = arucos[i].corners[0] + (arucos[i].corners[3] - arucos[i].corners[0]) / 2;
		Point x_top = arucos[i].corners[3] + (arucos[i].corners[2] - arucos[i].corners[3]) / 2;

		line(im, arucos[i].corners[0], arucos[i].corners[1], Scalar(255, 255, 0), 1); //paint sides
		line(im, arucos[i].corners[1], arucos[i].corners[2], Scalar(255, 255, 0), 1);
		line(im, arucos[i].corners[2], arucos[i].corners[3], Scalar(255, 255, 0), 1);
		line(im, arucos[i].corners[3], arucos[i].corners[0], Scalar(255, 255, 0), 1);

		vector<Point3f> points;
		points.push_back(Point3f(0, 0, 0));
		points.push_back(Point3f(0, 0, -100));
		vector <Point2f> points2 = arucos[i].project(points, cameraMatrix, distCoeffs);
		Point2f center = points2[0];
		line(im, center, x_top, Scalar(0, 255, 0), 1);
		line(im, center, y_right, Scalar(0, 0, 255), 1);
		line(im, center, points2[1], Scalar(255, 0, 0), 1);
		circle(im, center, 1, Scalar(255, 0, 255), 1);

		circle(im, arucos[i].corners[0], 1, Scalar(255, 0, 255), 1); //paint corners
		circle(im, arucos[i].corners[1], 1, Scalar(255, 0, 255), 1);
		circle(im, arucos[i].corners[2], 1, Scalar(255, 0, 255), 1);
		circle(im, arucos[i].corners[3], 1, Scalar(255, 0, 255), 1);

	}
}

void aruco_handler::paintIds(Mat &img){
	for (int i = 0; i < arucos.size(); i++) {
		vector<Point3f> auxvec;
		Point3f auxpoint(0, 0, -200);
		auxvec.push_back(auxpoint);
		vector<Point2f> vec = arucos[i].project(auxvec, cameraMatrix, distCoeffs);
		//rectangle(img, Rect(Point(vec[0].x -5, vec[0].y +3), Point(vec[0].x + 45, vec[0].y - 15)), Scalar(0,0,0),-1); //alternative
		putText(img, "ID: " + to_string(arucos[i].id), vec[0],FONT_HERSHEY_DUPLEX, 0.5, Scalar(255,255,255));
	}
}

void aruco_handler::shapes3d(Mat & im) {
	for (int i = 0; i < arucos.size(); i++) {
		vector<Point3f> a = shapes[arucos[i].id];
		if (a.size() != 0) { paint3d(im, a, i); }
	}
}

void aruco_handler::paint3d(Mat &im, vector<Point3f> points, int a) {
	vector<Point2f> projected;

	float angle = 0.05*arucos[a].nframes; //angle varies with the number of frames that aruco has been detected 
	
	float rot[3][3] = {{ cos(angle), -sin(angle), 0},{ sin(angle), cos(angle), 0},{ 0, 0, 1}}; //rotation matrix with the set angle
	vector<Point3f> rotated_points;
	for (int i = 0; i < points.size(); i++) { //rotate every point in the shape
		Point3f aux;
		aux.x = rot[0][0] * points[i].x + rot[0][1] * points[i].y + rot[0][2] * points[i].z;
		aux.y = rot[1][0] * points[i].x + rot[1][1] * points[i].y + rot[1][2] * points[i].z;
		aux.z = rot[2][0] * points[i].x + rot[2][1] * points[i].y + rot[2][2] * points[i].z;
		rotated_points.push_back(aux);
	}

	projected = arucos[a].project(rotated_points, cameraMatrix, distCoeffs); //proyect the shape to get the 2D points


	for (int i = 1; i < projected.size(); i+=2) {
		line(im, projected[i], projected[i - 1], library[arucos[a].id].color);  //paint the points with the color for that shape
	}
}

Mat aruco_handler::loop(Mat &img, Mat &prev, int ARUCO_SIZE, int ARUCO_RES) {
	Mat gray, prevgray;
	gray = this->preprocess(img);
	prevgray = this->preprocess(prev);
	this->opticalflow(gray, prevgray);
	this->find_rectangles(gray); //get rectangles
	this->perspective_correction(gray, (ARUCO_SIZE + 2)*ARUCO_RES); //first posible arucos set
	this->id(ARUCO_SIZE, ARUCO_RES); //id arucos and filter out non-found ones
	this->solvePnPs();
	return gray;
}