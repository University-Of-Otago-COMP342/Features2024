#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "Timer.h"

using namespace cv;
using namespace std;


cv::Mat translationMatrix(double dx, double dy) {
	cv::Mat T = cv::Mat::eye(3, 3, CV_64F);
	T.at<double>(0, 2) = dx;
	T.at<double>(1, 2) = dy;
	return T;
}

int main(int argc, char* argv[]) {

	Timer myTimer;

	//read images
	cv::Mat image1 = cv::imread("ImagesForTesting\\image1.jpg");
	cv::Mat image2 = cv::imread("ImagesForTesting\\image2.jpg");

	//create sift pointer and use it to detect features
	Ptr<SIFT> sift = SIFT::create();
	std::vector<cv::KeyPoint> keypoints1;
	cv::Mat descriptors1;
	sift->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	std::cout << "Found " << keypoints1.size() << " features" << std::endl;

	//use the sift pointer to detect features
	std::vector<cv::KeyPoint> keypoints2;
	cv::Mat descriptors2;
	sift->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
	std::cout << "Found " << keypoints2.size() << " features" << std::endl;


	if (image1.empty()) {
		std::cerr << "Could not load image from image1.jpg" << std::endl;
		return -1;
	}
	if (image2.empty()) {
		std::cerr << "Could not load image from image2.jpg" << std::endl;
		return -1;
	}

	//creating Mats to hold the two images with features overlayed side by side
	cv::Mat kptImage1;
	cv::drawKeypoints(image1, keypoints1, kptImage1, cv::Scalar(0, 255, 0),
		cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	cv::Mat kptImage2;
	cv::drawKeypoints(image2, keypoints2, kptImage2, cv::Scalar(0, 255, 0),
		cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//matcher algorithim to compare the features detected in the image
	//better matcher algorithim
	cv::Ptr<cv::DescriptorMatcher> matcher1 = cv::FlannBasedMatcher::create();

	

	//create a vector to store the matches, here we have a list of 
	std::vector<std::vector<cv::DMatch>> matches1;

	//find matches in the features
	//knnMatches are finding k Nearest Neighbor matches
	matcher1->knnMatch(descriptors1, descriptors2, matches1, 2);

	//this list will hold the feature point locations
	std::vector<cv::Point2f> goodPts1, goodPts2;


	//choose only the matches whose matches are significatly better than the second best matches
	std::vector<cv::DMatch> goodMatches;
	for (const auto& match : matches1) {
		if (match[0].distance < 0.8 * match[1].distance) {
			goodMatches.push_back(match[0]);

			//names are a carry over from SIFTS original use as object recognition 
			//query points are to be matched to training points
			goodPts1.push_back(keypoints1[match[0].queryIdx].pt);
			goodPts2.push_back(keypoints2[match[0].trainIdx].pt);
		}
	}

	//reset timer to 0
	myTimer.reset();

	//compute homography
	std::vector<unsigned char> inliers;
	cv::Mat H = cv::findHomography(goodPts2, goodPts1, inliers, cv::RANSAC);

	//read timer output
	std::cout << "++++++++ That code took " << myTimer.read() << " seconds +++++++" << std::endl;

	return 0;
}
