#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
class ChapterSixth
{
private:
	bool isEroding = false;
	bool isDilating = false;
	bool isGraussianing = false;

	Mat srcImage;
	Mat dstImage;
	Mat erodeImage;
	Mat dilateImage;
	Mat graussianImage;
public:
	ChapterSixth();
	~ChapterSixth();
	void onErodeValueChange(int pos, void *data);
	void onDilateValueChange(int pos, void *data);
	void onGraussianValueChange(int pos, void *data);

	void TestEraseBackground();
};

