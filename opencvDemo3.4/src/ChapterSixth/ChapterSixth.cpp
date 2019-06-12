#include "stdafx.h"
#include "ChapterSixth.h"

void onErodeValueChangeCallBack(int pos, void *data)
{
	if(pos != 0)
		((ChapterSixth *)data)->onErodeValueChange(pos, 0);
}
void onDilateValueChangeCallBack(int pos, void *data)
{
	if (pos != 0)
		((ChapterSixth *)data)->onDilateValueChange(pos, 0);
}
void onGraussianValueChangeCallBack(int pos, void *data)
{
	if (pos != 0)
		((ChapterSixth *)data)->onGraussianValueChange(pos, 0);
}

void ChapterSixth::onErodeValueChange(int pos, void *data)
{
	Mat element = getStructuringElement(MORPH_RECT, Size(pos, pos));
	if (!isEroding)
	{
		erodeImage = dstImage.clone();
	}
	//Mat dstImage;
	erode(erodeImage, dstImage, element);
	imshow("TestEraseBackground", dstImage);
	isEroding = true;
	isDilating = false;
	isGraussianing = false;
}

void ChapterSixth::onDilateValueChange(int pos, void *data)
{
	Mat element = getStructuringElement(MORPH_RECT, Size(pos, pos));
	if (!isDilating)
	{
		dilateImage = dstImage.clone();
	}
	dilate(dilateImage, dstImage, element);
	imshow("TestEraseBackground", dstImage);
	isEroding = false;
	isDilating = true;
	isGraussianing = false;
}

void ChapterSixth::onGraussianValueChange(int pos, void *data)
{
	//Mat dstImage;
	if (pos % 2 == 0)
	{
		pos = pos + 1;
	}
	if (!isGraussianing)
	{
		graussianImage = dstImage.clone();
	}
	GaussianBlur(graussianImage, dstImage, Size(pos, pos), 0);
	imshow("TestEraseBackground", dstImage);
	isEroding = false;
	isDilating = false;
	isGraussianing = true;
}

ChapterSixth::ChapterSixth()
{
}


ChapterSixth::~ChapterSixth()
{
}

void ChapterSixth::TestEraseBackground()
{
	namedWindow("TestEraseBackground", CV_WINDOW_AUTOSIZE);
	srcImage = imread("../TestMaterials/qiammo.png");
	dstImage = srcImage;

	imshow("TestEraseBackground", srcImage);
	createTrackbar("erode nuclear", "TestEraseBackground", 0, 20, onErodeValueChangeCallBack, this);
	createTrackbar("dilate nuclear", "TestEraseBackground", 0, 20, onDilateValueChangeCallBack, this);
	createTrackbar("grauss nuclear", "TestEraseBackground", 0, 20, onGraussianValueChangeCallBack, this);
}
