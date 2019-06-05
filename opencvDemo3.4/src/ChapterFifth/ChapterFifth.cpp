#include "stdafx.h"
#include "ChapterFifth.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <time.h>
using namespace std;
using namespace cv;

void ChapterFifth::onAlphaChange(int pos, void *data)
{
	Mat srcImageA = imread("G:\\TestMaterials\\photo\\1280_800_1.jpeg");
	//imshow("srcImageA", srcImageA);
	Mat srcImageB = imread("G:\\TestMaterials\\photo\\1280_800_3.jpg");
	//imshow("srcImageB", srcImageB);

	if (srcImageA.channels() != srcImageB.channels())
	{
		return;
	}

	if (srcImageA.rows != srcImageB.rows || srcImageA.cols != srcImageB.cols)
	{
		return;
	}
	Mat dstImage;
	double alphaValue = pos / 10.0;
	addWeighted(srcImageA, alphaValue, srcImageB, 1.0-alphaValue, 0.0, dstImage);
	imshow("Alpha", dstImage);

}

void ChapterFifth::onContrastAndBright(int pos, void * data)
{
	Mat srcImage = imread("G:\\TestMaterials\\photo\\1280_800_3.jpg");
//	Mat dstImage = srcImage.clone();

	for (int i=0; i<srcImage.rows; i++)
	{
		for (int j =0; j<srcImage.cols; j++)
		{
			for (int k=0; k<srcImage.channels(); k++)
			{
				srcImage.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(srcImage.at<Vec3b>(i, j)[k] + pos);
			}
		}
	}

	imshow("AdjustBright", srcImage);

}

void ChapterFifth::onDilateValueChange(int pos, void * data)
{
	Mat srcImage = imread("G:\\TestMaterials\\06.jpg");
	Mat element = getStructuringElement(MORPH_RECT, Size(pos, pos));
	Mat out;
	dilate(srcImage, out, element);
	imshow("out", out);
}

void ChapterFifth::onErodeValueChange(int pos, void * data)
{
	Mat srcImage = imread("G:\\TestMaterials\\06.jpg");
	Mat element = getStructuringElement(MORPH_RECT, Size(pos, pos));
	Mat out;
	erode(srcImage, out, element);
	imshow("out", out);
}

void ChapterFifth::onMorphologyExValueChange(int pos, void *data)
{
	//Mat srcImage = imread("G:\\TestMaterials\\04.jpg");
	Mat srcImage = imread("G:\\TestMaterials\\06.jpg");
	//Mat srcImage = imread("G:\\TestMaterials\\qiammo.png");
	imshow("TestMorphologyEx", srcImage);
	if (pos == 0)
	{
		return;
	}
	Mat dstImage;
	Mat element = getStructuringElement(MORPH_RECT, Size(pos, pos));
	morphologyEx(srcImage, dstImage, MORPH_BLACKHAT, element);
	imshow("dst", dstImage);
}
int ChapterFifth::loDiffValue = 0;
int ChapterFifth::upDiffValue = 0;
int ChapterFifth::modeValue = 0;
void ChapterFifth::onScalarLoDiffValueChange(int pos, void * data)
{
	loDiffValue = pos;
}

void ChapterFifth::onScalarUpDiffValueChange(int pos, void * data)
{
	upDiffValue = pos;
}

void ChapterFifth::onScalarModeValueChange(int pos, void * data)
{
	modeValue = pos;
}

int ChapterFifth::thresholdType = 0;
int ChapterFifth::thresholdValue = 0;
void ChapterFifth::onThresholdTypeValueChange(int pos, void * data)
{
	thresholdType = pos;
}

void ChapterFifth::onThresholdValueChange(int pos, void * data)
{
	thresholdValue = pos;
}

int ChapterFifth::cannyHighValue = 5;
int ChapterFifth::cannyLowValue = 5;
void ChapterFifth::onCannyHighValue(int pos, void * data)
{
	cannyHighValue = pos;
}

void ChapterFifth::onCannyLowValue(int pos, void * data)
{
	cannyLowValue = pos;
}



void ChapterFifth::TestAddWeight()
{
	namedWindow("Alpha", WINDOW_AUTOSIZE);

	Mat srcImageA = imread("G:\\TestMaterials\\photo\\1280_800_2.jpg");
	imshow("srcImageA", srcImageA);
	Mat srcImageB = imread("G:\\TestMaterials\\photo\\1280_800_3.jpg");
	imshow("srcImageB", srcImageB);
	
	createTrackbar("ajustAlpha", "Alpha", 0, 10, onAlphaChange, nullptr);
}

void ChapterFifth::TestSplit()
{
	Mat srcImageA = imread("G:\\TestMaterials\\photo\\1280_800_3.jpg");
	imshow("srcImageA", srcImageA);
	
	vector<Mat> channels;
	split(srcImageA, channels);
	int count = 0;
	for each (Mat tmp in channels)
	{
		string name = to_string(count);
		imshow(name, tmp);
		count++;
	}
}

void ChapterFifth::TestMerge()
{
	Mat srcImageA = imread("G:\\TestMaterials\\photo\\1280_800_3.jpg");
	//imshow("srcImageA", srcImageA);

	vector<Mat> channels;
	split(srcImageA, channels);

	Mat dstImage;
	merge(channels, dstImage);
	imshow("dstImage", dstImage);

}

void ChapterFifth::TestBright()
{
	namedWindow("AdjustBright", WINDOW_AUTOSIZE);


	createTrackbar("对比度", "AdjustBright", 0, 100, onContrastAndBright, 0);
	createTrackbar("亮度", "AdjustBright", 0, 100, onContrastAndBright, 0);
}

void ChapterFifth::TestDFT()
{
	Mat srcImage = imread("G:\\TestMaterials\\photo\\1280_800_3.jpg", IMREAD_GRAYSCALE);
	//Mat srcImage = imread("G:\\TestMaterials\\05.png", IMREAD_GRAYSCALE);
	imshow("src", srcImage);
	int bestRows = getOptimalDFTSize(srcImage.rows);
	int bestCols = getOptimalDFTSize(srcImage.cols);
	Mat dstImage;
	copyMakeBorder(srcImage, dstImage, 0, bestRows - srcImage.rows, 0, bestCols - srcImage.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(dstImage),Mat::zeros(dstImage.size(),CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magnitudeImage = planes[0];
	magnitudeImage += Scalar::all(1);
	log(magnitudeImage, magnitudeImage);
	magnitudeImage = magnitudeImage(Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2));
	int cx = magnitudeImage.cols / 2;
	int cy = magnitudeImage.rows / 2;
	Mat q0(magnitudeImage, Rect(0, 0, cx, cy));
	Mat q1(magnitudeImage, Rect(cx, 0, cx, cy));
	Mat q2(magnitudeImage, Rect(0, cy, cx, cy));
	Mat q3(magnitudeImage, Rect(cx, cy, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);
	imshow("dst", dstImage);

}

void ChapterFifth::TestYaml()
{
	FileStorage fs("test.ab", FileStorage::WRITE);
	fs << "frameCount" << 5;
	time_t rawTime;
	time(&rawTime);
	fs << "calibrationDate" << asctime(localtime(&rawTime));

	Mat cameraMatrix = (Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
	Mat distCoeffs = (Mat_<double>(5, 1) << 0.1, 0.001, -0.001, 0, 0);

	fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;

	fs << "feature" << "[";
	for (int i = 0; i < 3; i++)
	{
		int x = rand() % 640;
		int y = rand() % 480;
		uchar lbp = rand() % 256;
		fs << "{:" << "x" << x << "y" << y << "lbp" << "[:";
		for (int j = 0; j < 8; j++)
		{
			fs << ((lbp >> j) & 1);

		}
		fs << "]" << "}";
	}
	fs << "]";
	fs.release();
}

void ChapterFifth::TestDilate()
{
	namedWindow("Testdilate", CV_WINDOW_AUTOSIZE);
	createTrackbar("核大小", "Testdilate", 0, 20, onDilateValueChange, 0);
}

void ChapterFifth::TestErode()
{
	namedWindow("TestErode", CV_WINDOW_AUTOSIZE);
	createTrackbar("核大小", "TestErode", 0, 20, onErodeValueChange, 0);
}

void ChapterFifth::TestMorphologyEx()
{
	namedWindow("TestMorphologyEx", CV_WINDOW_AUTOSIZE);
	createTrackbar("核大小", "TestMorphologyEx", 0, 20, onMorphologyExValueChange, 0);
	onMorphologyExValueChange(0, 0);
}

void ChapterFifth::TestFloodFill()
{
	namedWindow("TestFloodFill", CV_WINDOW_AUTOSIZE);
	createTrackbar("LoDiffValue", "TestFloodFill", 0, 20, onScalarLoDiffValueChange, 0);
	createTrackbar("UpDiffValue", "TestFloodFill", 0, 20, onScalarUpDiffValueChange, 0);
	createTrackbar("mode", "TestFloodFill", 0, 8, onScalarModeValueChange, 0);

	
	while (true)
	{
		Mat srcImage = imread("G:\\TestMaterials\\06.jpg");
		imshow("TestFloodFill", srcImage);
		Rect ccomp = { 0,0,1024,1024 };
		floodFill(srcImage, Point(0, 0), Scalar(255, 255, 255), &ccomp, Scalar(loDiffValue, loDiffValue, loDiffValue), Scalar(upDiffValue, upDiffValue, upDiffValue), 8);
		imshow("TestFloodFill", srcImage);
		if (waitKey(300) == 'q')
		{
			break;
		}
	}

}

void ChapterFifth::TestPyrUp()
{
	Mat srcImage = imread("G:\\TestMaterials\\05.png");
	imshow("src", srcImage);
	Mat dstImage;

	pyrUp(srcImage, dstImage);
	imshow("dst", dstImage);

}

void ChapterFifth::TestPyrDown()
{
	Mat srcImage = imread("G:\\TestMaterials\\04.jpg");
	imshow("src", srcImage);
	Mat dstImage;

	pyrDown(srcImage, dstImage);
	imshow("dst", dstImage);

}

void ChapterFifth::TestResize()
{
	Mat srcImage = imread("G:\\TestMaterials\\04.jpg");
	imshow("src", srcImage);
	Mat dstImage = Mat::zeros(200, 200, CV_8UC3);
	Mat dstImage1 = Mat::zeros(200, 200, CV_8UC3);
	resize(srcImage, dstImage, dstImage.size(), 1.0, 1.0,2);
	resize(srcImage, dstImage1, dstImage1.size(), 5.0, 5.0);
	imshow("dst", dstImage);
	imshow("dst1", dstImage1);


}

void ChapterFifth::TestThreshold()
{
	namedWindow("TestThreshold", CV_WINDOW_AUTOSIZE);
	createTrackbar("ThresholdType", "TestThreshold", 0, 4, onThresholdTypeValueChange, 0);
	createTrackbar("ThresholdValue", "TestThreshold", 0, 20, onThresholdValueChange, 0);
	Mat srcImage = imread("G:\\TestMaterials\\06.jpg");
	imshow("TestThreshold", srcImage);
	cvtColor(srcImage, srcImage, COLOR_RGB2GRAY);
	Mat dstImage;


	while (true)
	{
		//threshold(srcImage, dstImage, thresholdValue, 255, thresholdType);
		adaptiveThreshold(srcImage, dstImage, 255, 0, thresholdType, 3, thresholdValue);
		imshow("TestThreshold1", dstImage);
		if (waitKey(500) == 'q')
			break;
	}


}

void ChapterFifth::TestCanny()
{
	namedWindow("TestCanny", CV_WINDOW_AUTOSIZE);
	createTrackbar("high value", "TestCanny", &cannyHighValue, 255, onCannyHighValue, 0);
	createTrackbar("low value", "TestCanny", &cannyLowValue, 255, onCannyLowValue, 0);
	//Mat srcImage = imread("G:\\TestMaterials\\06.jpg");
	Mat srcImage = imread("G:\\TestMaterials\\04.jpg");
	imshow("TestCanny", srcImage);
	cvtColor(srcImage, srcImage, COLOR_RGB2GRAY);
	Mat dstImage;


	while (true)
	{
		Canny(srcImage, dstImage, cannyHighValue, cannyLowValue);
		imshow("dstImage", dstImage);
		if (waitKey(500) == 'q')
			break;
	}
}

void ChapterFifth::TestLaplacian()
{
	Mat srcImage = imread("G:\\TestMaterials\\04.jpg");
	imshow("src", srcImage);
	GaussianBlur(srcImage, srcImage, Size(3, 3), 0);
	imshow("GaussianBlur", srcImage);
	cvtColor(srcImage, srcImage, COLOR_RGB2GRAY);
	imshow("cvtColor", srcImage);
	Laplacian(srcImage, srcImage, srcImage.depth());
	imshow("Laplacian", srcImage);
	convertScaleAbs(srcImage, srcImage);
	imshow("convertScaleAbs", srcImage);	
}

void ChapterFifth::TestSobel()
{
	Mat srcImage = imread("G:\\TestMaterials\\04.jpg");
	imshow("src", srcImage);
	cvtColor(srcImage, srcImage, COLOR_RGB2GRAY);
	Mat gradX;
	Mat gradY;
	Sobel(srcImage, gradX, srcImage.depth(), 1, 0);
	imshow("gradX", gradX);
	convertScaleAbs(gradX, gradX);
	imshow("gradX1", gradX);
	Sobel(srcImage, gradY, srcImage.depth(), 1, 0);
	imshow("gradY", gradY);
	convertScaleAbs(gradY, gradY);
	imshow("gradY1", gradY);

	Mat dstImage;
	addWeighted(gradX, 0.5, gradY, 0.5, 0, dstImage);
	imshow("dstImage", dstImage);

}

void ChapterFifth::TestScharr()
{
	Mat srcImage = imread("G:\\TestMaterials\\04.jpg");
	imshow("src", srcImage);
	Mat gradX;
	Mat gradY;
	Scharr(srcImage, gradX, srcImage.depth(), 1, 0);
	imshow("gradX", gradX);
	convertScaleAbs(gradX, gradX);
	imshow("gradX1", gradX);
	Scharr(srcImage, gradY, srcImage.depth(), 1, 0);
	imshow("gradY", gradY);
	convertScaleAbs(gradY, gradY);
	imshow("gradY1", gradY);

	Mat dstImage;
	addWeighted(gradX, 0.5, gradY, 0.5, 0, dstImage);
	imshow("dstImage", dstImage);
}

void ChapterFifth::TestHoughLines()
{
	Mat srcImage = imread("G:\\TestMaterials\\04.jpg");
	Mat cannyImage;
	Mat dstImage;
	Canny(srcImage, cannyImage, 50, 200);
	cvtColor(cannyImage,)

}



