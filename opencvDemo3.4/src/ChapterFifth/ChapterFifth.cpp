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
	Mat srcImageA = imread("../TestMaterials/photo/1280_800_1.jpeg");
	//imshow("srcImageA", srcImageA);
	Mat srcImageB = imread("../TestMaterials/photo/1280_800_3.jpg");
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
	Mat srcImage = imread("../TestMaterials/photo/1280_800_3.jpg");
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
	Mat srcImage = imread("../TestMaterials/06.jpg");
	Mat element = getStructuringElement(MORPH_RECT, Size(pos, pos));
	Mat out;
	dilate(srcImage, out, element);
	imshow("out", out);
}

void ChapterFifth::onErodeValueChange(int pos, void * data)
{
	Mat srcImage = imread("../TestMaterials/06.jpg");
	Mat element = getStructuringElement(MORPH_RECT, Size(pos, pos));
	Mat out;
	erode(srcImage, out, element);
	imshow("out", out);
}

void ChapterFifth::onMorphologyExValueChange(int pos, void *data)
{
	//Mat srcImage = imread("../TestMaterials/04.jpg");
	Mat srcImage = imread("../TestMaterials/06.jpg");
	//Mat srcImage = imread("../TestMaterials/qiammo.png");
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

int ChapterFifth::houghLinesValue = 0;
void ChapterFifth::onHoughLinesValueChange(int pos, void * data)
{
	houghLinesValue = pos;
}


int ChapterFifth::houghLinesPValue = 0;
void ChapterFifth::onHoughLinesPValueChange(int pos, void *data)
{
	houghLinesPValue = pos;
}

int ChapterFifth::houghCircleValue = 1;
void ChapterFifth::onHoughCircleValueChange(int pos, void * data)
{
	if(pos)
		houghCircleValue = pos;
}

int ChapterFifth::angleValue = 0;
void ChapterFifth::onAngleValueChange(int pos, void * data)
{
	angleValue = pos;
}

double ChapterFifth::scaleValue = 1.0;
void ChapterFifth::onScaleValueChange(int pos, void * data)
{
	if (pos)
		scaleValue = (double)pos / 100.0;
}

Point ChapterFifth::previousPoint = Point(-1, -1);
void ChapterFifth::onMouse(int event, int x, int y, int flags, void * data)
{
	if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
		previousPoint = Point(-1, -1);
	else if(event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
	{
		Point pt(x, y);
		if (previousPoint.x < 0)
		{
			previousPoint = pt;
		}
		line(inpaintMaskImage, previousPoint, pt, Scalar::all(255), 5, 8, 0);
		line(srcImageClone, previousPoint, pt, Scalar::all(255), 5, 8, 0);
		imshow("srcImageClone", srcImageClone);
	}

}
int ChapterFifth::binValue = 0;
void ChapterFifth::onBinValueChange(int pos, void * data)
{
	binValue = pos;
	MatND hist;
	int histSize = MAX(binValue, 2);
	float range[] = { 0,180 };
	const float* ranges = { range };
	calcHist(&hImage, 1, 0, Mat(), hist, 1, &histSize, &ranges);
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
	MatND backproj;
	calcBackProject(&hImage, 1, 0, hist, backproj, &ranges, 1, true);
	imshow("反向投影图", backproj);
}

void ChapterFifth::TestAddWeight()
{
	namedWindow("Alpha", WINDOW_AUTOSIZE);

	Mat srcImageA = imread("../TestMaterials/photo/1280_800_2.jpg");
	imshow("srcImageA", srcImageA);
	Mat srcImageB = imread("../TestMaterials/photo/1280_800_3.jpg");
	imshow("srcImageB", srcImageB);
	
	createTrackbar("ajustAlpha", "Alpha", 0, 10, onAlphaChange, nullptr);
}

void ChapterFifth::TestSplit()
{
	Mat srcImageA = imread("../TestMaterials/photo/1280_800_3.jpg");
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
	Mat srcImageA = imread("../TestMaterials/photo/1280_800_3.jpg");
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
	Mat srcImage = imread("../TestMaterials/photo/1280_800_3.jpg", IMREAD_GRAYSCALE);
	//Mat srcImage = imread("../TestMaterials/05.png", IMREAD_GRAYSCALE);
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
		Mat srcImage = imread("../TestMaterials/06.jpg");
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
	Mat srcImage = imread("../TestMaterials/05.png");
	imshow("src", srcImage);
	Mat dstImage;

	pyrUp(srcImage, dstImage);
	imshow("dst", dstImage);

}

void ChapterFifth::TestPyrDown()
{
	Mat srcImage = imread("../TestMaterials/04.jpg");
	imshow("src", srcImage);
	Mat dstImage;

	pyrDown(srcImage, dstImage);
	imshow("dst", dstImage);

}

void ChapterFifth::TestResize()
{
	Mat srcImage = imread("../TestMaterials/04.jpg");
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
	Mat srcImage = imread("../TestMaterials/06.jpg");
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
	Mat srcImage = imread("../TestMaterials/04.jpg");
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
	Mat srcImage = imread("../TestMaterials/04.jpg");
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
	Mat srcImage = imread("../TestMaterials/04.jpg");
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
	Mat srcImage = imread("../TestMaterials/04.jpg");
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
	namedWindow("TestHoughLines", CV_WINDOW_AUTOSIZE);
	createTrackbar("houghLineValue", "TestHoughLines", 0, 300, onHoughLinesValueChange, 0);
	//Mat srcImage = imread("../TestMaterials/04.jpg");
	Mat srcImage = imread("../TestMaterials/02.png");

	Mat cannyImage;
	Canny(srcImage, cannyImage, 50, 200);
	//if(cannyImage.type()== CV_8UC1)
	imshow("srcImage", cannyImage);
	//cvtColor(cannyImage, cannyImage, CV_BGR2GRAY);
	//imshow("dstImage", dstImage);
	while (true)
	{
		vector<Vec2f> lines;
		HoughLines(cannyImage, lines, 1, CV_PI / 180, houghLinesValue);
		Mat dstImage = cannyImage.clone();
		for each (Vec2f tmp in lines)
		{
			float rho = tmp[0];
			float theta = tmp[1];
			double a = cos(theta);
			double b = sin(theta);
			double x0 = a*rho;
			double y0 = b*rho;

			Point pt1, pt2;
			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));
			line(dstImage, pt1, pt2, Scalar(55, 100, 195));
		}
		imshow("dstImage", dstImage);
		waitKey(500);
	}
	

}

void ChapterFifth::TestHoughLinesP()
{
	namedWindow("TestHoughLinesP", CV_WINDOW_AUTOSIZE);
	createTrackbar("houghLinePValue", "TestHoughLinesP", 0, 1000, onHoughLinesPValueChange, 0);
	//Mat srcImage = imread("../TestMaterials/04.jpg");
	Mat srcImage = imread("../TestMaterials/02.png");

	Mat cannyImage;
	Canny(srcImage, cannyImage, 50, 200);
	//if(cannyImage.type()== CV_8UC1)
	imshow("TestHoughLinesP", cannyImage);
	//cvtColor(cannyImage, cannyImage, CV_BGR2GRAY);
	//imshow("dstImage", dstImage);
	while (true)
	{
		vector<Vec4i> lines;
		HoughLinesP(cannyImage, lines, 1, CV_PI / 180, houghLinesPValue);
		Mat dstImage = cannyImage.clone();
		for each (Vec4i tmp in lines)
		{
			line(dstImage, Point(tmp[0],tmp[1]), Point(tmp[2], tmp[3]), Scalar(255, 0, 255));
		}
		imshow("dstImage", dstImage);
		waitKey(500);
	}

}

void ChapterFifth::TestHoughCircle()
{
	namedWindow("TestHoughCircle", CV_WINDOW_AUTOSIZE);
	createTrackbar("houghLinePValue", "TestHoughCircle", 0, 500, onHoughCircleValueChange, 0);
	Mat srcImage = imread("../TestMaterials/04.jpg");
	//Mat srcImage = imread("../TestMaterials/02.png");

	cvtColor(srcImage, srcImage, CV_BGR2GRAY);
	imshow("TestHoughCircle", srcImage);

	GaussianBlur(srcImage, srcImage, Size(9, 9), 2, 2);
	while (true)
	{
		vector<Vec3f> circles;
		HoughCircles(srcImage, circles, HOUGH_GRADIENT, 1, houghCircleValue);
		Mat dstImage = srcImage.clone();
		for each (Vec3f tmp in circles)
		{
			Point center(cvRound(tmp[0]), cvRound(tmp[1]));
			int radius = cvRound(tmp[2]);

			//circle(dstImage, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			circle(dstImage, center, radius, Scalar(255, 155, 0), 1, 8, 0);
		}
		imshow("dstImage", dstImage);
		waitKey(500);
	}
}

void ChapterFifth::TestReMap()
{
	Mat srcImage = imread("../TestMaterials/04.jpg");
	imshow("srcImage", srcImage);
	Mat dstImage;
	dstImage.create(srcImage.size(), srcImage.type());

	Mat dstImageX;
	Mat dstImageY;
	dstImageX.create(srcImage.size(), CV_32FC1);
	dstImageY.create(srcImage.size(), CV_32FC1);

	for (int i = 0; i < srcImage.rows; i++)
	{
		for (int j = 0; j < srcImage.cols; j++)
		{
			/* cols
			    -| -| -| -| -| -| -| -| -| -|- rows
				-| -| -| -| -| -| -| -| -| -| -
				-| -| -| -| -| -| -| -| -| -| -
				-| -| -| -| -| -| -| -| -| -| -
				-| -| -| -| -| -| -| -| -| -| -*/
			//关于Y轴对称
			dstImageX.at<float>(i, j) = static_cast<float>(srcImage.cols - j);
			dstImageY.at<float>(i, j) = static_cast<float>(i);
		}
	}

	remap(srcImage, dstImage, dstImageX, dstImageY, INTER_LINEAR);
	imshow("dstImage", dstImage);
}

void ChapterFifth::TestAffineMap()
{
	namedWindow("TestAffineMap", CV_WINDOW_AUTOSIZE);
	createTrackbar("angleValue", "TestAffineMap", 0, 360, onAngleValueChange, 0);
	createTrackbar("scaleValue", "TestAffineMap", 0, 100, onScaleValueChange, 0);
	//Mat srcImage = imread("../TestMaterials/04.jpg");
	Mat srcImage = imread("../TestMaterials/06.jpg");
	Mat dstImage = Mat::zeros(srcImage.rows, srcImage.cols, srcImage.type());
	Mat dstImageA = Mat::zeros(srcImage.rows, srcImage.cols, srcImage.type());

	Point2f srcTriangle[3];
	Point2f dstTriangle[3];

	//Mat warpMat(2, 3, CV_32FC1);

	srcTriangle[0] = Point2f(0, 0);
	srcTriangle[1] = Point2f(static_cast<float>(srcImage.cols - 1), 0);
	srcTriangle[2] = Point2f(0, static_cast<float>(srcImage.rows - 1));

	dstTriangle[0] = Point2f(static_cast<float>(srcImage.cols*0.0), static_cast<float>(srcImage.rows*0.33));
	dstTriangle[1] = Point2f(static_cast<float>(srcImage.cols*0.65), static_cast<float>(srcImage.rows*0.35));
	dstTriangle[2] = Point2f(static_cast<float>(srcImage.cols*0.15), static_cast<float>(srcImage.rows*0.6));
	Mat warpMat = getAffineTransform(srcTriangle, dstTriangle);
	warpAffine(srcImage, dstImage, warpMat, dstImage.size());
	//imshow("dstImage", dstImage);

	Point center = Point(srcImage.cols / 2, srcImage.rows / 2);
	//double angle = -30.0;
	double scale = 1.0;
	while (true)
	{
		Mat rotMat = getRotationMatrix2D(center, angleValue, scaleValue);
		warpAffine(srcImage, dstImageA, rotMat, dstImageA.size());
		imshow("dstImageA", dstImageA);
		waitKey(500);
	}


}

void ChapterFifth::TestEqualizeHist()
{
	Mat srcImage = imread("../TestMaterials/hdr1.jpg");
	imshow("srcImage", srcImage);

	Mat dstGrayImage;
	cvtColor(srcImage, dstGrayImage, COLOR_BGR2GRAY);
	imshow("dstGrayImage", dstGrayImage);

	Mat dstImage;
	equalizeHist(dstGrayImage, dstImage);
	imshow("dstImage", dstImage);

	Mat dstBRGImage;
	cvtColor(dstImage, dstBRGImage, COLOR_GRAY2BGR);
	imshow("dstBRGImage", dstBRGImage);
}

void ChapterFifth::TestFindContours()
{
	Mat srcImage = imread("../TestMaterials/hdr1.jpg", 0);
	imshow("srcImage", srcImage);
	
	Mat dstImage = Mat::zeros(srcImage.rows, srcImage.cols, CV_8UC3);
	Mat srcImageA = srcImage > 100;
	imshow("srcImageA", srcImageA);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(srcImageA, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	/*for (int index = 0; index >= 0; index = hierarchy[index][0])
	{
		Scalar color(0, 0, 255);
		drawContours(dstImage, contours, index, color, FILLED, 8, hierarchy);
		imshow("dstImage", dstImage);
	}*/
	for (int index = 0; index<contours.size(); index++)
	{
		Scalar color(0, 0, 255);
		drawContours(dstImage, contours, index, color, FILLED, 8, hierarchy);
		imshow("dstImage", dstImage);
	}
}

void ChapterFifth::TestMinAreaRect()
{
	Mat image(800, 600, CV_8UC3);
	RNG& rng = theRNG();//返回默认的随机生成器
	while (true)
	{
		int count = rng.uniform(3, 200);
		vector<Point> points;
		for (int i=0; i<count; i++)
		{
			Point point;
			point.x = rng.uniform(image.cols / 4, image.cols * 3 / 4);
			point.y = rng.uniform(image.rows / 4, image.rows * 3 / 4);
			points.push_back(point);
		}

		RotatedRect box = minAreaRect(Mat(points));
		Point2f vertex[4];
		box.points(vertex);

		image = Scalar::all(0);
		for (int i = 0; i < count; i++)
		{
			circle(image, points[i], 3, Scalar(255, 0, 0), FILLED, LINE_AA);
		}

		for (int i = 0; i < 4; i++)
		{
			line(image, vertex[i], vertex[(i + 1) % 4], Scalar(0, 255, 0), 2, LINE_AA);
		}

		imshow("dstImage", image);
		char key = (char)waitKey();
		if (key == 27 || key == 'q' || key == 'Q')
		{
			break;
		}
	}
}

void ChapterFifth::TestMinEnclosingCircle()
{
	Mat image(800, 600, CV_8UC3);
	RNG& rng = theRNG();//返回默认的随机生成器
	while (true)
	{
		int count = rng.uniform(3, 200);
		vector<Point> points;
		for (int i = 0; i < count; i++)
		{
			Point point;
			point.x = rng.uniform(image.cols / 4, image.cols * 3 / 4);
			point.y = rng.uniform(image.rows / 4, image.rows * 3 / 4);
			points.push_back(point);
		}

		Point2f center;
		float radius = 0;
		minEnclosingCircle(points, center, radius);

		image = Scalar::all(0);
		for (int i = 0; i < count; i++)
		{
			circle(image, points[i], 3, Scalar(255, 0, 0), FILLED, LINE_AA);
		}

		circle(image, center, radius, Scalar(0, 255, 0), 2, LINE_AA);

		imshow("dstImage", image);
		char key = (char)waitKey();
		if (key == 27 || key == 'q' || key == 'Q')
		{
			break;
		}
	}
}

void ChapterFifth::TestContourArea()
{
	vector<Point> contour;
	contour.push_back(Point2f(0, 0));
	contour.push_back(Point2f(10, 0));
	//contour.push_back(Point2f(10, 10));
	//contour.push_back(Point2f(0, 0));

	double area = contourArea(contour,true);
	double len = arcLength(contour, true);
	cout << area << endl;
	cout << len << endl;

	vector<Point> approx;
	approxPolyDP(contour, approx, 5, true);
	double area1 = contourArea(approx);
	double len1 = arcLength(approx,true);
	cout << area1 << endl;
	cout << len1 << endl;


	waitKey(100000);
}

void ChapterFifth::TestWatershed()
{
	RNG& rng = theRNG();//返回默认的随机生成器
	Mat srcImage = imread("../TestMaterials/photo/1280_800_3.jpg");
	//imshow("srcImage", srcImage);

	Mat grayImage;
	cvtColor(srcImage, grayImage, CV_RGB2GRAY);
	GaussianBlur(grayImage, grayImage, Size(5, 5), 2);
	Canny(grayImage, grayImage, 80, 150);
	//imshow("grayImage", grayImage);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(grayImage, contours, hierarchy, RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point());

	Mat imageContours = Mat::zeros(srcImage.size(), CV_8UC1);
	Mat marks(srcImage.size(), CV_32S);
	marks = Scalar::all(0);

	//for (int i=0; i<contours.size();i++)
	//{
	//	drawContours(imageContours, contours, i, Scalar(255), 1, 8, hierarchy);
	//	//drawContours(imageContours, contours, i, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 1, 8, hierarchy);
	//}
	//imshow("imageContours", imageContours);

	Mat imageContoursA = Mat::zeros(srcImage.size(), CV_8UC1);
	int compCount = 0;
	for (int index = 0; index >= 0; index = hierarchy[index][0], compCount++)
	{
		drawContours(imageContoursA, contours, index, Scalar(255), 1, 8, hierarchy);
		//drawContours(marks, contours, index, Scalar::all(compCount + 1), 1, 8, hierarchy);
		drawContours(marks, contours, index, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 1, 8, hierarchy);
	}
	imshow("imageContoursA", imageContoursA);

	Mat marksShows;
	convertScaleAbs(marks, marksShows);
	imshow("marksShows", marksShows);
	watershed(srcImage, marks);
	imshow("waterImage", marks);

	Mat afterWatershed;
	convertScaleAbs(marks, afterWatershed);
	imshow("afterWatershed", afterWatershed);

	Mat dstImage = afterWatershed*0.5 + grayImage*0.5;
	imshow("dstImage", dstImage);

	Mat PerspectiveImage = Mat::zeros(srcImage.size(), CV_8UC3);
	for (int i = 0; i < marks.rows; i++)
	{
		for (int j = 0; j < marks.cols; j++)
		{
			int index = marks.at<int>(i, j);
			if (marks.at<int>(i, j) == -1)
			{
				PerspectiveImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else
			{
				PerspectiveImage.at<Vec3b>(i,j) = Vec3b(rng.uniform(0, index), rng.uniform(0, index), rng.uniform(0, index));
			}
		}
	}
	imshow("PerspectiveImage", PerspectiveImage);
}

Mat ChapterFifth::inpaintMaskImage = Mat();
Mat ChapterFifth::srcImageClone = Mat();
void ChapterFifth::TestInpaint()
{
	Mat srcImage = imread("../TestMaterials/photo/1280_800_3.jpg");
	srcImageClone = srcImage.clone();
	inpaintMaskImage = Mat::zeros(srcImageClone.size(), CV_8U);
	imshow("srcImageClone", srcImageClone);

	setMouseCallback("srcImageClone", onMouse, 0);
	
	while (true)
	{
		char key = (char)waitKey();
		if (key == 'c')
		{
			srcImage.copyTo(srcImageClone);
			inpaintMaskImage = Scalar::all(0);
			imshow("srcImageClone", srcImageClone);

		}
		else if (key == 'r')
		{
			Mat dstImage;
			inpaint(srcImageClone, inpaintMaskImage, dstImage, 5, INPAINT_TELEA);
			imshow("dstImage", dstImage);
		}

	}
}

void ChapterFifth::TestCalcHist()
{
#define OPEN 5
#if (OPEN==0)//HSV分布直方图
	Mat srcImage = imread("../TestMaterials/photo/1280_800_3.jpg");
	Mat hsvImage;
	cvtColor(srcImage, hsvImage, CV_BGR2HSV);
	imshow("hsvImage", hsvImage);

	int hueBinNum = 100;
	int saturationBinNum = 100;
	int histSize[] = { hueBinNum,saturationBinNum };
	float hueRanges[] = { 0,180 };
	float saturationRanges[] = { 0,256 };
	const float* ranges[] = { hueRanges,saturationRanges };
	MatND dstHist;
	int channels[] = { 0,1 };
	calcHist(&hsvImage, 1, channels, Mat(), dstHist, 2, histSize, ranges, true, false);

	double maxValue = 0;
	minMaxLoc(dstHist, 0, &maxValue, 0, 0);
	int scale = 10;
	Mat histImg = Mat::zeros(saturationBinNum*scale, hueBinNum * 10, CV_8UC3);
	for (int hue = 0; hue < hueBinNum; hue++)
	{
		for (int saturation = 0; saturation < saturationBinNum; saturation++)
		{
			float binValue = dstHist.at<float>(hue, saturation);
			int intensity = cvRound(binValue * 255 / maxValue);
			//rectangle(histImg, Point(hue*scale, saturation*scale), Point((hue + 1)*scale - 1, (saturation + 1)*scale - 1), Scalar(0,255,0), FILLED);
			rectangle(histImg, Point(hue*scale, saturation*scale), Point((hue + 1)*scale - 1, (saturation + 1)*scale - 1), Scalar::all(intensity), FILLED);
			
		}
	}

	imshow("srcImage", srcImage);
	imshow("histImg", histImg);
#elif (OPEN==1)//RGB各分量曲线图
	Mat src, dst;
	//加载图像
	src = imread("../TestMaterials/photo/1280_800_3.jpg");

	namedWindow("INPUT_TITLE", CV_WINDOW_AUTOSIZE);
	namedWindow("OUTPUT_TITLE", CV_WINDOW_AUTOSIZE);

	imshow("INPUT_TITLE", src);

	//分通道显示
	vector<Mat> bgr_planes;
	split(src, bgr_planes);

	//设定像素取值范围
	int histSize = 256;
	float range[] = { 0,255 };
	const float *histRanges = { range };

	//三个通道分别计算直方图
	Mat b_hist, g_hist, r_hist;
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRanges, true, false);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRanges, true, false);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRanges, true, false);

	//创建直方图画布并归一化处理
	int hist_h = 400;
	int hist_w = 512;
	int bin_w = hist_w / histSize;
	Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));
	//归一化处理，将某个值归并在某个区间中
	normalize(b_hist, b_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, hist_h, NORM_MINMAX, -1, Mat());

	//render histogram chart  在直方图画布上画出直方图
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point((i - 1)*bin_w, hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point((i)*bin_w, hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, LINE_AA);
		line(histImage, Point((i - 1)*bin_w, hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point((i)*bin_w, hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, LINE_AA);
		line(histImage, Point((i - 1)*bin_w, hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point((i)*bin_w, hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("OUTPUT_TITLE", histImage);

#elif (OPEN==2)
#define HIST_SIZE (256)
	Mat srcImage = imread("../TestMaterials/photo/1280_800_2.jpg");
	Mat dstHist;
	int dims = 1;
	float hranges[] = { 0,255 };
	const float *ranges[] = { hranges };
	int size = HIST_SIZE;
	int channels = 2;
	calcHist(&srcImage, 1, &channels, Mat(), dstHist, dims, &size, ranges, true, false);//size:直方图直条数量
	int scale = 1;
	Mat dstImage(size*scale, size*2, CV_8U, Scalar(0));

	double minValue = 0.0;
	double maxValue = 0.0;
	minMaxLoc(dstHist, &minValue, &maxValue);
	int hpt = saturate_cast<int>(0.9*size);
	for (int i = 0; i < HIST_SIZE; i++)
	{
		float binValue = dstHist.at<float>(i);
		int realValue = saturate_cast<int>(binValue*hpt / maxValue);
		rectangle(dstImage, Point(i*scale, size - 1), Point((i + 1)*scale - 1, size - realValue), Scalar(255));
	}
	imshow("dstImage", dstImage);

#elif (OPEN==3)//RGB各分量直方图
	//载入图片
	Mat srcImage = imread("../TestMaterials/photo/1280_800_2.jpg");
	//imshow("原图", srcImage);

	//参数准备
	int bins = 256;
	int hist_size[] = { bins };
	float range[] = { 0, 256 };
	const float *ranges[] = { range };
	MatND redHist, greenHist, blueHist;
	int channels_r[] = { 0 };

	//进行直方图的计算(红色部分)
	calcHist(&srcImage, 1, channels_r, Mat(), redHist, 1, hist_size, ranges, true, false);

	//绿色分量计算
	int channels_g[] = { 1 };
	calcHist(&srcImage, 1, channels_g, Mat(), greenHist, 1, hist_size, ranges, true, false);

	//蓝色分量计算
	int channels_b[] = { 1 };
	calcHist(&srcImage, 1, channels_b, Mat(), blueHist, 1, hist_size, ranges, true, false);

	//绘制rgb颜色直方图
	//参数准备
	double maxValue_red, maxValue_green, maxValue_blue;
	minMaxLoc(redHist, 0, &maxValue_red, 0, 0);
	minMaxLoc(greenHist, 0, &maxValue_green, 0, 0);
	minMaxLoc(blueHist, 0, &maxValue_blue, 0, 0);
	int scale = 1;
	int histHeight = 256;
	//bins * 3 表示宽度要容纳三幅直方图，RGB
	Mat histImage = Mat::zeros(histHeight, bins * 3, CV_8UC3);

	//开始绘制
	for (int i = 0; i < bins; i++)
	{
		//参数准备
		float binValue_red = redHist.at<float>(i);
		float binValue_green = greenHist.at<float>(i);
		float binValue_blue = blueHist.at<float>(i);

		int intensity_red = cvRound(binValue_red * histHeight / maxValue_red); //计算绘制高度
		int intensity_green = cvRound(binValue_green * histHeight / maxValue_green); //计算绘制高度
		int intensity_blue = cvRound(binValue_blue * histHeight / maxValue_blue); //计算绘制高度

																				  //绘制红色部分直方图
		//图像的坐标原点在左上角，histHeight - 1只是让图像贴近窗口最底层，histHeight - intensity_red算出实际的高度
		rectangle(histImage, Point(i*scale, histHeight - 1), Point((i + 1)*scale - 1, histHeight - intensity_red), Scalar(0, 0, 255));

		//绘制绿色部分直方图
		rectangle(histImage, Point((i + bins)*scale, histHeight - 1), Point((i + bins + 1)*scale - 1, histHeight - intensity_green), Scalar(0, 255, 0));

		//绘制蓝色部分直方图
		rectangle(histImage, Point((i + bins * 2)*scale, histHeight - 1), Point((i + bins * 2 + 1)*scale - 1, histHeight - intensity_blue), Scalar(255, 0, 0));
	}

	//在窗口中显示
	imshow("RGB", histImage);
#elif (OPEN==4) //图像的灰度直方图
    //Mat srcImage = imread("../TestMaterials/photo/1280_800_3.jpg");
    Mat srcImage = imread("../TestMaterials/01.jpg");
	Mat grayImage;
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);

	MatND hist;
	int bin = 256;
	int histSize[] = { bin };
	float range[] = { 0,255 };
	const float* ranges[] = { range };
	int channel = 0;
	/*void cv::calcHist(const Mat* images, int nimages, const int* channels,
		InputArray _mask, SparseMat& hist, int dims, const int* histSize,
		const float** ranges, bool uniform, bool accumulate)*/
	//calcHist(&srcImage, 1, channels_b, Mat(), blueHist, 1, hist_size, ranges, true, false);

	calcHist(&grayImage, 1, &channel, Mat(), hist, 1, histSize, ranges, true, false);
	Mat dstImage = Mat::zeros(bin, bin, CV_8UC3);
	double minValue = 0;
	double maxValue = 0;
	minMaxLoc(hist, &minValue, &maxValue, 0, 0);
	for (int i = 0; i < bin; i++)
	{
		float tmpValue = hist.at<float>(i);
		int realValue = saturate_cast<int>(tmpValue * bin / maxValue);
		rectangle(dstImage, Point(i, bin - 1), Point(i, bin - realValue), Scalar(255, 255, 255));
	}
	imshow("dstImage", dstImage);
#else  //图像的灰度直方图，直方图条的数量为200
#define HIST_SIZE (200)
#define IMAGE_WIDTH (800)
#define IMAGE_HEIGHT (800)
	Mat srcImage = imread("../TestMaterials/photo/1280_800_3.jpg");
	//Mat srcImage = imread("../TestMaterials/01.jpg");
	Mat grayImage;
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);

	MatND hist;
	int bin = HIST_SIZE;
	int histSize[] = { bin };
	float range[] = { 0,255 };
	const float* ranges[] = { range };
	int channel = 0;

	calcHist(&grayImage, 1, &channel, Mat(), hist, 1, histSize, ranges, true, false);
	Mat dstImage = Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
	double minValue = 0;
	double maxValue = 0;
	minMaxLoc(hist, &minValue, &maxValue, 0, 0);

	//每个条的宽度
	int perHistWidth = IMAGE_WIDTH / HIST_SIZE;
	for (int i = 0; i < HIST_SIZE; i++)
	{
		float tmpValue = hist.at<float>(i);
		int realValue = saturate_cast<int>(tmpValue * IMAGE_HEIGHT/maxValue);
		rectangle(dstImage, Point(i*perHistWidth, IMAGE_HEIGHT), Point((i+1)*perHistWidth, IMAGE_HEIGHT - realValue), Scalar(255, 255, 255));
	}
	imshow("dstImage", dstImage);

#endif

}

Mat ChapterFifth::hImage = Mat();
void ChapterFifth::TestCalcBackProject()
{
	namedWindow("TestCalcBackProject", CV_WINDOW_AUTOSIZE);
	createTrackbar("色调组距", "TestCalcBackProject", 0, 100, onBinValueChange, 0);
	Mat srcImage = imread("../TestMaterials/04.jpg");
	//srcImageClone = srcImage.clone();

	Mat hsvImage;
	cvtColor(srcImage, hsvImage, COLOR_BGR2HSV);

	hImage.create(hsvImage.size(), hsvImage.depth());
	int ch[] = { 0, 0 };
	mixChannels(&hsvImage, 1, &hImage, 1, ch, 1);

}

//如何判断不匹配
void ChapterFifth::TestMatchTemplate()
{
	Mat srcImage = imread("../TestMaterials/04.jpg");
	Mat templateImage = imread("../TestMaterials/1212.jpg");

	Mat resultImage;
	matchTemplate(srcImage, templateImage, resultImage, CV_TM_CCOEFF);

	double minValue;
	double maxValue;
	Point minLocation;
	Point maxLocation;
	minMaxLoc(resultImage, &minValue, &maxValue, &minLocation, &maxLocation, Mat());

	//此处的Rect的起始位置是由matchTemplate最后一个参数决定的，因为有的数值是越大越匹配
	//rectangle(srcImage, Rect(maxLocation.x, maxLocation.y, templateImage.cols, templateImage.rows), Scalar(150, 250, 180), 2, 8);
	rectangle(srcImage, Rect(maxLocation.x, maxLocation.y, templateImage.cols, templateImage.rows), Scalar(0, 0, 255), 2, 8);
	imshow("srcImage", srcImage);
}



