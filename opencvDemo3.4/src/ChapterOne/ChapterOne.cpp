#include "stdafx.h"
#include "ChapterOne.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;


ChapterOne::ChapterOne()
{
}


ChapterOne::~ChapterOne()
{
}

void ChapterOne::testErode()
{
	Mat srcImage = imread("G:\\TestMaterials\\05.png");
	imshow("src", srcImage);
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(6, 6));
	Mat dstImage;
	erode(srcImage, dstImage, element);
	imshow("dst", dstImage);
}

void ChapterOne::testImage()
{
	IplImage *src, *dst_blur, *dst_median, *dst_gaussian;
	//src = cvLoadImage("G:/TestMaterials/05.png", CV_LOAD_IMAGE_COLOR);
	src = cvLoadImage("G:/TestMaterials/gaosi.jpg", CV_LOAD_IMAGE_COLOR);
	//src = cvLoadImage("G:/TestMaterials/04.jpg", CV_LOAD_IMAGE_COLOR);
	CvSize srcSize(src->width, src->height);
	const CvArr* tmpTest = src;
	IplImage *srcTest = new IplImage();

	dst_blur = cvCreateImage(srcSize, IPL_DEPTH_8U, 3);
	dst_median = cvCreateImage(srcSize, IPL_DEPTH_8U, 3);
	dst_gaussian = cvCreateImage(srcSize, IPL_DEPTH_8U, 3);

	cvNamedWindow("src", 1);
	cvNamedWindow("blur", 1);
	cvNamedWindow("median", 1);
	cvNamedWindow("gaussian", 1);

	cvSmooth(src, dst_blur, CV_BLUR, 1, 1, 0, 0);			//邻域平均滤波
	cvSmooth(src, dst_median, CV_MEDIAN, 7, 8, 0, 0);		//中值滤波 
	cvSmooth(src, dst_gaussian, CV_GAUSSIAN, 4, 3, 0, 0);	//高斯滤波


	cvShowImage("src", src);
	cvShowImage("blur", dst_blur);
	cvShowImage("median", dst_median);
	cvShowImage("gaussian", dst_gaussian);

	cvWaitKey(0);
	cvReleaseImage(&src);
	cvReleaseImage(&dst_blur);
	cvReleaseImage(&dst_median);
	cvReleaseImage(&dst_gaussian);
}

void ChapterOne::TestCanny()
{
	//Mat srcImage = imread("G:\\TestMaterials\\05.png");
	Mat srcImage = imread("G:\\TestMaterials\\06.jpg");
	imshow("src", srcImage);
	//Mat dstImage;
	Mat edge;
	Mat grayImage;


	//dstImage.create(srcImage.size(), srcImage.type());
	//imshow("dstImage", dstImage);
	//cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);

	//blur(srcImage, srcImage, Size(3, 3));
	//高斯模糊中，size的宽高必须是奇数 
	GaussianBlur(srcImage, srcImage, Size(7, 7), 0, 0, BORDER_REPLICATE);
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);

	//imshow("grayImage COLOR_BGR2GRAY", grayImage);

	/*Mat grayImage1;
	cvtColor(srcImage, grayImage1, COLOR_RGB2GRAY);
	imshow("grayImage1 COLOR_RGB2GRAY", grayImage1);*/

	blur(grayImage, edge, Size(3, 3));
	//imshow("edge1", edge);
	Mat dstImage1;
	Mat dstImage2;
	
	//只支持单通道图片，因此需要将图片进行灰度转换
	Canny(edge, dstImage1, 3, 9, 3);
	imshow("edge", dstImage1);

}

void ChapterOne::TestCaptureVideo()
{
	VideoCapture video;
	Mat frame; 
	/*if (!video.open("G:\\TestStream\\f(x) - LA chA TA.avi"))
	{
		return;
	}*/

	if (!video.open("G:\\TestStream\\WeChat_20181223183047.mp4"))
	{
		return;
	}

	int frameWidth = video.get(CAP_PROP_FRAME_WIDTH);
	int frameHeight = video.get(CAP_PROP_FRAME_HEIGHT);
	int frameRate = video.get(CAP_PROP_FPS);
	int waitTime = int(1000.0 / frameRate);

	while (true)
	{
		if (!video.read(frame))
			break;

		imshow("video", frame);
		waitKey(waitTime);
	}

}

void ChapterOne::TestCaptureCamera()
{
	VideoCapture camera;
	Mat frame;
	Mat edges;

	if(!camera.open(0,CAP_DSHOW))
		return;

	if (!camera.isOpened())
		return;

	int frameHeight = camera.get(CAP_PROP_FRAME_HEIGHT);
	int frameWidth = camera.get(CAP_PROP_FRAME_WIDTH);
	int frameRate = camera.get(CAP_PROP_FPS);
	int frameFormat = camera.get(CAP_PROP_FORMAT);
	int waitTime = 1000 / 60;

	while (true)
	{
		if(!camera.read(frame))
			continue;

		cvtColor(frame, edges, CV_BGR2GRAY);
		blur(edges, edges, Size(7, 7));
		imshow("src", edges);

		Canny(edges, edges, 0, 30, 3);

		imshow("camera",edges);
		waitKey(waitTime);

	}

	camera.release();
	
}
