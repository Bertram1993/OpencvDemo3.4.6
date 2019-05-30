// opencvDemo3.4.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "src/ChapterOne/ChapterOne.h"
using namespace std;
using namespace cv;

#pragma comment( linker, "/subsystem:windows /entry:mainCRTStartup" )






int main()
{
	ChapterOne TestChapterOne;
	TestChapterOne.TestCaptureCamera();
	waitKey(0);

	return 0;
}

