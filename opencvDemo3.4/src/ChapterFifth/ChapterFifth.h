#pragma once
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
class ChapterFifth
{
private:
	static int loDiffValue ;
	static int upDiffValue;
	static int modeValue;
	static int thresholdType;
	static int thresholdValue;
	static int cannyHighValue;
	static int cannyLowValue;

	static void onAlphaChange(int pos,void *data);
	static void onContrastAndBright(int pos, void *data);
	static void onDilateValueChange(int pos, void *data);
	static void onErodeValueChange(int pos, void *data);
	static void onMorphologyExValueChange(int pos, void *data);
	static void onScalarLoDiffValueChange(int pos, void *data);
	static void onScalarUpDiffValueChange(int pos, void *data);
	static void onScalarModeValueChange(int pos, void *data);
	static void onThresholdTypeValueChange(int pos, void *data);
	static void onThresholdValueChange(int pos, void *data);
	static void onCannyHighValue(int pos, void *data);
	static void onCannyLowValue(int pos, void *data);
public:
	ChapterFifth() = default;
	~ChapterFifth() = default;
	void TestAddWeight();
	void TestSplit();
	void TestMerge();
	void TestBright();
	void TestDFT();
	void TestYaml();
	void TestDilate();
	void TestErode();
	void TestMorphologyEx();
	static void TestFloodFill();
	void TestPyrUp();
	void TestPyrDown();
	void TestResize();
	static void TestThreshold();
	void TestCanny();
	void TestLaplacian();
	void TestSobel();
	void TestScharr();
	void TestHoughLines();
};

