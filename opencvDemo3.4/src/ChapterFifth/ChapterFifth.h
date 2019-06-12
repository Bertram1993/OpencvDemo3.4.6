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
	static int houghLinesValue;
	static int houghLinesPValue;
	static int houghCircleValue;
	static int angleValue;
	static double scaleValue;
	static Point previousPoint;
	static Mat inpaintMaskImage;
	static Mat srcImageClone;
	static int binValue;
	static Mat hImage;


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
	static void onHoughLinesValueChange(int pos, void *data);
	static void onHoughLinesPValueChange(int pos, void *data);
	static void onHoughCircleValueChange(int pos, void *data);
	static void onAngleValueChange(int pos, void *data);
	static void onScaleValueChange(int pos, void *data);
	static void onMouse(int event, int x, int y, int flags, void* data); 
	static void onBinValueChange(int pos, void *data);

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
	static void TestHoughLines();
	static void TestHoughLinesP();
	static void TestHoughCircle();
	void TestReMap();
	void TestAffineMap();
	void TestEqualizeHist();
	void TestFindContours();
	void TestMinAreaRect();
	void TestMinEnclosingCircle();
	void TestContourArea();
	void TestWatershed();
	static void TestInpaint();
	void TestCalcHist();
	static void TestCalcBackProject();
	void TestMatchTemplate();


};

