此工程是自己学习opencv练习的demo,程序中比较难理解的部分有注释。

使用方法：

1、将该工程git clone 到本地上

2、使用visual studio 2015 打开

3、通过修改opencvDemo3.4.cpp文件中的main函数，选择修改为自己想要测试运行的功能

4、运行工程

eg:
若想测试显示图片的灰度直方图功能，在main函数中修改如下：

int main()
{

	ChapterFifth TestChapterThird;
	TestChapterThird.TestCalcHist();
	
	waitKey(0);

	return 0;
}

然后运行工程



PS：每个类里面以Test命名为开头代表一个功能，通过修改main函数后可单独运行


