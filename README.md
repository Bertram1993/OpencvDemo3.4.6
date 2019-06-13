此工程是自己学习opencv时练习的demo，程序中比较难理解的部分有注释。

使用方法：

1、将该工程git clone到本地

2、使用visual studio 2015打开

3、通过修改opencvDemo3.4.cpp文件中的main函数，选择修改为自己想要测试运行的功能函数

4、运行工程



eg:

若想测试显示图片的灰度直方图功能，在main函数中修改如下：

int main)

{

​	ChapterFifth TestChapterThird;

   TestChapterThird.TestCalcHist();

   waitkey(0);

  return 0;

}

然后运行工程

效果图如下：

![avatar](https://github.com/Bertram1993/OpencvDemo3.4.6/blob/master/TestMaterials/灰色直方图效果图.jpg)

PS：每个类里面以Test前缀命名的函数代表一个测试功能，，通过修改main函数后可单独运行

