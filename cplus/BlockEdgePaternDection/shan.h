#ifndef SHAN_H

#define SHAN_H
#include<opencv2\opencv.hpp>
#include<math.h>

using namespace cv;

// calculate entropy of an image
double Entropy(Mat img)
{
 // 将输入的矩阵为图像
 double temp[256];
 // 清零
 for(int i=0;i<256;i++)
 {
  temp[i] = 0.0;
 }
 
  Mat_<uchar> t = img;

 // 计算每个像素的累积值
 for(int m=0;m<img.rows;m++)
 {// 有效访问行列的方式
  
  for(int n=0;n<img.cols;n++)
  {
   int i = t(m,n);
   temp[i] = temp[i]+1;
  }
 }
 // 计算每个像素的概率
 for(int i=0;i<256;i++)
 {
  temp[i] = temp[i]/(img.rows*img.cols);
 }
 double result = 0;
 // 根据定义计算图像熵
 for(int i =0;i<256;i++)
 {
  if(temp[i]==0.0)
   result = result;
  else
   result = result-temp[i]*(log(temp[i])/log(2.0));
 }
 return result; 
}


#endif
