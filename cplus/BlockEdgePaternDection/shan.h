#ifndef SHAN_H

#define SHAN_H
#include<opencv2\opencv.hpp>
#include<math.h>

using namespace cv;

// calculate entropy of an image
double Entropy(Mat img)
{
 // ������ľ���Ϊͼ��
 double temp[256];
 // ����
 for(int i=0;i<256;i++)
 {
  temp[i] = 0.0;
 }
 
  Mat_<uchar> t = img;

 // ����ÿ�����ص��ۻ�ֵ
 for(int m=0;m<img.rows;m++)
 {// ��Ч�������еķ�ʽ
  
  for(int n=0;n<img.cols;n++)
  {
   int i = t(m,n);
   temp[i] = temp[i]+1;
  }
 }
 // ����ÿ�����صĸ���
 for(int i=0;i<256;i++)
 {
  temp[i] = temp[i]/(img.rows*img.cols);
 }
 double result = 0;
 // ���ݶ������ͼ����
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
