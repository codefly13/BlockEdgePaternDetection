#include<iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include "edgePatern.h"
#include "shan.h"

using namespace cv;  

Mat edge_measure(Mat img, float gamma)
{
	float s[4] = {0,0,0,0};

	for(int i=0; i<2; i++)
	{
		for(int j=0; j<2; j++)
		{
			s[i*2+j] = cv::mean(img(Rect(j*4,i*4,4,4)))[0];
			//std::cout<<"("<<i<<","<<j<<")"<<img(Rect(j*4,i*4,4,4))<<std::endl;
		}
	}

	//printf("mean : %f %f %f %f\n", s[0], s[1], s[2], s[3]);

	float theta0 = abs((s[0] + s[1]) - (s[2] + s[3])) / 2.0;
	float theta90 = abs((s[0] + s[2]) - (s[1] + s[3])) / 2.0;
	float theta45 = max(abs(s[0] - (s[1] + s[2] + s[3])/3.0), abs(s[3] - (s[0] + s[1] + s[2])/3.0));
	float theta135 = max(abs(s[1] - (s[0] + s[2] + s[3])/3.0), abs(s[2] - (s[0] + s[1] + s[3])/3.0));
	// thetaNE = gamma*abs((s[0] + s[3]) - (s[1]+s[2]))/2
	float thetaNE = gamma;
	Mat edgeV = (Mat_<float>(1,5) << thetaNE, theta0, theta45, theta90, theta135);
	//printf("%f %f %f %f %f\n", thetaNE, theta0, theta45, theta90, theta135);
	return edgeV;
}



Mat edge_measure_dct(Mat img, float gamma, int choose)
{

	Mat roi_dct;
	cv::dct(img, roi_dct);
	Mat_<float> roi_ = roi_dct;

	float theta0 = abs(roi_(1, 0));
	float theta90 = abs(roi_(0, 1));
	float theta45 = 4.0/3*max(abs(roi_(1, 0)/2.0+roi_(0, 1)/2.0+roi_(1, 1)/2.0), abs(roi_(1, 1)/2.0-roi_(0, 1)/2.0-roi_(1, 0)/2.0));
	float theta135 = 4.0/3*max(abs(roi_(1, 0)/2.0-roi_(0, 1)/2.0-roi_(1, 1)/2.0), abs(roi_(0, 1)/2.0-roi_(1, 0)/2.0-roi_(1, 1)/2.0));
	float thetaNE = 0;
	switch (choose)
	{
	case 0:
		thetaNE = gamma;
		break;
	case 1:
		// 指数形式
		thetaNE = exp(-abs(roi_(1, 1))/4.0)*gamma;
		break;
	case 2:
		// sigmod函数形式的
		thetaNE = (1.5-1.0/(1+exp(-abs(roi_(1, 1)/4.0))))*gamma;
		break;
	case 3:
		// shan
		thetaNE = (1.5-1.0/(1+exp(-Entropy(img))))*gamma;
		break;
	case 4:
		// shan
		thetaNE = exp(-Entropy(img))*gamma;
	case 5:
		// shan
		thetaNE = (1.5-1.0/(1+exp(-(0.5*Entropy(img) + 0.5*roi_(1, 1)/4.0))))*gamma;
	default:
		break;
	}
	Mat edgeV = (Mat_<float>(1,5) << thetaNE, theta0, theta45, theta90, theta135);
	//std::cout<<edgeV<<std::endl;
	return edgeV;
}




int main()  
{  
	Mat **ep = NULL;
	gen_edgePatern(ep);

	namedWindow("NE", WINDOW_NORMAL);
	namedWindow("E0", WINDOW_NORMAL);
	namedWindow("E45", WINDOW_NORMAL);
	namedWindow("E90", WINDOW_NORMAL);
	namedWindow("E135", WINDOW_NORMAL);
	imshow("NE", *ep[0]);
	imshow("E0", *ep[1]);
	imshow("E45", *ep[2]);
	imshow("E90", *ep[3]);
	imshow("E135", *ep[4]);

	//Mat im = imread("Lena_512512.bmp");
	char imgfilename[256];
	for (int th= 12; th<13; th++)
	{
		sprintf(imgfilename, "test%d.jpg", th);
		Mat im = imread(imgfilename);
		cvtColor(im, im, CV_RGB2GRAY);
		imshow("src", im);
		//waitKey(0);
		Mat im_bep = Mat::zeros(im.rows, im.cols, CV_8UC1);


		for(int gamma=1; gamma<4096*8; gamma*=2)
		{
			Mat im_bep1 = Mat::zeros(im.rows, im.cols, CV_8UC1);
			Mat im_bep2 = Mat::zeros(im.rows, im.cols, CV_8UC1);
			Mat im_bep3 = Mat::zeros(im.rows, im.cols, CV_8UC1);
			Mat im_bep4 = Mat::zeros(im.rows, im.cols, CV_8UC1);
			Mat im_bep5 = Mat::zeros(im.rows, im.cols, CV_8UC1);
			Mat im_bep6 = Mat::zeros(im.rows, im.cols, CV_8UC1);
			char filename1[256], filename2[256], filename3[256], filename4[256], filename5[256], filename6[256];
			for(int i= 0; i<(im.rows-4)/8; i++)
			{
				for(int j=0; j<(im.cols-4)/8; j++)
				{
					Point2i maxIdx6,maxIdx5,maxIdx4,maxIdx3,maxIdx2,maxIdx1;
					Mat roi = im(Rect(j * 8+4, i * 8+4, 8, 8));
					//Mat edge_value = edge_measure(roi, 15);
					Mat edge_value1 = edge_measure_dct(Mat_<double>(roi), gamma, 0);
					Mat edge_value2 = edge_measure_dct(Mat_<double>(roi), gamma, 1);
					Mat edge_value3 = edge_measure_dct(Mat_<double>(roi), gamma, 2);
					Mat edge_value4 = edge_measure_dct(Mat_<double>(roi), gamma, 3);
					Mat edge_value5 = edge_measure_dct(Mat_<double>(roi), gamma, 4);
					Mat edge_value6 = edge_measure_dct(Mat_<double>(roi), gamma, 5);

					minMaxLoc(edge_value1, NULL, NULL, NULL, &maxIdx1);
					//std::cout<<edge_value1<<std::endl;
					minMaxLoc(edge_value2, NULL, NULL, NULL, &maxIdx2);
					minMaxLoc(edge_value3, NULL, NULL, NULL, &maxIdx3);
					minMaxLoc(edge_value4, NULL, NULL, NULL, &maxIdx4);
					//std::cout<<edge_value4<<std::endl;
					minMaxLoc(edge_value5, NULL, NULL, NULL, &maxIdx5);
					minMaxLoc(edge_value6, NULL, NULL, NULL, &maxIdx6);
					//printf("%d %d %d %d\n", maxIdx.x, maxIdx2.x, maxIdx3.x, maxIdx4.x);

					//im_bep(Rect(j * 8, i * 8, 8, 8)) = *ep[maxIdx.x];
					(*ep[maxIdx1.x]).copyTo(im_bep1(Rect(j * 8+4, i * 8+4, 8, 8)));
					(*ep[maxIdx2.x]).copyTo(im_bep2(Rect(j * 8+4, i * 8+4, 8, 8)));
					(*ep[maxIdx3.x]).copyTo(im_bep3(Rect(j * 8+4, i * 8+4, 8, 8)));
					(*ep[maxIdx4.x]).copyTo(im_bep4(Rect(j * 8+4, i * 8+4, 8, 8)));
					(*ep[maxIdx5.x]).copyTo(im_bep5(Rect(j * 8+4, i * 8+4, 8, 8)));
					(*ep[maxIdx6.x]).copyTo(im_bep6(Rect(j * 8+4, i * 8+4, 8, 8)));

				}

				//namedWindow("t1", WINDOW_NORMAL);
				//imshow("t1", *ep[maxIdx.x]);
				//namedWindow("t2", WINDOW_NORMAL);
				//imshow("t2", im_bep);
				//waitKey(0);
			}


			sprintf(filename1, "./result/test%d/td_g%d.jpg", th, gamma);
			sprintf(filename2, "./result/test%d/ex_g%d.jpg", th, gamma);
			sprintf(filename3, "./result/test%d/sig_g%d.jpg", th, gamma);
			sprintf(filename4, "./result/test%d/shan_g%d.jpg", th, gamma);
			sprintf(filename5, "./result/test%d/shanex_g%d.jpg", th, gamma);
			sprintf(filename6, "./result/test%d/shanb11_g%d.jpg", th, gamma);

			printf(filename2);
			printf("\n");
			imwrite(filename1, im_bep1);
			imwrite(filename2, im_bep2);
			imwrite(filename3, im_bep3);
			imwrite(filename4, im_bep4);
			imwrite(filename5, im_bep5);
			imwrite(filename6, im_bep6);
		}
	}
	//imshow("bep", im_bep);
	//imshow("bep2", im_bep2);
	//imshow("bep3", im_bep3);
	//imshow("bep4", im_bep4);
	//waitKey(0);  
	delete ep;
	ep = NULL;
}  
