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


Mat edge_measure_dct(Mat img, float gamma)
{
	//std::cout<<img<<std::endl<<std::endl;
	Mat roi_dct;
	cv::dct(img, roi_dct);
	Mat_<float> roi_ = roi_dct;

	//std::cout<<roi_dct<<std::endl<<std::endl;

	float theta0 = abs(roi_(1, 0)) / 4.0;
    float theta90 = abs(roi_(0, 1)) / 4.0;
    float theta45 = 1.0/6*max(abs(roi_(1, 0)+roi_(0, 1)+roi_(1, 1)), abs(roi_(1, 1)-roi_(0, 1)-roi_(1, 0)));
    float theta135 = 1.0/6*max(abs(roi_(1, 0)-roi_(0, 1)-roi_(1, 1)), abs(roi_(0, 1)-roi_(1, 0)-roi_(1, 1)));
    // thetaNE = gamma*abs(roi_dct.at<float>(1, 1));
    float thetaNE = gamma;

	Mat edgeV = (Mat_<float>(1,5) << thetaNE, theta0, theta45, theta90, theta135);
    
	//std::cout<<edgeV<<std::endl;
	return edgeV;
}


Mat edge_measure3_dct(Mat img, float gamma)
{
	//std::cout<<img<<std::endl<<std::endl;
	Mat roi_dct;
	cv::dct(img, roi_dct);
	Mat_<float> roi_ = roi_dct;

	//std::cout<<roi_dct<<std::endl<<std::endl;

	float theta0 = abs(roi_(1, 0)) / 4.0;
    float theta90 = abs(roi_(0, 1)) / 4.0;
    float theta45 = 1.0/6*max(abs(roi_(1, 0)+roi_(0, 1)+roi_(1, 1)), abs(roi_(1, 1)-roi_(0, 1)-roi_(1, 0)));
    float theta135 = 1.0/6*max(abs(roi_(1, 0)-roi_(0, 1)-roi_(1, 1)), abs(roi_(0, 1)-roi_(1, 0)-roi_(1, 1)));
    // thetaNE = gamma*abs(roi_dct.at<float>(1, 1));
    //float thetaNE = gamma;
	float thetaNE = (1.5-1.0/(1+exp(-Entropy(img))))*gamma;
	Mat edgeV = (Mat_<float>(1,5) << thetaNE, theta0, theta45, theta90, theta135);
    
	//std::cout<<edgeV<<std::endl;
	return edgeV;
}


Mat edge_measure2_dct(Mat img, float gamma)
{
	//std::cout<<img<<std::endl<<std::endl;
	Mat roi_dct;
	cv::dct(img, roi_dct);
	Mat_<float> roi_ = roi_dct;

	//std::cout<<roi_dct<<std::endl<<std::endl;

	float theta0 = abs(roi_(1, 0)) / 4.0;
    float theta90 = abs(roi_(0, 1)) / 4.0;
    float theta45 = 1.0/6*max(abs(roi_(1, 0)+roi_(0, 1)+roi_(1, 1)), abs(roi_(1, 1)-roi_(0, 1)-roi_(1, 0)));
    float theta135 = 1.0/6*max(abs(roi_(1, 0)-roi_(0, 1)-roi_(1, 1)), abs(roi_(0, 1)-roi_(1, 0)-roi_(1, 1)));
    // thetaNE = gamma*abs(roi_dct.at<float>(1, 1));
    //float thetaNE = gamma;
	float thetaNE = (1.5-1.0/(1+exp(-roi_(1, 1))))*gamma;
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

	Mat im = imread("Lena_512512.bmp");
	cvtColor(im, im, CV_RGB2GRAY);
	imshow("src", im);
	Mat im_bep = Mat::zeros(im.rows, im.cols, CV_8UC1);


	for(int gamma=5; gamma<80; gamma+=5)
	{
		Mat im_bep2 = Mat::zeros(im.rows, im.cols, CV_8UC1);
		Mat im_bep3 = Mat::zeros(im.rows, im.cols, CV_8UC1);
		Mat im_bep4 = Mat::zeros(im.rows, im.cols, CV_8UC1);
		char filename2[256], filename3[256], filename4[256];
		for(int i=0; i<im.rows/8; i++)
		{
			for(int j=0; j<im.cols/8; j++)
			{
					Point2i maxIdx4,maxIdx3,maxIdx2,maxIdx;
					Mat roi = im(Rect(j * 8, i * 8, 8, 8));
					//Mat edge_value = edge_measure(roi, 15);
					Mat edge_value2 = edge_measure_dct(Mat_<double>(roi), gamma);
					Mat edge_value3 = edge_measure2_dct(Mat_<double>(roi), gamma);
					Mat edge_value4 = edge_measure3_dct(Mat_<double>(roi), gamma);

					//minMaxLoc(edge_value, NULL, NULL, NULL, &maxIdx);
					minMaxLoc(edge_value2, NULL, NULL, NULL, &maxIdx2);
					minMaxLoc(edge_value3, NULL, NULL, NULL, &maxIdx3);
					minMaxLoc(edge_value4, NULL, NULL, NULL, &maxIdx4);
					//printf("%d %d %d %d\n", maxIdx.x, maxIdx2.x, maxIdx3.x, maxIdx4.x);

					//im_bep(Rect(j * 8, i * 8, 8, 8)) = *ep[maxIdx.x];
					//(*ep[maxIdx.x]).copyTo(im_bep(Rect(j * 8, i * 8, 8, 8)));
					(*ep[maxIdx2.x]).copyTo(im_bep2(Rect(j * 8, i * 8, 8, 8)));
					(*ep[maxIdx3.x]).copyTo(im_bep3(Rect(j * 8, i * 8, 8, 8)));
					(*ep[maxIdx4.x]).copyTo(im_bep4(Rect(j * 8, i * 8, 8, 8)));

				}

				//namedWindow("t1", WINDOW_NORMAL);
				//imshow("t1", *ep[maxIdx.x]);
				//namedWindow("t2", WINDOW_NORMAL);
				//imshow("t2", im_bep);
				//waitKey(0);
			}

		sprintf(filename2, "./result/bep2_g%d.jpg", gamma);
		sprintf(filename3, "./result/bep3_g%d.jpg", gamma);
		sprintf(filename4, "./result/bep4_g%d.jpg", gamma);

		printf(filename2);
		printf("\n");
		imwrite(filename2, im_bep2);
		imwrite(filename3, im_bep3);
		imwrite(filename4, im_bep4);
	}

	//imshow("bep", im_bep);
	//imshow("bep2", im_bep2);
	//imshow("bep3", im_bep3);
	//imshow("bep4", im_bep4);
    waitKey(0);  
	delete ep;
	ep = NULL;
}  
