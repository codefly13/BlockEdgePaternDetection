#ifndef edgePatern

#include <opencv.hpp>
#include <highgui\highgui.hpp>

using namespace cv;

void gen_edgePatern(Mat ** &ep)
{
	ep = new Mat*[5];

	uchar edge_gray = 180;
	float gamma = 25;

	Mat *NE = new Mat(8, 8, CV_8UC1);
	Mat_<uchar> &NE_ = (Mat_<uchar>&)(*NE);
	for(int i = 0; i<8; i++)
	{
		for(int j=0; j<8; j++)
		{
			NE_(i, j) = 255;
		}
	}

	Mat *E0 = new Mat();
	NE->copyTo(*E0);
	Mat_<uchar> &E0_ = (Mat_<uchar>&)(*E0);
	for(int i=0; i<8; i++)
	{
		E0_(3, i) = edge_gray;
		E0_(4, i) = edge_gray;
	}

	Mat *E45 = new Mat();
	NE->copyTo(*E45);
	Mat_<uchar> &E45_ = (Mat_<uchar>&)(*E45);
	for(int i=0; i<8; i++)
	{
		if(i == 0)
		{
			E45_(i, 8-i-2) = edge_gray;
			E45_(i, 8-i-1) = edge_gray;
		}
		else if(i == 7)
		{
			E45_(i, 0) = edge_gray;
			E45_(i, 1) = edge_gray;
		}
		else
		{
			E45_(i, 8-i-2) = edge_gray;
			E45_(i, 8-i-1) = edge_gray;
			E45_(i, 8-i) = edge_gray;
		}
	}

	Mat *E90 = new Mat();
	NE->copyTo(*E90);
	Mat_<uchar> &E90_ = (Mat_<uchar>&)(*E90);
	for(int i=0; i<8; i++)
	{
		E90_(i, 3) = edge_gray;
		E90_(i, 4) = edge_gray;
	}

	Mat *E135 = new Mat();
	NE->copyTo(*E135);
	Mat_<uchar> &E135_ = (Mat_<uchar>&)(*E135);
	for(int i=0; i<8; i++)
	{
		if(i == 0)
		{
			E135_(i, i) = edge_gray;
			E135_(i, i+1) = edge_gray;
		}
		else if(i == 7)
		{
			E135_(i, 6) = edge_gray;
			E135_(i, 7) = edge_gray;
		}
		else
		{
			E135_(i, i-1) = edge_gray;
			E135_(i, i) = edge_gray;
			E135_(i, i+1) = edge_gray;
		}
	}


	ep[0] = NE;
	ep[1] = E0;
	ep[2] = E45;
	ep[3] = E90;
	ep[4] = E135;
}

#endif // !1
