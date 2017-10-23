#include"library.h"
using namespace std;

cv::Mat GetGradK(cv::Mat u,cv::Mat k,cv::Mat f){
	
	cv::Mat gradk(k.size(),k.type());
	gradk=0.0;	
	cv::Mat k_flip,u_flip;
	cv::Mat ConvolutedU,ConvOperand;
	cv::Mat Gradient3Channel;
	cv::Mat ConvOperand_channel[3],u_flip_channel[3],gradk_channel[3];

	conv2(u,k,CONVOLUTION_VALID,ConvolutedU);

	ConvOperand=ConvolutedU-f;
	
	cv::flip(u,u_flip,-1);
	
	cv::split(u_flip, u_flip_channel);
	cv::split(ConvOperand, ConvOperand_channel);

	for(int i=0;i<3;i++)
		conv2(u_flip_channel[i],ConvOperand_channel[i],CONVOLUTION_VALID,gradk_channel[i]);

	gradk=gradk_channel[0]+gradk_channel[1]+gradk_channel[2];

	return gradk;
}

