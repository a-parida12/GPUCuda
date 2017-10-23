#ifndef LIBRARY_H
#define LIBRARY_H
#include"helper.h"
#include<iostream>
#include<math.h>
struct ParamGammaCor
{
	int gammaCorrection;
	int gamma;
	int iters;
	int visualize;
};

struct Param
{
	float lambdaMultiplier;
	float maxLambda;
	float finalLambda;
	float kernelSizeMultiplier;
};
struct ReturnBuildPyramid //Review Return types
{
	cv::Mat* imArray; //Pointer to the array of matrices
	std::vector<int> hp; //Mp
	std::vector<int> wp; //Np;
	std::vector<int> hKp; //MKp;
	std::vector<int> wKp; //NKp;
	std::vector<float> lambdas;
	int scales;
};
struct BlindReturn{
	cv::Mat k;
	cv::Mat u;
};
 
enum ConvolutionType {   
  	CONVOLUTION_FULL, 
  	CONVOLUTION_SAME,
	CONVOLUTION_VALID
};

BlindReturn blind(cv::Mat f, float lambda, cv::Mat u, cv::Mat k, int scale, int iter);
ReturnBuildPyramid buildPyramid (int hK, int wK, float lambda, float lambdaMultiplier, int w, int h, int nc, float scaleMultiplier = 1.1, float largestLambda = 0.11);
void coarseToFine (cv::Mat f, int MK, int NK, ParamGammaCor blind_params, Param params, int w, int h, int nc);
void conv2(const cv::Mat &img, const cv::Mat& kernel, ConvolutionType type, cv::Mat& dest);
cv::Mat dec(cv::Mat f, float lambda, cv::Mat u, cv::Mat k, int scale, int iter);
cv::Mat GetGradK(cv::Mat u,cv::Mat k,cv::Mat f);
cv::Mat GPUConvolve(cv::Mat,cv::Mat);
cv::Mat KernelProjection(cv::Mat k);
float MaxFinder(cv::Mat);
cv::Mat NonConvexOptimization(cv::Mat u,cv::Mat k,cv::Mat f, float lambda);
cv::Mat scalarMult(cv::Mat, float);
cv::Mat SubtractAB(cv::Mat A,cv::Mat B);

cv::Mat gradTVcc(cv::Mat f, float epsilon = 0.001);



#endif  // LIBRARY_H
