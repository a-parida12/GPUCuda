#include "library.h"
#include <stdio.h>
using namespace std;

int main(int argc, char **argv)
{
	printf("Inside main\n");
	cudaDeviceSynchronize();  CUDA_CHECK;
// input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }

//Load the input image using opencv
	cv::Mat mIn = cv::imread(image.c_str(), 1);

//convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
//convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
//get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
//    cout << "Input image dimensions: w * h: " << mIn.cols << " x " << mIn.rows << " and No. of Channels " <<mIn.channels()<<endl;
///////////////////////////////////////////////////////////////////////////////// Computation
// Kernel Definition		
 	
	int MK = 75;////////////////Put 75X35
	int NK = 35;////////////////////
	float lambda = 3.5e-4;

// gamma parameters
	ParamGammaCor params;
	params.gammaCorrection = 0;
	params.gamma = 0;
	params.iters = 1000;
	params.visualize = 1;
	
// coarse to fine parameters
	
	Param ctf_params;	
	ctf_params.lambdaMultiplier = 1.9;
	ctf_params.maxLambda = 1.1e-1;
	ctf_params.finalLambda = lambda;
	ctf_params.kernelSizeMultiplier = 1.1;

Timer mytimer;
mytimer.start();
coarseToFine(mIn, MK, NK, params, ctf_params, w, h, nc);

mytimer.end();

cout<<"Time:"<<mytimer.get()<<"s"<<endl;
///////////////////////////////////////////////////////////////////////////////////////// Post Processing
// show input image, save it
    int xImage = 100;
    int yImage = 100;   
    showImage("Blurry Input", mIn, xImage, yImage);  // show at position (x_from_left=100, y_from_above=100)
    cv::imwrite("Blurry Input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]

printf("\nThe End \n");

// wait for key inputs
    cv::waitKey(0);

// close all opencv windows
    cvDestroyAllWindows();
    return 0;
}
