#include"library.h"

using namespace cv;
using namespace std;

void conv2(const Mat &img, const Mat& kernel, ConvolutionType type, Mat& dest) {

  	Mat source = img;

  	if(CONVOLUTION_FULL == type) {
    	source = Mat();
    	const int additionalRows = kernel.rows, additionalCols = kernel.cols;
    	copyMakeBorder(img, source, (additionalRows)/2, additionalRows/2, (additionalCols)/2, additionalCols/2, BORDER_CONSTANT, Scalar(0));
  	}
 
  	Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);
  
	int borderMode = BORDER_CONSTANT;
  
	Mat kernelflipped;
  
	
			flip(kernel,kernelflipped,-1);
		
			filter2D(source, dest, img.depth(), kernelflipped, anchor, 0, borderMode);

	//dest=GPUConvolve(source,kernel).clone();
	
	
	if(CONVOLUTION_VALID == type) {
    	dest = dest.colRange((kernel.cols)/2, dest.cols - kernel.cols/2)
               .rowRange((kernel.rows)/2, dest.rows - kernel.rows/2);
  	}

}
