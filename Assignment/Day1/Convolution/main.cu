
// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2017, September 11 - October 9
// ###

#include "helper.h"
#include <iostream>
#include <stdio.h>
using namespace std;

// uncomment to use the camera
//#define CAMERA

__global__ void convoluteGPU (float *in, float *out, int w, int h, int nc, float *kernel, int kernelRadius)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein
	//printf("thread id check\n");
	int kernelWidth = 2 * kernelRadius + 1;
    
    int currentlocation = iz*w*h + ix + iy * w;
    out[currentlocation]=0;
	//printf("current location check\n");
	if (ix < w && iy <h && iz < nc){ 

		for (int x=0; x<kernelWidth; x++) {
			for(int y=0; y<kernelWidth; y++) {

				int cx = max(min(w-1, ix + x - kernelRadius), 0);
				int cy = max(min(h-1, iy + y - kernelRadius), 0);

				out[currentlocation] += kernel[x+y*kernelWidth] * in[iz*w*h+cx+cy*w];
			}
		}
	}
}


int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;




    // Reading command line parameters:
    // getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
    // If "-param" is not specified, the value of "var" remains unchanged
    //
    // return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise

#ifdef CAMERA
#else
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }
#endif
    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed    

    // Init camera / Load input image
#ifdef CAMERA

    // Init camera
  	cv::VideoCapture camera(0);
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    cv::Mat mIn;
    camera >> mIn;
    
#else
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
    
#endif

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    cout << "image: " << w << " x " << h << endl;




    // Set the output image format
    // ###
    // ###
    // ### TODO: Change the output image format as needed
    // ###
    // ###
    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    //cv::Mat mOut_gray(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed




    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn = new float[(size_t)w*h*nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
	float *imgOut = new float[(size_t)w*h*mOut.channels()];
	float *imgOutConvolutedCPU = new float[(size_t)w*h*mOut.channels()];
	
	    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (cv::waitKey(30) < 0)
    {
    // Get camera image
    camera >> mIn;
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
#endif

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);

    
    // ###
    // ###
    // ### TODO: Main computation
    // ###
    // ###

	// create kernel	
	
	
    float sigma=1.0;


	int radius_kernel=ceil(3*sigma);
	int width_kernel=2*radius_kernel+1;
	float sum=0.0;
	float sigmasquare_x2 = 2.0 * sigma * sigma;
	float gKernel[width_kernel*width_kernel];
	float mKernel=0.0;

	for (int x=0;x<width_kernel;x++)
	{
	 	for(int y=0;y<width_kernel;y++)
		{
			int a= x-radius_kernel;
			int b= y-radius_kernel;
            gKernel[x+y*width_kernel] = expf(-(a*a+b*b)/sigmasquare_x2)/(M_PI * sigmasquare_x2);
            sum += gKernel[x+y*width_kernel];
			
			if(gKernel[x+y*width_kernel] > mKernel){
				mKernel = gKernel[x+y*width_kernel];
			}
			
        }													
	
	}
	float copy_gKernel[width_kernel*width_kernel];
    for(int i = 0; i < width_kernel; ++i)
    {
        for (int j = 0; j < width_kernel; ++j)
            {
			gKernel[i+j*width_kernel]/=sum;
			//cout<<gKernel[i+j*width_kernel]<<"\t";
			copy_gKernel[i+j*width_kernel]=gKernel[i+j*width_kernel]/gKernel[width_kernel/2+(width_kernel/2)*width_kernel];
        }
		//cout<<endl;
    }

	 cv::Mat mOutKernel(width_kernel,width_kernel,CV_32FC1);
    
    convert_layered_to_mat(mOutKernel, copy_gKernel);
    showImage("Gaussian Kernel", mOutKernel, 250, 100);
	

// apply convolution with clamping
	Timer timer; timer.start();
	for(int c=0; c<nc; c++) {
		

		for (int ix=0; ix<w; ix++) {
		    for(int iy=0; iy<h; iy++) {

				int currentlocation = h*w*c +ix+iy*w;

		    	for (int x=0; x<width_kernel; x++) {
					for(int y=0; y<width_kernel; y++) {

						//clamping strategy
					    int cx = max(min(w-1, ix + x - radius_kernel), 0);
					    int cy = max(min(h-1, iy + y - radius_kernel), 0);

						imgOutConvolutedCPU[currentlocation] += gKernel[x+y*width_kernel] * imgIn[h*w*c+cx+cy*w];
	//cout<<imgOutConvolutedCPU[currentlocation]<<endl;
					}
				}
			}
		}
	}
	
	 timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time on CPU: " << t*1000 << " ms" << endl;

    convert_layered_to_mat(mOut, imgOutConvolutedCPU);
    showImage("Output Convoluted CPU", mOut, 150, 100);
	
   //--Init for Cuda kernel call

	float *imgOutConvolutedGPU = new float[(size_t)w*h*nc];
    
    float *g_imgIn;
    float *g_imgOut;
    float *g_gKernel;
    
    cudaMalloc( &g_imgIn, w*h*nc * sizeof(float) );CUDA_CHECK;
    cudaMalloc( &g_imgOut, w*h*nc * sizeof(float) );CUDA_CHECK;
    cudaMalloc( &g_gKernel, width_kernel * width_kernel * sizeof(float) );CUDA_CHECK;
    
    cudaMemcpy( g_imgIn, imgIn, w*h*nc * sizeof(float), cudaMemcpyHostToDevice );CUDA_CHECK; 
    cudaMemcpy( g_gKernel, gKernel, width_kernel * width_kernel * sizeof(float), cudaMemcpyHostToDevice );CUDA_CHECK;
    
    dim3 Block = dim3(32,32,1);
    dim3 Grid = dim3((w +Block.x -1) / Block.x, (h + Block.y -1) / Block.y, (nc+ Block.z -1) / Block.z);
    				
	//call cuda kernel for convolution

	Timer timer1; timer1.start();
	convoluteGPU <<<Grid,Block>>> (g_imgIn, g_imgOut, w, h, nc, g_gKernel, radius_kernel);CUDA_CHECK;
    timer1.end();  t = timer1.get();  // elapsed time in seconds
    cout << "time on GPU: " << t*1000 << " ms" << endl;

	//copy output gpu->cpu
    cudaMemcpy(imgOutConvolutedGPU,g_imgOut, nc*h*w * sizeof(float), cudaMemcpyDeviceToHost );
    CUDA_CHECK;
    
    //free gpu allocation
    cudaFree(g_imgOut);
    CUDA_CHECK;
    cudaFree(g_imgIn);
    CUDA_CHECK;
    cudaFree(g_gKernel);
    CUDA_CHECK;

	convert_layered_to_mat(mOut, imgOutConvolutedGPU);
    showImage("Output Convoluted GPU", mOut, 200, 100);

	convert_layered_to_mat(mOut, imgIn);
    showImage("Input Image", mOut, 250, 100);

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif




    // save input and result
    //cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    //cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



