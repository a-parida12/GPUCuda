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

__global__ void histogram256(float* in, int* hist, int w, int h, int nc){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;
    int idz = threadIdx.z + blockIdx.z*blockDim.z;

    float intensity = 0;

    if (idx+idy*w< w*h){
       for (int j =0; j < nc; j++){
            intensity += in[idx + w*idy + j*w*h]; 
        }
	intensity = intensity*(255.0/nc);   
     
        atomicAdd( &(hist[(int)intensity]), 1);  
    }
}

__global__ void histogram256_shared(float* in, int* hist, int w, int h, int nc){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;
    int idz = threadIdx.z + blockIdx.z*blockDim.z;

	__shared__ int hist_shared[256];

	 if ( threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
	for(int i=0 ; i<256 ; i++){
        	hist_shared[i] = 0;
			}
    }
	__syncthreads();
	
    float intensity = 0;

    if (idx+idy*w< w*h){
       for (int j =0; j < nc; j++){
            intensity += in[idx + w*idy + j*w*h]; 
        }
	intensity = intensity*(255.0/nc);   
     
        atomicAdd( &(hist_shared[(int)intensity]), 1);  
    }

	__syncthreads();

	if ( threadIdx.x == 0 && threadIdx.y == 0 ){
		for(int i=0 ; i<256 ; i++){
				//printf("%d\n",hist_shared[i]);
			 	atomicAdd( &(hist[i]), hist_shared[i]);        	
		}
    }
	__syncthreads();	
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
    //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
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
    // ### Atomic Histogram Implementation
	
	float* g_imgIn;
    int* g_hist256;
	
    cudaMalloc(&g_imgIn , w*h*nc*sizeof(float));
    cudaMalloc(&g_hist256 , 256*sizeof(int));
	
	cudaMemset(g_hist256,0,256*sizeof(int));
    cudaMemcpy(g_imgIn, imgIn , w*h*nc*sizeof(float) , cudaMemcpyHostToDevice);


    dim3 block = dim3(32,32,1);
    dim3 grid = dim3((w + block.x + 1)/block.x, (h + block.y + 1)/block.y, 1 );

    Timer timer;
    timer.start();
    histogram256<<<grid, block>>>(g_imgIn, g_hist256 , w, h, nc);
    timer.end();  
    float t = timer.get();  // elapsed time in seconds
    cout << "time global atomic: " << t*1000 << " ms" << endl;
   
    int* hist_global = new int[256];
    cudaMemcpy(hist_global, g_hist256, 256*sizeof(int), cudaMemcpyDeviceToHost);
    showHistogram256("Histogram Gloabal Memory" , hist_global, 100 + w, 100); 
	
	cudaMemset(g_hist256,0,256*sizeof(int));
	
	Timer timer1;
    timer1.start();
    histogram256_shared<<<grid, block>>>(g_imgIn, g_hist256 , w, h, nc);
    timer1.end();  
    float t1 = timer1.get();  // elapsed time in seconds
    cout << "time share atomic: " << t1*1000 << " ms" << endl;
   	int* hist_shared= new int[256];
    cudaMemcpy(hist_shared, g_hist256, 256*sizeof(int), cudaMemcpyDeviceToHost);
    showHistogram256("Histogram Shared Memory" , hist_shared, 100 + w, 250); 
	    

	// ### 
    // ###
    // ###
    






    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    //convert_layered_to_mat(mOut, imgOut);
   // showImage("Output", mOut, 100+w+40, 100);

    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif




    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



