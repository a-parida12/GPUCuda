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
using namespace std;

// uncomment to use the camera
//#define CAMERA

__device__ float square(float a){return a*a;}

__global__ void cuda_gradient(float* d_imgIn,float* d_imgOut_gradx,float* d_imgOut_grady, int n, int h, int w)
{
 int idx = threadIdx.x + blockDim.x * blockIdx.x;
 		
 
 if(idx+1<n && (idx+1)%w!=0){
		d_imgOut_gradx[idx]=d_imgIn[idx+1]-d_imgIn[idx];
		
	}

	else {
		d_imgOut_gradx[idx]=0;//d_imgIn[idx];
		}
 if (idx+w <n){
	d_imgOut_grady[idx]=d_imgIn[idx+w]-d_imgIn[idx];
	}

 else {
		d_imgOut_grady[idx]=0;}//d_imgIn[idx];}

}

__global__ void cuda_divergence(float* d_imgOut_gradx,float* d_imgOut_grady, float* d_imgOut_div,int n, int h, int w)
{
 int idx = threadIdx.x + blockDim.x * blockIdx.x;
 		
 
 if(idx+1<n && (idx+1)%w!=0){
		d_imgOut_div[idx]=d_imgOut_gradx[idx+1]-d_imgOut_gradx[idx];
}
 else 
		d_imgOut_div[idx]=d_imgOut_gradx[idx];

 if(idx+w <n){
		d_imgOut_div[idx]+=(d_imgOut_grady[idx+w]-d_imgOut_grady[idx]);
}
 else 
		d_imgOut_div[idx]+=d_imgOut_grady[idx];
}

__global__ void cuda_l2(float* d_imgIn,float* d_imgOut_l2, int n, int h, int w,int nc)
{
 int idx = threadIdx.x + blockDim.x * blockIdx.x;
 		
 
 if(idx<n/nc){
		d_imgOut_l2[idx]=sqrtf(square(d_imgIn[idx])+square(d_imgIn[h*w+idx])+square(d_imgIn[2*h*w+idx]));
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
    cv::Mat mOut(h,w,mIn.type()); 
	
	 // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    cv::Mat mOut_gray(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed




    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn = new float[(size_t)w*h*nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
	float *imgOut_l2 = new float[(size_t)w*h*mOut.channels()];
    float *imgOut_gradx = new float[(size_t)w*h*mOut.channels()];
	float *imgOut_grady = new float[(size_t)w*h*mOut.channels()];
	
	float *imgOut_div = new float[(size_t)w*h*mOut.channels()];
	

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

	//----GPU----//
	float* d_imgIn,*d_imgOut_gradx, *d_imgOut_grady, *d_imgOut_div,*d_imgOut_l2;
	
	cudaMalloc(&d_imgIn,h*w*nc*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_imgOut_l2,h*w*nc*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_imgOut_gradx,h*w*nc*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_imgOut_grady,h*w*nc*sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_imgOut_div,h*w*nc*sizeof(float));
	CUDA_CHECK;
	
		

	cudaMemcpy( d_imgIn, imgIn, h*w*nc* sizeof(float), cudaMemcpyHostToDevice );CUDA_CHECK;
	
	dim3 block = dim3(256,1,1);
	dim3 grid = dim3(((h*w*nc)+block.x-1)/block.x,1,1);	
	

	Timer timer_cuda; timer_cuda.start();
    // ###
    // ###
    // ### TODO: Main computation
	for(int rep=0;rep<1;rep++)
	{
		cuda_gradient<<<grid,block>>>(d_imgIn,d_imgOut_gradx,d_imgOut_grady,h*w*nc,h,w);
		cuda_divergence<<<grid,block>>>(d_imgOut_gradx,d_imgOut_grady, d_imgOut_div,h*w*nc,h,w);
		cuda_l2<<<grid,block>>>(d_imgOut_div,d_imgOut_l2,h*w*nc, h, w, nc);
	}
    // ###
    // ###
    timer_cuda.end();  float t_cuda = timer_cuda.get(); 
	cout << "time on GPU: " << t_cuda*1000 << " ms" << endl;


	cudaMemcpy( imgOut_l2, d_imgOut_l2, h*w * sizeof(float), cudaMemcpyDeviceToHost );CUDA_CHECK;
	cudaMemcpy( imgOut_gradx, d_imgOut_gradx, h*w*nc * sizeof(float), cudaMemcpyDeviceToHost );CUDA_CHECK;
	cudaMemcpy( imgOut_grady, d_imgOut_grady, h*w*nc * sizeof(float), cudaMemcpyDeviceToHost );CUDA_CHECK;
	cudaMemcpy( imgOut_div, d_imgOut_div, h*w*nc * sizeof(float), cudaMemcpyDeviceToHost );CUDA_CHECK;
	
	cudaFree(d_imgOut_grady);CUDA_CHECK;
	cudaFree(d_imgOut_gradx);CUDA_CHECK;
	cudaFree(d_imgOut_div);CUDA_CHECK;
	cudaFree(d_imgIn);CUDA_CHECK;



    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut_gradx);
    showImage("Output Gradx", mOut, 100+w+40, 100);
	convert_layered_to_mat(mOut, imgOut_grady);
    showImage("Output Grady", mOut, 100+2*w+80, 100);

	convert_layered_to_mat(mOut, imgOut_div);
    showImage("Output Div", mOut, 100+w+40, 200);
	
	convert_layered_to_mat(mOut_gray, imgOut_l2);
    showImage("Output Absolute laplace", mOut_gray, 100, 200);
    // ### Display your own output images here as needed

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
    delete[] imgOut_gradx;
	delete[] imgOut_grady;
	delete[] imgOut_div;
	
    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



