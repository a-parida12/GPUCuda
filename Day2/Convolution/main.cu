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
#include<stdio.h>
// uncomment to use the camera
//#define CAMERA



__global__ void cuda_convolution_shared ( float *in, float *out, int w, int h, int nc, float *kernel, int r)
{


	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein
	int kernelWidth = 2 * r + 1;	
	int currentlocation = iz*w*h + ix + iy * w;
	out[currentlocation]=0;

	extern __shared__ float shmem[];

	int bx=blockDim.x;
	int by=blockDim.y;

	int shmem_width= bx + 2 * r;
	int shmem_height=by + 2 * r;
	
	for( int i = threadIdx.x + threadIdx.y * bx; i < shmem_width * shmem_height ; i = i + bx * by) {   
        int shw = i % shmem_width;
        int shh = i / shmem_height;
      
       
        shmem[i] = in[max(min(w-1, -r + bx * blockIdx.x + shw), 0) + max(min(h-1, -r + by* blockIdx.y + shh), 0) * w + iz*w*h];
    }
    __syncthreads();

	if (ix<w && iy<h && iz < nc){ 
        for (int x=0; x<kernelWidth; x++) {
            for(int y=0; y<kernelWidth; y++) {
                                
                out[currentlocation] += kernel[x + y * kernelWidth] * shmem[threadIdx.x+x+ (threadIdx.y+y) * shmem_width];
            }
        }

        __syncthreads();

    }

}

#define kernelsize 100
__constant__ float kernel_const[kernelsize];
texture<float,2,cudaReadModeElementType>texRef;

__global__ void cuda_convolution_texture_constantKernel( float *in, float *out, int w, int h, int nc, int r){

	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein
	
	int kernelWidth = 2 * r + 1;
    
    int currentlocation = iz*w*h + ix + iy * w;
    out[currentlocation]=0;
	
	if (ix < w && iy <h && iz < nc){ 
	
		for (int x=0; x<kernelWidth; x++) {
			for(int y=0; y<kernelWidth; y++) {

				int cx = max(min(w-1, ix + x - r), 0);
				int cy = max(min(h-1, iy + y - r), 0);
				float val = tex2D(texRef, cx+0.5f, iz*h+cy+0.5f); 
				
				if( x+y*kernelWidth <kernelsize) {
                    out[currentlocation] += kernel_const[x+y*kernelWidth] * val;
                }	
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
	
    for(int i = 0; i < width_kernel; ++i)
        for (int j = 0; j < width_kernel; ++j)
			gKernel[i+j*width_kernel]/=sum;
       


 	//----CUDA IMPLEMENTATION BEGINS----//

	float *imgOutConvolutedGPU_shared = new float[(size_t)w*h*nc];
    float *imgOutConvolutedGPU_texture = new float[(size_t)w*h*nc];

    float *g_imgIn;
    float *g_imgOut;
    float *g_gKernel;
    
    cudaMalloc( &g_imgIn, w*h*nc * sizeof(float) );CUDA_CHECK;
    cudaMalloc( &g_imgOut, w*h*nc * sizeof(float) );CUDA_CHECK;
    cudaMalloc( &g_gKernel, width_kernel * width_kernel * sizeof(float) );CUDA_CHECK;

	cudaMemcpy( g_imgIn, imgIn, w*h*nc * sizeof(float), cudaMemcpyHostToDevice );CUDA_CHECK; 
    cudaMemcpy( g_gKernel, gKernel, width_kernel * width_kernel * sizeof(float), cudaMemcpyHostToDevice );CUDA_CHECK;
   
   
    
    int blockLength = 32;
    
    dim3 Block = dim3(blockLength,blockLength,1);
    dim3 Grid = dim3((w + Block.x -1) / Block.x, (h + Block.y -1) / Block.y, (nc + Block.z -1) / Block.z);
    							
    // shared memory							
    int sharedMemoryLength = width_kernel + blockLength -1;
   	size_t smBytes = sharedMemoryLength * sharedMemoryLength * sizeof(float);
	
	Timer timer; timer.start();
    cuda_convolution_shared <<<Grid,Block,smBytes>>> (g_imgIn, g_imgOut, w, h, nc, g_gKernel,radius_kernel);
    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time Shared Convolution on GPU: " << t*1000 << " ms" << endl;
    
    cudaMemcpy( imgOutConvolutedGPU_shared, g_imgOut, w*h*nc * sizeof(float), cudaMemcpyDeviceToHost );
    CUDA_CHECK;

	 convert_layered_to_mat(mOut, imgOutConvolutedGPU_shared);
    showImage("Output Convoluted GPU Shared Memory", mOut, 200, 120);

	// initialize texture memory
    texRef.addressMode[0] = cudaAddressModeClamp; // clamp x to border
    texRef.addressMode[1] = cudaAddressModeClamp; // clamp y to border
    texRef.filterMode = cudaFilterModeLinear; // linear interpolation
    texRef.normalized = false; // access as (x+0.5f,y+0.5f), not as ((x+0.5f)/w,(y+0.5f)/h)
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, &texRef, g_imgIn, &desc, w, h*nc, w*sizeof(g_imgIn[0]));
    CUDA_CHECK;
	

	// copying the kernel to constant memory

	cudaMemcpyToSymbol (kernel_const, gKernel, width_kernel * width_kernel * sizeof(float));CUDA_CHECK;

	timer.start();
    cuda_convolution_texture_constantKernel <<<Grid,Block>>> (g_imgIn, g_imgOut, w, h, nc,radius_kernel);
    timer.end(); t = timer.get();  // elapsed time in seconds
    cout << "time texure Convolution on GPU: " << t*1000 << " ms" << endl;
    
    cudaMemcpy( imgOutConvolutedGPU_texture, g_imgOut, w*h*nc * sizeof(float), cudaMemcpyDeviceToHost );
    CUDA_CHECK;
    
   convert_layered_to_mat(mOut, imgOutConvolutedGPU_texture);
    showImage("Output Convoluted GPU Texture Memory and Constant kernel", mOut, 150, 120);
	//----CUDA Implementation ends here---//
	

    // show input image
    showImage("Input", mIn, 100, 120);  // show at position (x_from_left=100,y_from_above=100)

    
    
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
    delete[] imgOutConvolutedGPU_shared;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



