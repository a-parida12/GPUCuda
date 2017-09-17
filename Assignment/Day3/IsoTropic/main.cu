// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2017, September 11 - October 9
// ###

#include "helper.h"
#include <stdio.h>
#include <iostream>
using namespace std;

// uncomment to use the camera
//#define CAMERA

#define BLOCKSIZE 32


__global__ void computeImageGradient (float *d_imgIn, float *d_gradx, float *d_grady, int w, int h)
{
	int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
	int ind_y = threadIdx.y + blockDim.y * blockIdx.y;
	int ind_z = threadIdx.z + blockDim.z * blockIdx.z;
	int indx_next = (ind_x+1) + w * ind_y + ind_z  * w * h;
	int ind_curr = ind_x + w * ind_y + ind_z  * w * h;

	 d_gradx[ind_curr] = d_imgIn[indx_next]-d_imgIn[ind_curr];

	 int indy_next = ind_x + w * (ind_y+1) + ind_z  * w * h;

	 if( ind_x < w && ind_y<h && ind_z<3) d_grady[ind_curr] = d_imgIn[indy_next]-d_imgIn[ind_curr];
}




__global__ void computeConvolutionSharedMem (float *d_imgIn, float *d_imgConvo, float *d_kernel, int w, int h, int nc, int KernelRadius)
{


	int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
	int ind_y = threadIdx.y + blockDim.y * blockIdx.y;
	int ind_z = threadIdx.z + blockDim.z * blockIdx.z;
	int ind_curr = ind_x+ind_y*w+ind_z*w*h;

	int iblockx = threadIdx.x;
	int iblocky = threadIdx.y;
	int iblockz = threadIdx.z;

	int bx = blockDim.x;
	int by = blockDim.y;

	int shw = bx+2*KernelRadius;
	int shh = by+2*KernelRadius;

	extern __shared__ float shared_mem[];
	int tid = threadIdx.x + threadIdx.y * blockDim.x;

	int mx = -KernelRadius + bx*blockIdx.x;
	int my = -KernelRadius + by*blockIdx.y;
	//for (int j = 0; j<nc; j++){
		for (int i = tid; i<shh*shw; i+=bx*by){
			int shx = i%shw;
			int shy = i/shw;
			int clamp_indx = max(min(w-1, mx+shx), 0);
		        int clamp_indy = max(min(h-1, my+shy), 0);
			shared_mem[i] = d_imgIn[clamp_indx+clamp_indy*w+ind_z*h*w];
			//printf("tsharemMem %d: %f : index %d clamp_indx: %d clamp_indy: %d %TIDx: %d SHx: %d SHy: %d\n", i, shared_mem[i], clamp_indx+clamp_indy*w+ind_z*h*w, clamp_indx, clamp_indy, ind_curr, mx+shx, my+shy );
		}
	
	//}
	__syncthreads();
	//printf( "shared mem width %d ShareMem Height%d", shw, shh);
	d_imgConvo[ind_curr] = 0;
	if( ind_x < w && ind_y<h && ind_z<3){
		for (int m = 0; m < 2*KernelRadius+1; m++){
			for (int n = 0; n < 2*KernelRadius+1; n++){
				//int clamp_indx = max(min(w-1, iblockx + m - KernelRadius), 0);
		                //int clamp_indy = max(min(h-1, iblocky + n - KernelRadius), 0);
				float pix_Val = shared_mem[(iblockx + m)+(iblocky + n)*shw+iblockz*shh*shw]; 
				d_imgConvo[ind_curr]+=d_kernel[m+n*(2*KernelRadius+1)]*pix_Val;
				//printf("thIDx: %d thIDy: %d thIDz: %d\n", iblockx, iblocky, iblockz);
				//d_imgConvo[ind_curr]=;
			}
		}
	}
	

}



__device__ float diffusivity(float);
__global__ void applyDiffusion (float *d_imgOut, float *d_imgIn, float *d_gradx, float *d_grady, float *d_gradient,int w, int h, float tao)
{
	int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
	int ind_y = threadIdx.y + blockDim.y * blockIdx.y;
	int ind_z = threadIdx.z + blockDim.z * blockIdx.z;

	int ind_curr = ind_x + w * ind_y + ind_z  * w * h;

	float v1 = d_gradx[ind_curr];
	float v2 = d_grady[ind_curr];

	float l2 = sqrtf(v1*v1+v2*v2);
	float eps = 0.1f;

	float g = 1.0f/(max(eps,l2));//diffusivity(l2);

	d_gradx[ind_curr] = g*d_gradx[ind_curr];
	d_grady[ind_curr] = g*d_grady[ind_curr];
	
	int indx_prev = (ind_x-1) + w * ind_y + ind_z  * w * h;
	int indy_prev = ind_x + w * (ind_y-1) + ind_z  * w * h;
	
	float divX = d_gradx[ind_curr]-d_gradx[indx_prev];
	float divY = d_grady[ind_curr]-d_grady[indy_prev];
	
	float fact = tao*(divX+divY);

	d_gradient[ind_curr] = fact;
	if( ind_x < w && ind_y<h && ind_z<3){
		d_imgOut[ind_curr] = d_imgIn[ind_curr] + fact;
	}
}

__device__ float diffusivity(float val)//FIND g(Val)
{
	return val;
}

void formGaussianKernel(float, float*, float*);

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
    //cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOuty(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    cv::Mat mOut1Ch(h,w,CV_32FC1); 
    cv::Mat mOut3Ch(h,w,CV_32FC3);  
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
    float *imgOut1Cha = new float[(size_t)w*h*mOut1Ch.channels()];
    float *imgOut1Chb = new float[(size_t)w*h*mOut1Ch.channels()];
    float *imgOut1Chc = new float[(size_t)w*h*mOut1Ch.channels()];

    float *imgOut3Cha = new float[(size_t)w*h*mOut3Ch.channels()];
    float *imgOut3Chb = new float[(size_t)w*h*mOut3Ch.channels()];


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




    Timer timer; timer.start();
    // ###
    // ###
    // ### TODO: Main computation
    // ###
    // ###
	float tao = 0.02f;
	int N_itr = 20;
	float sigma = 2;//sqrtf(2*tao*N_itr);
	int radius = ceil(3*sigma);
	int diameter = 2*radius + 1;
	float *G_sigma = new float[(size_t)diameter * diameter];
	float *G_sigma_copy = new float[(size_t)diameter * diameter];


	formGaussianKernel(sigma, G_sigma, G_sigma_copy);

	cv::Mat imGaussian(diameter,diameter,CV_32FC1);  
	/*convert_layered_to_mat(imGaussian, G_sigma_copy);
    	showImage("GaussianKernel", imGaussian, 100+2*w+40, 100);*/

	int n = w*h*nc;
	

		
		//RUN ON GPU
	    	cudaDeviceSynchronize();
		float *d_imgIn, *d_gradx, *d_grady, *d_imgOut, *d_imgDiv, *d_gradient, *d_S, *d_kernel;//*d_kernelGradx, *d_kernelGrady,, *d_m11, *d_m12, *d_m22
		cudaMalloc(&d_imgIn, n*sizeof(float));
		cudaMalloc(&d_S, n*sizeof(float));
		cudaMalloc(&d_kernel, diameter*diameter*sizeof(float));
		cudaMalloc(&d_gradx, n*sizeof(float));
		cudaMalloc(&d_grady, n*sizeof(float));
		cudaMalloc(&d_imgDiv, n*sizeof(float));	
		cudaMalloc(&d_gradient, n*sizeof(float));	
		
		cudaMalloc(&d_imgOut, n*sizeof(float));
		CUDA_CHECK;

		cudaMemcpy(d_imgIn, imgIn, n * sizeof(float), cudaMemcpyHostToDevice );
		cudaMemcpy(d_kernel, G_sigma, diameter*diameter* sizeof(float), cudaMemcpyHostToDevice );
		
		cudaDeviceSynchronize();

		dim3 block = dim3(BLOCKSIZE,BLOCKSIZE,1); // 128*1*1 threads per block
		dim3 grid = dim3((w+block.x-1)/block.x, (h+block.y-1)/block.y, (nc+block.z-1)/block.z);

		int shared_memsize = (BLOCKSIZE+2*radius)*(BLOCKSIZE+2*radius)*nc*sizeof(float);
		computeConvolutionSharedMem <<<grid, block, shared_memsize>>> (d_imgIn, d_S, d_kernel, w, h, nc, radius);
		for( int i = 0; i < N_itr; i++){

			computeImageGradient <<<grid, block>>> (d_imgIn, d_gradx, d_grady, w, h);
			applyDiffusion <<<grid, block>>> (d_imgIn, d_imgIn, d_gradx, d_grady, d_gradient, w, h, tao);
		}



		cudaMemcpy( imgOut3Chb, d_imgIn, n*sizeof(float), cudaMemcpyDeviceToHost);
		convert_layered_to_mat(mOut3Ch, imgOut3Chb);
		showImage("ImDiffused", mOut3Ch, 100+w+40, 100);
		
		cudaMemcpy( imgOut3Chb, d_S, n*sizeof(float), cudaMemcpyDeviceToHost);
		convert_layered_to_mat(mOut3Ch, imgOut3Chb);
		showImage("GaussianConvolution", mOut3Ch, 100+2*w+40, 100);
	
		/*cudaMemcpy( imgOut3Cha, d_gradient, n*sizeof(float), cudaMemcpyDeviceToHost);
		convert_layered_to_mat(mOut3Ch, imgOut3Cha);
    		showImage("ImGradient", mOut3Ch, 100+2*w+40, h+100);*/
		
		//cudaMemcpy( imGrady, d_grady, n * sizeof(float), cudaMemcpyDeviceToHost);
		//cuda_check();
		cudaFree(d_imgIn);
		cudaFree(d_S);
		cudaFree(d_kernel);
		cudaFree(d_gradx);
		cudaFree(d_grady);
		cudaFree(d_imgDiv);
		cudaFree(d_gradient);

		cudaFree(d_imgOut);
		
		CUDA_CHECK;
	//}

    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;

    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)



    // ### Display your own output images here as needed
    // show output image: first convert to interleaved opencv format from the layered raw array
    /*convert_layered_to_mat(mOut, imgOut);
    showImage("Gradient", mOut, 100+w+40, 100);*/
    // show output image: first convert to interleaved opencv format from the layered raw array
    #ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif




    // save input and result
   // cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    //cv::imwrite("image_gradx.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;
    
    delete[] imgOut1Cha;
    delete[] imgOut1Chb;
    delete[] imgOut1Chc;

    delete[] imgOut3Cha;
    delete[] imgOut3Chb;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}

void formGaussianKernel(float sigma,float *G_sigma, float *G_sigma_copy)
{
	//float sigma = 3.0;
	int radius = ceil(3*sigma);
	int diameter = 2*radius + 1;
	float den = 2*sigma*sigma;//M_PI*
	float sum_kernel = 0;
	float r;
	int idx, idy;
	for (int x = -radius; x <= radius; x++)	{
	        for(int y = -radius; y <= radius; y++)	{
			idx = radius+x;
			idy = radius+y;
            		r = x*x + y*y;
            		G_sigma[idx+idy*diameter] = (exp(-r/den))/(M_PI * den);
	            	sum_kernel += G_sigma[idx+idy*diameter];
        	}
    	}

	for (int x = -radius; x <= radius; x++)
	        for(int y = -radius; y <= radius; y++){
			idx = radius+x;
			idy = radius+y;
			G_sigma[idx+idy*diameter]/=sum_kernel;
		}



	float  max_kernel = G_sigma[(diameter*diameter)/2];
	for (int x = -radius; x <= radius; x++)
	        for(int y = -radius; y <= radius; y++){
			idx = radius+x;
			idy = radius+y;
			G_sigma_copy[idx+idy*diameter]=G_sigma[idx+idy*diameter]/max_kernel;
		}
	

}

