#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <math.h>
#include "helper.h"

typedef float2 Complex;

__global__ void ComplexMUL(Complex *A, Complex *B, Complex *C, float scale)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx < scale){
    cufftComplex result;
    cufftComplex mA = A[idx];
    cufftComplex mB = B[idx];
    result.x = (mA.x * mB.x - mA.y * mB.y)/scale;
    result.y = (mA.x * mB.y + mA.y * mB.x)/scale;
    C[idx] = result;}

}

__global__ void  copyVal(float*out_red,float*out_green,float*out_blue,cufftComplex *in_red, cufftComplex *in_green,cufftComplex *in_blue,int w, int h, int W, int H ){
//__global__ void  copyVal(float*out_red,cufftComplex *in_red,int w, int h, int W, int H ){

	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein
	
	int currentlocation = iz*w*h + ix + iy * w;

	if (ix < w && iy <h && iz < 1){
	out_red[currentlocation]=in_red[W-w+ix+(H-h+iy)*W].x;
	out_blue[currentlocation]=in_blue[W-w+ix+(H-h+iy)*W].x;
	out_green[currentlocation]=in_green[W-w+ix+(H-h+iy)*W].x;
	}

}

__global__ void pad(float*out_red,float*out_green,float*out_blue,float*in_red, float*in_green, float*in_blue, int w, int h, int W,float* out_kernel, float* in_kernel,int wk,int hk){

//__global__ void pad(float*out_red,float*in_red, int w, int h, int W,float* out_kernel, float* in_kernel,int wk,int hk){



	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein
	
	int currentlocation = iz*w*h + ix + iy * w;
	//printf("Hi\n");
	if (ix < w && iy <h && iz < 1){
			
	
		out_red[ix + iy * W]=in_red[currentlocation];
		out_green[ix + iy * W]=in_green[currentlocation];
		out_blue[ix + iy * W]=in_blue[currentlocation];


	} 

	if(ix < wk && iy <hk && iz < 1)
			out_kernel[ix + iy * W]=in_kernel[iz*wk*hk + ix + iy * wk];
		//TEST	out_kernel[ix + iy * W]=1;
}

__global__ void ConvertComplex(float* in_red,float* in_green,float* in_blue,cufftComplex *out_red,cufftComplex *out_green,cufftComplex *out_blue, int w, int h,float*in_kernel,cufftComplex *out_kernel){
//__global__ void ConvertComplex(float* in_red,cufftComplex *out_red, int w, int h,float*in_kernel,cufftComplex *out_kernel){
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein
	
	int currentlocation = iz*w*h + ix + iy * w;

	if (ix < w && iy <h && iz < 1){
		out_red[currentlocation].x=in_red[currentlocation];
		out_blue[currentlocation].x=in_blue[currentlocation];
		out_green[currentlocation].x=in_green[currentlocation];
		out_kernel[currentlocation].x=in_kernel[currentlocation];

		out_red[currentlocation].y=0;
		out_blue[currentlocation].y=0;
		out_green[currentlocation].y=0;
		out_kernel[currentlocation].y=0;
	}
}


cv::Mat conv_CUFFT(cv::Mat src, cv::Mat kernel){

	int w=src.cols;
	int h=src.rows;
	int nc=src.channels();
		
	int krows=kernel.rows;
	int kcols=kernel.cols;

	//Split the images to channel
	cv::Mat src_channels[3];
	cv::split(src, src_channels);

	float *R_src_channels = new float[(size_t)w*h];
	float *B_src_channels = new float[(size_t)w*h];
	float *G_src_channels = new float[(size_t)w*h];

	float *kernelArray = new float[(size_t)krows*kcols];
	convert_mat_to_layered (R_src_channels, src);
	convert_mat_to_layered (R_src_channels, src_channels[0]);
	convert_mat_to_layered (B_src_channels,  src_channels[1]);
	convert_mat_to_layered (G_src_channels, src_channels[2]);

	convert_mat_to_layered (kernelArray, kernel);
	////////////////////////////////////////////////////////////////////////////////////
	
	float* pad_red,*pad_blue, *pad_green,*pad_kernel;
 
	int n=(h+2*(krows/2))*(w+2*(kcols/2));
	int nk=krows*kcols;
	cudaMalloc(&pad_red,n*sizeof(float));
	cudaMemset(pad_red,0, n*sizeof(float));
	cudaMalloc(&pad_blue,n*sizeof(float));
	cudaMemset(pad_blue,0, n*sizeof(float));
	
	cudaMalloc(&pad_green,n*sizeof(float));
	cudaMemset(pad_green,0, n*sizeof(float));

	cudaMalloc(&pad_kernel,n*sizeof(float));
	cudaMemset(pad_kernel,0, n*sizeof(float));

	float* R_src,*G_src, *B_src, *kernel_src;
	cudaMalloc(&R_src,h*w*sizeof(float));
	cudaMemcpy(R_src, R_src_channels,w*h*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&G_src,h*w*sizeof(float));
	cudaMemcpy(G_src, G_src_channels,w*h*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&B_src,h*w*sizeof(float));
	cudaMemcpy(B_src, B_src_channels,w*h*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&kernel_src,krows*kcols*sizeof(float));
	cudaMemcpy(kernel_src, kernelArray,krows*kcols*sizeof(float), cudaMemcpyHostToDevice);

	dim3 Block = dim3(32,32,1);
    dim3 Grid = dim3((w +Block.x -1) / Block.x, (h + Block.y -1) / Block.y, (1+ Block.z -1) / Block.z);
	
	pad<<< Grid,Block >>>(pad_red,pad_green,pad_blue,R_src,G_src,B_src,w,h, w+2*(kcols/2),pad_kernel,kernel_src,kcols,krows);
	
	int W = w+2*(kcols/2);		
	//pad<<< Grid,Block >>>(pad_red,R_src,w,h, W,pad_kernel,kernel_src,kcols,krows);
	


////////////////////////////////////////////////////////////////////////////////
//Convert to Complex
///////////////////////////////////////////////////////////////////////////////


	cufftComplex* pad_red_complex,*pad_blue_complex, *pad_green_complex,*pad_kernel_complex;
	
	cudaMalloc(&pad_red_complex,n*sizeof(Complex));
	cudaMalloc(&pad_blue_complex,n*sizeof(cufftComplex));
	cudaMalloc(&pad_green_complex,n*sizeof(cufftComplex));
	
	cudaMalloc(&pad_kernel_complex,n*sizeof(Complex));
ConvertComplex<<<Grid,Block>>>(pad_red,pad_green,pad_blue,pad_red_complex,pad_green_complex,pad_blue_complex, w+2*(kcols/2),h+2*(krows/2),pad_kernel,pad_kernel_complex);

	//ConvertComplex<<<Grid,Block>>>(pad_red,pad_red_complex, w+2*(kcols/2),h+2*(krows/2),pad_kernel,pad_kernel_complex);


// initialize CUFFT library
  	cufftHandle plan;

  	cufftPlan1d(&plan, n ,CUFFT_C2C, 1);

	printf("Transforming signal cufftExecR2C\n");
    cufftExecC2C(plan, (cufftComplex *)pad_red_complex, (cufftComplex *)pad_red_complex, CUFFT_FORWARD);
  cufftExecC2C(plan, (cufftComplex *)pad_green_complex, (cufftComplex *)pad_green_complex, CUFFT_FORWARD);
  	cufftExecC2C(plan, (cufftComplex *)pad_blue_complex, (cufftComplex *)pad_blue_complex, CUFFT_FORWARD);
    
	cufftExecC2C(plan, (cufftComplex *)pad_kernel_complex, (cufftComplex *)pad_kernel_complex, CUFFT_FORWARD);


	cufftComplex *g_RedOut, *g_BlueOut, *g_GreenOut;

	cudaMalloc(&g_RedOut, n*sizeof(Complex));
  	cudaMalloc(&g_BlueOut, n*sizeof(cufftComplex));
  	cudaMalloc(&g_GreenOut,n*sizeof(cufftComplex));

	ComplexMUL<<<32,8>>>(pad_kernel_complex, pad_red_complex, g_RedOut, n);
	 ComplexMUL<<<32,8>>>(pad_kernel_complex, pad_green_complex, g_GreenOut, n);
	 ComplexMUL<<<32,8>>>(pad_kernel_complex, pad_blue_complex, g_BlueOut,  n);
 
	cufftExecC2C(plan, (cufftComplex *)g_RedOut,(cufftComplex *) g_RedOut, CUFFT_INVERSE);
	cufftExecC2C(plan, g_GreenOut, g_GreenOut, CUFFT_INVERSE);
	cufftExecC2C(plan, g_BlueOut, g_BlueOut, CUFFT_INVERSE);
	
	copyVal<<<Grid, Block>>>(R_src,G_src,B_src,g_RedOut, g_GreenOut, g_BlueOut,w,h, w+2*(kcols/2),h+2*(krows/2));

	copyVal<<<Grid, Block>>>(R_src,g_RedOut, w,h, w+2*(kcols/2),h+2*(krows/2));

 	cudaMemcpy(R_src_channels, R_src,w*h*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(G_src_channels, G_src, w*h*sizeof(float), cudaMemcpyDeviceToHost);	
	cudaMemcpy(B_src_channels, B_src, w*h*sizeof(float), cudaMemcpyDeviceToHost);


	convert_layered_to_mat(src_channels[0], R_src_channels);
	convert_layered_to_mat(src_channels[1], B_src_channels);
	convert_layered_to_mat(src_channels[2], G_src_channels);
	
	cv::Mat ImgOut;	
	cv::merge(src_channels,3,ImgOut);


convert_layered_to_mat(src, R_src_channels);
	cufftDestroy(plan);
 
	return ImgOut;

}

