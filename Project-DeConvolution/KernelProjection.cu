#include "helper.h"
#include "cublas.h"
# include<iostream>
using namespace std;
__global__ void RemoveNegative(float *in, float *out,int w,int h, int nc)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein

	int currentlocation = iz*w*h + ix + iy * w;
	
	

	if (ix < w && iy <h && iz < nc){ 
	
		if( in[currentlocation] >= 0.0 )
		       out[currentlocation]=in[currentlocation];
     
	}
}

__global__ void DivideSum(float *in, float *out,int w,int h, int nc,float sum)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein

	int currentlocation = iz*w*h + ix + iy * w;
	
	if (ix < w && iy <h && iz < nc){

		out[currentlocation]=in[currentlocation]/sum;
	}

}

cv::Mat KernelProjection(cv::Mat k){
//kernal projection
	cv::Mat src=k;

	int w=src.cols;
	int h=src.rows;
	int nc=src.channels();
	
	float *srcArray = new float[(size_t)w*h*nc];
	float *destArray = new float[(size_t)w*h*nc];
	
	float *g_srcArray;
	float *g_destArray;

	convert_mat_to_layered ( srcArray,src);

	cudaMalloc( &g_srcArray, w*h*nc * sizeof(float) );
	cudaMalloc( &g_destArray, w*h*nc * sizeof(float) );
	

	cudaMemcpy( g_srcArray, srcArray, w*h*nc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(g_destArray, 0, w*h*nc * sizeof(float));
	
	dim3 Block = dim3(32,32,1);
	dim3 Grid = dim3((w +Block.x -1) / Block.x, (h + Block.y -1) / Block.y, (nc+ Block.z -1) / Block.z);

	RemoveNegative<<<Grid,Block>>> (g_srcArray, g_destArray, w, h, nc);

    float sumk=cublasSasum(h*w*nc,g_destArray,1);
    
	DivideSum<<<Grid,Block>>> (g_destArray, g_srcArray, w, h, nc,sumk);
	
	cudaMemcpy(destArray,g_srcArray, nc*h*w * sizeof(float), cudaMemcpyDeviceToHost );
	
	convert_layered_to_mat(k, destArray);

	cudaFree(g_srcArray);//CUDA_CHECK;
	cudaFree(g_destArray);//CUDA_CHECK;

	delete[] srcArray;
	delete[] destArray;

	return k;
}
