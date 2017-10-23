#include"library.h"
#include "cublas.h"

__global__ void ScalarMultiply(float* in, float* out, int w, int h, int c, float factor){

	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein

	int currentlocation = iz*w*h + ix + iy * w;

	if (ix < w && iy <h && iz < c)
	       out[currentlocation]=factor*in[currentlocation];

}

cv::Mat scalarMult(cv::Mat src,float factor){
	
	int w=src.cols;
	int h=src.rows;
	int nc=src.channels();

	float *srcArray = new float[(size_t)w*h*nc];
	
	float *g_srcArray;
	float *g_destArray;
	convert_mat_to_layered ( srcArray,src);

	
	cudaMalloc( &g_srcArray, w*h*nc * sizeof(float) );
	cudaMalloc( &g_destArray, w*h*nc * sizeof(float) );
	cudaMemcpy( g_srcArray, srcArray, w*h*nc * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaMemset(g_destArray, 0, w*h*nc * sizeof(float));

	//dim3 Block = dim3(32,32,1);
	//dim3 Grid = dim3((w +Block.x -1) / Block.x, (h + Block.y -1) / Block.y, (nc+ Block.z -1) / Block.z);

	//ScalarMultiply<<<Grid,Block>>>(g_srcArray,g_destArray,w,h,nc,factor);

	cublasSaxpy(nc*w*h,factor,g_srcArray,1,g_destArray,1);
	cudaMemcpy(srcArray,g_destArray, nc*h*w * sizeof(float), cudaMemcpyDeviceToHost );

	convert_layered_to_mat(src, srcArray);

	cudaFree(g_srcArray);//CUDA_CHECK;
	cudaFree(g_destArray);//CUDA_CHECK;

	delete[] srcArray;
	//delete[] destArray;

	return src;
}
