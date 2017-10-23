// Square, add, sqrt, max //max(epsilon,sqrt(fxforw(:,:,cc).^2+fyforw(:,:,cc).^2))
#include"library.h"
using namespace std;
//////////////////////////////////////////////////////////////////////////////////////fxforw
__global__ void squareElements(float *in, float *out, int w, int h, int nc) {
    /* which element does this compute? */
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein

    int currentlocation = ix + iy * w + iz * w * h ;
	out[currentlocation] = 0;

    /* if valid, squre the array element */
	if (ix < w && iy <h && iz < nc)
 	{
            out[currentlocation] = (in[currentlocation] * in[currentlocation]);
	}
}
__global__ void addElements(float *in1, float *in2, float *out, int w, int h, int nc) {
    /* which element does this compute? */
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein

    int currentlocation = ix + iy * w + iz * w * h ;
	out[currentlocation] = 0;

    /* if valid, squre the array element */
	if (ix < w && iy <h && iz < nc)
 	{
            out[currentlocation] = (in1[currentlocation] + in2[currentlocation]);
	}
}
__global__ void sqrtElements(float *in, float *out, int w, int h, int nc) {
    /* which element does this compute? */
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein

    int currentlocation = ix + iy * w + iz * w * h ;
	out[currentlocation] = 0;

    /* if valid, sqrt the array element */
    if (ix < w && iy <h && iz < nc)
 	{
            out[currentlocation] = sqrtf(in[currentlocation]);
	}
}
__global__ void cmpElements(float *in, float *out, int w, int h, int nc, float eps) {
    /* which element does this compute? */
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein

    int currentlocation = ix + iy * w + iz * w * h ;
	out[currentlocation] = 0;

    /* if valid, sqrt the array element */
    if (ix < w && iy <h && iz < nc)
 	{
            out[currentlocation] = (in[currentlocation]<eps ? eps : in[currentlocation]);
	}
}
cv::Mat L2GradGPU(cv::Mat e, cv::Mat f, float eps)
//float* L2GradGPU(float* srcArraye, float* srcArrayf, float eps, int w, int h, int nc)
{ // Here f and e are of the same size and type
	int w=f.cols;
	int h=f.rows;
	int nc=f.channels();
	int n = w*h*nc;
	
	cv::Mat dest(f.size(), f.type());
	
	float *srcArrayf = new float[(size_t)n];
	float *srcArraye = new float[(size_t)n];
	float *destArray = new float[(size_t)n];	
	
	convert_mat_to_layered (srcArrayf, f);
	convert_mat_to_layered (srcArraye, e);

	dim3 Block = dim3(32,32,1);
	dim3 Grid = dim3((w + Block.x -1) / Block.x, (h + Block.y -1) / Block.y, (nc + Block.z -1) / Block.z);

	float *g_In_f;
	float *g_In_e;
	float *g_Out_f2;
	float *g_Out_e2;
	float *g_Out_sum;
	float *g_Out_sqrt;
	float *g_Out_cmp;
    
	cudaMalloc( &g_In_f, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_In_e, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_Out_f2, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_Out_e2, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_Out_sum, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_Out_sqrt, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_Out_cmp, n * sizeof(float) );//CUDA_CHECK;
 
	cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
   
	cudaMemcpyAsync( g_In_f, srcArrayf, n * sizeof(float), cudaMemcpyHostToDevice, stream1 );///CUDA_CHECK; 
	cudaMemcpyAsync( g_In_e, srcArraye, n * sizeof(float), cudaMemcpyHostToDevice, stream2 );///CUDA_CHECK; 

	squareElements<<<Grid, Block, 0, stream1>>>(g_In_f, g_Out_f2, w, h, nc);//CUDA_CHECK;
	squareElements<<<Grid, Block, 0, stream2>>>(g_In_e, g_Out_e2, w, h, nc);//CUDA_CHECK;

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	addElements<<<Grid, Block>>>(g_Out_f2, g_Out_e2, g_Out_sum, w, h, nc);//CUDA_CHECK;
	sqrtElements<<<Grid, Block>>>(g_Out_sum, g_Out_sqrt, w, h, nc);//CUDA_CHECK;
	cmpElements<<<Grid, Block>>>(g_Out_sqrt, g_Out_cmp, w, h, nc, eps);//CUDA_CHECK;

	cudaMemcpy(destArray, g_Out_cmp, n * sizeof(float), cudaMemcpyDeviceToHost );//CUDA_CHECK;
	convert_layered_to_mat (dest, destArray); 

	cudaFree(g_In_f);//CUDA_CHECK;
	cudaFree(g_In_e);//CUDA_CHECK;
	cudaFree(g_Out_f2);//CUDA_CHECK;
	cudaFree(g_Out_e2);//CUDA_CHECK;
	cudaFree(g_Out_sum);//CUDA_CHECK;
	cudaFree(g_Out_sqrt);//CUDA_CHECK;
	cudaFree(g_Out_cmp);//CUDA_CHECK;

	delete[] srcArraye;
	delete[] srcArrayf;
	delete[] destArray;

	return dest;
}
