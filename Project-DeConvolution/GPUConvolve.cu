#include"library.h"

using namespace std;

__global__ void convoluteGPU (float *in, float *out, int w, int h, int nc, float *kernel,int kernelHeight, int kernelWidth)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein
    
    int currentlocation = iz*w*h + ix + iy * w;
	out[currentlocation]=0;
	if (ix < w && iy <h && iz < nc){ 

		for (int x=0; x<kernelWidth; x++) {
			for(int y=0; y<kernelHeight; y++) {

				int cx = max(min(w-1, ix + x - kernelWidth/2), 0);
				int cy = max(min(h-1, iy + y - kernelHeight/2), 0);

				out[currentlocation] += kernel[x+y*kernelWidth] * in[iz*w*h+cx+cy*w];
			}
		}
	}
}


cv::Mat GPUConvolve(cv::Mat src,cv::Mat kernel){
	
	int w=src.cols;
	int h=src.rows;
	int nc=src.channels();
	
	int krows=kernel.rows;
	int kcols=kernel.cols;

	cv::Mat dest(src.size(),src.type());

	float *srcArray = new float[(size_t)w*h*nc];
	float *destArray = new float[(size_t)w*h*nc];	
	float *kernelArray = new float[(size_t)krows*kcols];
	
	convert_mat_to_layered ( srcArray,src);
	convert_mat_to_layered (kernelArray, kernel);

	float *g_srcArray;
	float *g_destArray;
	float *g_kernelArray;
    
	cudaMalloc( &g_srcArray, w*h*nc * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_destArray, w*h*nc * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_kernelArray, krows * kcols * sizeof(float) );//CUDA_CHECK;
   
	 cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	//
	cudaMemcpyAsync( g_srcArray, srcArray, w*h*nc * sizeof(float), cudaMemcpyHostToDevice,stream1 );///CUDA_CHECK; 
	cudaMemcpyAsync( g_kernelArray, kernelArray, krows * kcols * sizeof(float), cudaMemcpyHostToDevice,stream1 );//CUDA_CHECK;

	//cudaMemset(g_destArray,0.0,w*h*nc * sizeof(float));CUDA_CHECK;

	dim3 Block = dim3(32,32,1);
	dim3 Grid = dim3((w +Block.x -1) / Block.x, (h + Block.y -1) / Block.y, (nc+ Block.z -1) / Block.z);
    cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream1);		
	
	convoluteGPU <<<Grid,Block>>> (g_srcArray, g_destArray, w, h, nc, g_kernelArray, krows,kcols);//CUDA_CHECK;


	cudaMemcpy(destArray,g_destArray, nc*h*w * sizeof(float), cudaMemcpyDeviceToHost );//CUDA_CHECK;

	convert_layered_to_mat(dest, destArray);

	cudaFree(g_srcArray);//CUDA_CHECK;
	cudaFree(g_destArray);//CUDA_CHECK;
	cudaFree(g_kernelArray);//CUDA_CHECK;

	delete[] srcArray;
	delete[] destArray;
	delete[] kernelArray;

	return dest;

}
