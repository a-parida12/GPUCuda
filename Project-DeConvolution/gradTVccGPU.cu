// Total Variation gradient
// vector-valued channel-by-channel
#include "library.h"

using namespace std;
__global__ void fyforw1GPU(float* in, float* out, int w, int h, int nc) //f([2:end end],:,:)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein
     		
    int currentlocation = ix + iy * w + iz * w * h ;
	out[currentlocation] = in[currentlocation];

	if (ix < w && iy <(h-1) && iz < nc) //ix+1<w because it is fxforw
 	{
		out[currentlocation] = in[currentlocation+w];		
	}
}
__global__ void fxforw1GPU(float* in, float* out, int w, int h, int nc) //f(:,[2:end end],:)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein
     		
    int currentlocation = ix + iy * w + iz * w * h ;
	out[currentlocation] = in[currentlocation];

	if (ix < (w-1) && iy <h && iz < nc) //ix+1<w because it is fxforw
 	{
		out[currentlocation] = in[currentlocation+1];		
	}
}
__global__ void fyback2GPU(float* in, float* out, int w, int h, int nc) //f([1 1:end-1],:,:)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein
     		
    int currentlocation = ix + iy * w + iz * w * h ;
	out[currentlocation] = in[currentlocation];

	if (ix < w && iy <(h-1) && iz < nc) //ix+1<w because it is fxforw
 	{
		out[currentlocation+w] = in[currentlocation];		
	}
}
__global__ void fxback2GPU(float* in, float* out, int w, int h, int nc) //f(:,[1 1:end-1],:)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein
     		
    int currentlocation = ix + iy * w + iz * w * h ;
	out[currentlocation] = in[currentlocation];

	if (ix < (w-1) && iy <h && iz < nc) //ix+1<w because it is fxforw
 	{
		out[currentlocation+1] = in[currentlocation];		
	}
}
__global__ void diff(float* in1, float* in2, float* out, int w, int h, int nc)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//xaxis of imagein
    int iy = threadIdx.y + blockDim.y * blockIdx.y;//yaxis of imagein
    int iz = threadIdx.z + blockDim.z * blockIdx.z;	//channels imagein
     		
    int currentlocation = ix + iy * w + iz * w * h ;
	out[currentlocation] = 0;

	if (ix < (w) && iy <h && iz < nc)
 	{
		out[currentlocation] = in1[currentlocation] - in2[currentlocation];
	}
}

cv::Mat gradTVccGPU(cv::Mat src, float eps)
{ 
	int w=src.cols;
	int h=src.rows;
	int nc=src.channels();
	int n=w*h*nc;
	
	float *srcArray = new float[(size_t)n];
	convert_mat_to_layered (srcArray, src);

	float *g_In;

	float *g_Outfyforw1;
	float *g_Outfyforw;
	float *g_Outfxforw1;
	float *g_Outfxforw;
	float *g_Outfyback2;
	float *g_Outfyback;
	float *g_Outfxback2;
	float *g_Outfxback;
	float *g_Outfymixd1;
	float *g_Outfymixd;
	float *g_Outfxmixd1;
	float *g_Outfxmixd;
    
	cudaMalloc( &g_In, n * sizeof(float) );//CUDA_CHECK;

	cudaMalloc( &g_Outfyforw1, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_Outfyforw, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_Outfxforw1, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_Outfxforw, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_Outfyback2, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_Outfyback, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_Outfxback2, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_Outfxback, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_Outfymixd1, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_Outfymixd, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_Outfxmixd1, n * sizeof(float) );//CUDA_CHECK;
	cudaMalloc( &g_Outfxmixd, n * sizeof(float) );//CUDA_CHECK;
    
	float *destArrayfyforw = new float[(size_t)n];	
	float *destArrayfxforw = new float[(size_t)n];	
	float *destArrayfyback = new float[(size_t)n];	
	float *destArrayfxback = new float[(size_t)n];	
	float *destArrayfymixd = new float[(size_t)n];	
	float *destArrayfxmixd = new float[(size_t)n];	

	cv::Mat destfyforw(src.size(),src.type());
	cv::Mat destfxforw(src.size(),src.type());
	cv::Mat destfyback(src.size(),src.type());
	cv::Mat destfxback(src.size(),src.type());
	cv::Mat destfymixd(src.size(),src.type());
	cv::Mat destfxmixd(src.size(),src.type());
	
	dim3 Block = dim3(32,32,1);
	dim3 Grid = dim3((w + Block.x -1) / Block.x, (h + Block.y -1) / Block.y, (nc + Block.z -1) / Block.z);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
fyforw1GPU f([2:end end],:,:)
fxforw1GPU f(:,[2:end end],:)
fyback2GPU f([1 1:end-1],:,:)
fxback2GPU f(:,[1 1:end-1],:)
*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStream_t stream3;
	cudaStream_t stream4;
	cudaStream_t stream5;
	cudaStream_t stream6;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	cudaStreamCreate(&stream4);
	cudaStreamCreate(&stream5);
	cudaStreamCreate(&stream6);


// fyforw = f([2:end end],:,:)-f
	cudaMemcpy( g_In, srcArray, n * sizeof(float), cudaMemcpyHostToDevice);///CUDA_CHECK; 	

	fyforw1GPU<<<Grid, Block, 0, stream1>>>(g_In, g_Outfyforw1, w, h, nc);//CUDA_CHECK;
	diff<<<Grid, Block, 0, stream1>>>(g_Outfyforw1, g_In, g_Outfyforw, w, h, nc);//CUDA_CHECK;
	cudaMemcpyAsync(destArrayfyforw, g_Outfyforw, n * sizeof(float), cudaMemcpyDeviceToHost, stream1 );//CUDA_CHECK;

// fxforw = f(:,[2:end end],:)-f
	fxforw1GPU<<<Grid, Block, 0, stream2>>>(g_In, g_Outfxforw1, w, h, nc);//CUDA_CHECK;
	diff<<<Grid, Block, 0, stream2>>>(g_Outfxforw1, g_In, g_Outfxforw, w, h, nc);//CUDA_CHECK;
	cudaMemcpyAsync(destArrayfxforw, g_Outfxforw, n * sizeof(float), cudaMemcpyDeviceToHost, stream2 );//CUDA_CHECK;

// fyback = f-f([1 1:end-1],:,:)
	fyback2GPU<<<Grid, Block, 0, stream3>>>(g_In, g_Outfyback2, w, h, nc);//CUDA_CHECK;
	diff<<<Grid, Block, 0, stream3>>>(g_In, g_Outfyback2, g_Outfyback, w, h, nc);//CUDA_CHECK;
	cudaMemcpyAsync(destArrayfyback, g_Outfyback, n * sizeof(float), cudaMemcpyDeviceToHost, stream3);//CUDA_CHECK;
	
// fxback = f-f(:,[1 1:end-1],:)	
	fxback2GPU<<<Grid, Block, 0, stream4>>>(g_In, g_Outfxback2, w, h, nc);//CUDA_CHECK;
	diff<<<Grid, Block, 0, stream4>>>(g_In, g_Outfxback2, g_Outfxback, w, h, nc);//CUDA_CHECK;
	cudaMemcpyAsync(destArrayfxback, g_Outfxback, n * sizeof(float), cudaMemcpyDeviceToHost, stream4 );//CUDA_CHECK;
	
// fymixd = f([2:end end],[1 1:end-1],:)-f(:,[1 1:end-1],:);
	cudaStreamSynchronize(stream1);	
	fxback2GPU<<<Grid, Block, 0, stream5>>>(g_Outfyforw1, g_Outfymixd1, w, h, nc);//CUDA_CHECK;
	cudaStreamSynchronize(stream4);	
	diff<<<Grid, Block, 0, stream5>>>(g_Outfymixd1, g_Outfxback2, g_Outfymixd, w, h, nc);//CUDA_CHECK;
	cudaMemcpyAsync(destArrayfymixd, g_Outfymixd, n * sizeof(float), cudaMemcpyDeviceToHost, stream5 );//CUDA_CHECK;
	
// fxmixd = f([1 1:end-1],[2:end end],:)-f([1 1:end-1],:,:);
	//	cudaStreamDestroy(stream3);	
	cudaStreamSynchronize(stream3);	
	fxforw1GPU<<<Grid, Block, 0, stream6>>>(g_Outfyback2, g_Outfxmixd1, w, h, nc);//CUDA_CHECK;
	diff<<<Grid, Block, 0, stream6>>>(g_Outfxmixd1, g_Outfyback2, g_Outfxmixd, w, h, nc);//CUDA_CHECK;
	cudaMemcpyAsync(destArrayfxmixd, g_Outfxmixd, n * sizeof(float), cudaMemcpyDeviceToHost, stream6);//CUDA_CHECK;

//	cudaStreamDestroy(stream2);
	cudaStreamSynchronize(stream2);	

//	cudaStreamDestroy(stream5);
	cudaStreamSynchronize(stream5);	

//	cudaStreamDestroy(stream6);
	cudaStreamSynchronize(stream6);	

	convert_layered_to_mat(destfyback, destArrayfyback);
	convert_layered_to_mat(destfxforw, destArrayfxforw);
	convert_layered_to_mat(destfyforw, destArrayfyforw);	
	convert_layered_to_mat(destfxback, destArrayfxback);
	convert_layered_to_mat(destfymixd, destArrayfymixd);
	convert_layered_to_mat(destfxmixd, destArrayfxmixd);

// Free CUDA Memory after computation
	cudaFree(g_In);//CUDA_CHECK;
	cudaFree(g_Outfyforw1);//CUDA_CHECK;
	cudaFree(g_Outfyforw);//CUDA_CHECK;
	cudaFree(g_Outfxforw1);//CUDA_CHECK;
	cudaFree(g_Outfxforw);//CUDA_CHECK;
	cudaFree(g_Outfyback2);//CUDA_CHECK;
	cudaFree(g_Outfyback);//CUDA_CHECK;
	cudaFree(g_Outfxback2);//CUDA_CHECK;
	cudaFree(g_Outfxback);//CUDA_CHECK;
	cudaFree(g_Outfymixd1);//CUDA_CHECK;
	cudaFree(g_Outfymixd);//CUDA_CHECK;
	cudaFree(g_Outfxmixd1);//CUDA_CHECK;
	cudaFree(g_Outfxmixd);//CUDA_CHECK;

// Free Memory
	delete[] srcArray;
	delete[] destArrayfyforw;
	delete[] destArrayfxforw;
	delete[] destArrayfyback;
	delete[] destArrayfxback;
	delete[] destArrayfymixd;
	delete[] destArrayfxmixd;

// Cloning the outputs
	cv::Mat fxforwCloned = destfxforw.clone();
	cv::Mat fyforwCloned = destfyforw.clone();
	cv::Mat fxbackCloned = destfxback.clone();
	cv::Mat fybackCloned = destfyback.clone();
	cv::Mat fxmixdCloned = destfxmixd.clone();
	cv::Mat fymixdCloned = destfymixd.clone();

// Calculating divTV
	cv::Mat divTV(src.cols, src.rows,CV_32FC3);
	divTV = 0.0;
	divTV  = (fyforwCloned + fxforwCloned)/L2GradGPU(fxforwCloned, fyforwCloned, eps) - fybackCloned/L2GradGPU(fybackCloned, fxmixdCloned, eps) - fxbackCloned/L2GradGPU(fymixdCloned, fxbackCloned, eps);

// Test
	cout<<"fyforw: "<<endl<<fyforwCloned<<endl;
	cout<<"fxforw: "<<endl<<fxforwCloned<<endl;
	cout<<"fyback: "<<endl<<fybackCloned<<endl;
	cout<<"fxback: "<<endl<<fxbackCloned<<endl;
	cout<<"fymixd: "<<endl<<fymixdCloned<<endl;
	cout<<"fxmixd: "<<endl<<fxmixdCloned<<endl;
	cout<<"divTV: "<<endl<<divTV<<endl;

	return (divTV);
}

