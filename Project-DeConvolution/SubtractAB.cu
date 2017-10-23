#include"library.h"
#include "cublas.h"
using namespace std;

cv::Mat SubtractAB(cv::Mat A,cv::Mat B){
//A-B
	int w=A.cols;
	int h=A.rows;
	int nc=A.channels();


	float *A_Array = new float[(size_t)w*h*nc];
	float *B_Array = new float[(size_t)w*h*nc];

	convert_mat_to_layered ( A_Array,A);
	convert_mat_to_layered ( B_Array,B);

	float *g_A_Array;
	float *g_B_Array;

	cudaMalloc( &g_A_Array, w*h*nc * sizeof(float) );
	cudaMalloc( &g_B_Array, w*h*nc * sizeof(float) );
	
	cudaMemcpy( g_A_Array, A_Array, w*h*nc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( g_B_Array, B_Array, w*h*nc * sizeof(float), cudaMemcpyHostToDevice);
	
	cublasSaxpy(nc*w*h,-1,g_B_Array,1,g_A_Array,1);

	cudaMemcpy(A_Array,g_A_Array, w*h*nc * sizeof(float), cudaMemcpyDeviceToHost );

	convert_layered_to_mat(A, A_Array);


	cudaFree(g_A_Array);//CUDA_CHECK;
	cudaFree(g_B_Array);//CUDA_CHECK;
	
	delete[] A_Array;
	delete[] B_Array;
	
	return A;

	
}
