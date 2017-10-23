#include "library.h"
#include "cublas.h"

using namespace std;

float MaxFinder(cv::Mat src){
	float maximum;
	int w=src.cols;
	int h=src.rows;
	int nc=src.channels();
	
	float *srcArray = new float[(size_t)w*h*nc];
	float *g_srcArray;
	cudaMalloc( &g_srcArray, w*h*nc * sizeof(float) );
	convert_mat_to_layered ( srcArray,src);
	cudaMemcpy( g_srcArray, srcArray, w*h*nc * sizeof(float), cudaMemcpyHostToDevice);

	int max_id=cublasIsamax(h*w*nc,g_srcArray,1);
	
	maximum=srcArray[max_id-1];
	cudaFree(g_srcArray);
	delete[] srcArray;
	return maximum;

}
