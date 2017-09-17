#include "helper.h"
#include <iostream>
#include "cublas.h"
using namespace std;

__global__ void summing(float* input, float* output, int n){
    int t_numx = threadIdx.x + blockDim.x*blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float blocksum;
	
	if (tid ==0)
		blocksum = 0;

	__syncthreads();

	if (t_numx <n)
    	atomicAdd(&blocksum, input[t_numx]);
	

	__syncthreads();
	
	if (tid ==0){
		output[blockIdx.x]=0;   
	    output[blockIdx.x] += blocksum;
	}
}


int main(int argc, char **argv)
{
	int n = 1e5;    
	float* array = new float[n];
	for (int i=0 ; i<n; i++){
		array[i] = i;
	}

	float* g_array, *b_cuda;
	cudaMalloc(&g_array, n*sizeof(float));
	cudaMemcpy(g_array, array , n*sizeof(float), cudaMemcpyHostToDevice);

	int block = 1024;// keep block size equal to SM
	int grid = ((block + n -1)/block);

	cudaMalloc(&b_cuda,  grid*sizeof(float));

	float* b = new float[grid]; 
	float sum_l =0;
	Timer timer1; timer1.start();
	summing <<<grid, block>>> (g_array, b_cuda , n);

	cudaMemcpy(b , b_cuda, grid*sizeof(float) , cudaMemcpyDeviceToHost);
	for (int i=0 ; i<grid ; i++){
		sum_l +=b[i];
	}
	timer1.end();  float t = timer1.get();  // elapsed time in seconds

	cout << "Sum: " << sum_l  << " Time on own: "<< t*1000 << endl;
	
	Timer timer2; timer2.start();
	sum_l = cublasSasum (n, g_array, 1);
	timer2.end(); float x = timer2.get(); 
	cout << "Sum: " << sum_l  << " Time on Cubla: "<< t*1000 << endl;
	

	free(array);

	free(b);
	cudaFree(g_array);
	cudaFree(b_cuda);

return 0;
}



