#include <stdio.h>
#include <cuda_runtime.h>
#include "vector.h"
__global__ void  cuda_add(float* a,const float* b,int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<size)
	{
		a[i] += b[i];
	}
}
__global__ void  cuda_sum(const float* input,float* output,int size)
{
	extern __shared__ float sharedData[];
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	sharedData[tid] = (idx < size) ? input[idx] : 0;
	__syncthreads();
	for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
		if (tid < s) {
			sharedData[tid] += sharedData[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) {
		atomicAdd(output,sharedData[0]);
	}
}


namespace lyramilk { namespace tensor {
	vector::vector(float* nums,int count)
	{
		data.assign(nums,count);
		//size = count;
		//cudaMalloc((void**)&data, sizeof(float) * count);
		//cudaMemcpy(data, nums, sizeof(float) * count, cudaMemcpyHostToDevice);
	}

	vector::~vector()
	{
	}

	vector& vector::operator +=(const vector& b)
	{
		const int threadsPerBlock = 256;
		int blocksPerGrid = (data.size() + threadsPerBlock - 1) / threadsPerBlock;

		cuda_add<<<blocksPerGrid, threadsPerBlock>>>(data.ptr,b.data.ptr,size);
		return *this;
	}

	double vector::sum()
	{
		const int threadsPerBlock = 256;
		int blocksPerGrid = (data.size() + threadsPerBlock - 1) / threadsPerBlock;

		float *output;
		cudaMalloc(&output, sizeof(float));

		cuda_sum<<<blocksPerGrid, threadsPerBlock>>>(data,output,size);
		float result;
		cudaMemcpy(&result, data, sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(output);
		return result;
	}

	long vector::size()
	{
		return data.size();
	}

}}
