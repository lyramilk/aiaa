#include <stdio.h>
#include <cuda_runtime.h>
#include "vec.h"

#include <vector>

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
	if(blockIdx.x == 0 && threadIdx.x == 0){
		//只让第0块的第0个线程输出。要不然太多了。
		printf("核函数收到的size是%d\n",size);
	}
	// 每个块用一块共享内存。虽说是共享内存，但是每个线程只操作里面对应索引的数据。后面规约的时候再把它们合并在一起。
	extern __shared__ float sharedData[];

	// blockDim.x	块的大小(块中的线程数量)
	// blockIdx.x	块的索引，比如当前运行在第几个块上
	// threadIdx.x;	//线程的索引

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// 将数据加载到共享内存
	sharedData[tid] = (idx < size) ? input[idx] : 0;


	if(blockIdx.x == 0 && threadIdx.x == 0){
		// 核函数中竟然可以用printf还是很惊喜的，查了一下发现是cuda 3.2版本中添加的，有这个函数太舒服了。
		printf("input=%f sharedData=%f,threadidx=%d,size=%d\n",input[idx],sharedData[tid],tid,size);
	}
	__syncthreads();

	
	/** 暴力求和，在第0个线程上把整个 sharedData中的值加在一起。
	if(tid == 0){
		float ret = 0;
		for(int i=0;i<blockDim.x;++i){
			ret += sharedData[i];
		}
		output[blockIdx.x] = ret;
	}
	/**/


	/*
		归约求和，共享内存的大小和线程数量是一样的，每次循环将共享内存一分为二把右边加到左边。
		这种方式可以降低计算次数。即使在单线程cpu计算中，用这种方式也比直接循环整个数组快，gpu计算中多核计算效果还会更好些。。
	*/

	int b = 1;
	if(tid == 0){
		for(b=1;b<blockDim.x;b<<=1){
if(blockIdx.x == 0){
printf("b值=%d\n",b);
}
		}
	}
	__syncthreads();

if(tid == 0){
	printf("块%d,b值=%d,最近的2的n次方数%d\n",blockIdx.x,b);
}
	for (int s = b >> 1; s > 0; s >>= 1) {
		if (tid < s && tid + s < blockDim.x) {
			sharedData[tid] += sharedData[tid + s];
			if(blockIdx.x == 0 && threadIdx.x == 0){
				// 核函数中竟然可以用printf还是很惊喜的，查了一下发现是cuda 3.2版本中添加的，有这个函数太舒服了。
				printf("abc %d sd=%f,\n",s,sharedData[tid]);
			}
		}
		__syncthreads();
	}

	// 写回结果
	if (tid == 0) {
		output[blockIdx.x] = sharedData[0];
		printf("块%d输出值=%f,\n",blockIdx.x,output[blockIdx.x]);
	}
}


vec::vec(float* nums,int count)
{
	size = count;
	cudaMalloc((void**)&gpudata, sizeof(float) * count);
	cudaMemcpy(gpudata, nums, sizeof(float) * count, cudaMemcpyHostToDevice);
}

vec::~vec()
{
	cudaFree(gpudata);
}

vec& vec::operator +=(const vec& b)
{
	const int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

	cuda_add<<<blocksPerGrid, threadsPerBlock>>>(gpudata,b.gpudata,size);
	return *this;
}

double vec::sum()
{
	const int threadsPerBlock = 340;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

	float *gpu_output;
	// 每个块分配一个output，因为共享内存只能在块内用。为什么要用块内的共享内存而不是直接分配在这里，据说共享内存比较快。
	cudaMalloc(&gpu_output, sizeof(float) * blocksPerGrid);

	printf("块数量=%d,每块中线程数量%d,size是多少%d\n",blocksPerGrid,threadsPerBlock,size);
	cuda_sum<<<blocksPerGrid, threadsPerBlock,sizeof(float) * threadsPerBlock>>>(gpudata,gpu_output,size);

	std::vector<float> cpuoutput;
	cpuoutput.resize(blocksPerGrid);
	cudaMemcpy(cpuoutput.data(), gpu_output, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost);

	printf("c75\n");
	float result = 0;
	for(int i=0;i<blocksPerGrid;++i){
		result += cpuoutput[i];
		printf("u=%f\n",cpuoutput[i]);
	}
	return result;
}

