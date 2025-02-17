#include <stdio.h>
#include <cuda_runtime.h>
#include "memory.h"
namespace lyramilk { namespace tensor {



	//	float* gf;
	memory<float>::memory()
	{
		ptr = nullptr;
		count = 0;
	}

	memory<float>::memory(const memory& ov)
	{
		ptr = nullptr;
		count = 0;
	}

	memory<float>::~memory()
	{
		if(ptr!=nullptr){
			cudaFree(ptr);
		}
	}

	void memory<float>::assign(const float* p,long size)
	{
		count = size;
		cudaMalloc((void**)&ptr, sizeof(float) * count);
		cudaMemcpy(ptr, p, sizeof(float) * count, cudaMemcpyHostToDevice);

	}

	void memory<float>::assign(const std::vector<float>& ov)
	{
		count = ov.size();
		cudaMalloc((void**)&ptr, sizeof(float) * count);
		cudaMemcpy(ptr, ov.data(), sizeof(float) * count, cudaMemcpyHostToDevice);
	}

	memory<float>::operator std::vector<float>&&()
	{
		
	}

	long memory<float>::size()
	{
		return count;
	}



}}
