#ifndef _LYRAMILK_TENSOR_VECTOR_H_
#define _LYRAMILK_TENSOR_VECTOR_H_

#include "memory.h"

namespace lyramilk { namespace tensor {
	class vector
	{
		memory<float> data;
	public:
		vector(float* nums,int count);
		virtual ~vector();
		vector& operator +=(const vector& b);

		double sum();

		long size();
	};
}}



#endif 