#ifndef _LYRAMILK_TENSOR_MEMORY_H_
#define _LYRAMILK_TENSOR_MEMORY_H_

#include <vector>

namespace lyramilk { namespace tensor {

	template<typename T>
	class memory
	{};

	template<>
	class memory<float>
	{
		float* ptr;
		long count;
	public:
		memory();
		memory(const memory& ov);
		~memory();

		void assign(const float* p,long size);
		void assign(const std::vector<float>& ov);
		operator std::vector<float>&&();

		long size();
	};
}}



#endif
