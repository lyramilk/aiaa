#include <iostream>
#include "cudabp/vector.h"

#define N 1000


int main(int argc,const char* argv[])
{
	float* a = new float[N];
	for(int i=0;i<N;++i){
		a[i] = 0.5;
	}
	float* b = new float[N];
	for(int i=0;i<N;++i){
		b[i] = 0.001;
	}

	cudabp::vector v1(a,N);
	cudabp::vector v2(b,N);

	for(int i=0;i<1;++i){
		//v1 += v2;
	}

	printf("结果是:%f\n",v1.sum());
	return 0;
}

