#include <iostream>
#include "vec.h"

#define N 1000


int main(int argc,const char* argv[])
{
	float* a = new float[N];
	for(int i=0;i<N;++i){
		a[i] = 0.42;
	}
	float* b = new float[N];
	for(int i=0;i<N;++i){
		b[i] = 0.08;
	}

	vec v1(a,N);
	vec v2(b,N);

	for(int i=0;i<1;++i){
		v1 += v2;
	}

	printf("结果是:%f\n",v1.sum());
	return 0;
}

