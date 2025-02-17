#include <iostream>
#include "vec.h"
#include <unistd.h>
#include <time.h>




#define N 100000000
int main(int argc,const char* argv[])
{
	float* a = new float[N];
	for(int i=0;i<N;++i){
		a[i] = 0.42;
	}
	float* b = new float[N];
	for(int i=0;i<N;++i){
		b[i] = 0.00008;
	}

	vec v1(a,N);
	vec v2(b,N);

	clock_t start = clock();
	for(int i=0;i<1000;++i){
		v1 += v2;
	}
	double sum = v1.sum();
	clock_t end = clock();
	double milliseconds = (double)(end - start) / CLOCKS_PER_SEC * 1000;
	printf("结果是:%f,耗时:%f\n",sum,milliseconds);
	return 0;
}

