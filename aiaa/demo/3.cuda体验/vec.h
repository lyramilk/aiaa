#ifndef _VEC_H_
#define _VEC_H_


class vec
{
	float* gpudata;
	int size;
public:
	vec(float* nums,int count);
	virtual ~vec();
	vec& operator +=(const vec& b);

	double sum();

};



#endif 