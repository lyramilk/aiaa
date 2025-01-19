nvcc vec.cu -o vec.o -c
g++ test.cc -o test.o -c
nvcc vec.o test.o -o testcuda
./testcuda