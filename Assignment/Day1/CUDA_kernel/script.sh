#!/bin/bash
clear
echo EXERCISE 2:First CUDA Kernel

nvcc -o addArrays addArrays.cu
nvcc -o squareArray squareArray.cu
make 


./addArrays
./squareArray


echo
echo

