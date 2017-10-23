#!/bin/bash
clear
mkdir ./result

cd ./result/

cmake ../
make 
clear
echo FINAL PROJECT:Blind DeConvolution

./main -i ../postcard.png


echo
echo

