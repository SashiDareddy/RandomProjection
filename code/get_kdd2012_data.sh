#!/bin/bash

#download kdd2012 sample datasets from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
echo "Downloading training file..."
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdd12.tr.bz2 -P ../data
echo "Downloading validation file..."
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdd12.val.bz2 -P ../data

echo "unzipping archives.."
bunzip2 ../data/kdd12.tr.bz2
bunzip2 ../data/kdd12.val.bz2