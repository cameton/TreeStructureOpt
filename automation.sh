#!/bin/bash
for i in 1 2 4 8 16 32 64 128 256 
do 
    JULIA_NUM_THREADS=$i julia MDTest_iter.jl
done
