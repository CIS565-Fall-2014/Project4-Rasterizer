#!/bin/bash
export DYLD_LIBRARY_PATH='/usr/local/cuda/lib:glfw/lib';
./bin/rasterizer "$1" 
