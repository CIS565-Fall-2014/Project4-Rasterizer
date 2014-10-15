#All cuda paths are for v5.0+

NVCC     = nvcc -m64
CC       = gcc -m64
CPPC     = g++ -m64

CUDA_FLAGS = -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/include

LFLAGS = -lglut -lGL -lGLEW 

all: 565rasterizer

rasterizeKernels.o: ../src/rasterizeKernels.cu ../src/rasterizeKernels.h ../src/rasterizeTools.h ../src/utilities.h ../src/cudaMat4.h 
	$(NVCC) $(CUDA_FLAGS) -c ../src/rasterizeKernels.cu

glslUtility.o: ../src/glslUtility.cpp ../src/glslUtility.h  
	$(CPPC) $(CUDA_FLAGS) -c ../src/glslUtility.cpp

utilities.o: ../src/utilities.cpp ../src/utilities.h ../src/cudaMat4.h 
	$(CPPC) $(CUDA_FLAGS) -c ../src/utilities.cpp

main.o: ../src/main.cpp ../src/main.h ../src/rasterizeKernels.h ../src/utilities.h ../src/cudaMat4.h 
	$(CPPC) $(CUDA_FLAGS) -c ../src/main.cpp

obj.o: ../src/ObjCore/obj.cpp
	$(CPPC) -c ../src/ObjCore/obj.cpp

objloader.o: ../src/ObjCore/objloader.cpp
	$(CPPC) -c ../src/ObjCore/objloader.cpp

565rasterizer: main.o rasterizeKernels.o glslUtility.o utilities.o obj.o objloader.o
	$(NVCC) $(LFLAGS) *.o -o 565rasterizer


clean:
	rm *.o 
	rm 565rasterizer

test: 565rasterizer
	./565rasterizer
