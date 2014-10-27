EXEC := "build/Project4-Rasterizer"
ARGS := "mesh=objs/cow.obj"

.PHONY: all debug release run run-debug format clean

all: debug

debug:
	(mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=debug && make)

release:
	(mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=release && make)

run:
	${EXEC} ${ARGS}

run-debug:
	CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1 cuda-gdb --args ${EXEC} ${ARGS}

format:
	astyle --mode=c --style=1tbs -pcHs4 -r 'src/*.cpp' 'src/*.h' 'src/*.cu' 'src/*.h'

clean:
	rm -rf build
