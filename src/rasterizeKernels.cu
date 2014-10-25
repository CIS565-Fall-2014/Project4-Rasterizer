// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

glm::vec3 *framebuffer;
fragment *depthbuffer;
float *device_vbo;
float *device_cbo;
int *device_ibo;
triangle* primitives;
float *device_vbo_window_coords;

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//Writes a given fragment to a fragment buffer at a given location
__host__ __device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    depthbuffer[index] = frag;
  }
}

//Reads a fragment from a given location in a fragment buffer
__host__ __device__ fragment getFromDepthbuffer(int x, int y, fragment* depthbuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return depthbuffer[index];
  }else{
    fragment f;
    return f;
  }
}

//Writes a given pixel to a pixel buffer at a given location
__host__ __device__ void writeToFramebuffer(int x, int y, glm::vec3 value, glm::vec3* framebuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    framebuffer[index] = value;
  }
}

//Reads a pixel from a pixel buffer at a given location
__host__ __device__ glm::vec3 getFromFramebuffer(int x, int y, glm::vec3* framebuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return framebuffer[index];
  }else{
    return glm::vec3(0,0,0);
  }
}

//Kernel that clears a given pixel buffer with a given color
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image, glm::vec3 color){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = color;
    }
}

//Kernel that clears a given fragment buffer with a given fragment
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment frag){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      fragment f = frag;
      f.position.x = x;
      f.position.y = y;
      buffer[index] = f;
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;      
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}


/*********** DANNY'S PRIMARY CONTRIBUTION - START ***********/

// TODO: Implement a vertex shader.
// Convert vertices from model-space to clip-space.
__global__
void vertexShadeKernel( float *vbo,
						int vbosize,
						glm::mat4 mvp_matrix,
						glm::vec2 resolution,
						float *vbo_window_coords )
{
	int index = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	// Divide by 3 because each vertex has 3 components (x, y, and z).
	if ( index < vbosize / 3 ) {
		// Create point to transform.
		int vbo_index = index * 3;
		glm::vec4 v( vbo[vbo_index + 0], vbo[vbo_index + 1], vbo[vbo_index + 2], 1.0f );

		// Transform point from object-space to clip-space by multiplying by the composite model, view, projection matrices.
		glm::vec4 vt = mvp_matrix * v;

		// Transform point to NDC-space by dividing x-, y-, and z-components by w-component (perspective division).
		// [-1, 1].
		glm::vec3 v_ndc( vt.x / vt.w, vt.y / vt.w, vt.z / vt.w );

		// Transform x and y range from [-1, 1] to [0, 1].
		glm::vec2 v_remapped( ( v_ndc.x + 1.0f ) / 2.0f, ( v_ndc.y + 1.0f ) / 2.0f );

		// Transform x- and y-coordinates to window-space.
		glm::vec2 v_window( v_remapped.x * resolution.x, v_remapped.y * resolution.y );

		// Save transformed vertices.
		vbo_window_coords[vbo_index + 0] = v_window.x;
		vbo_window_coords[vbo_index + 1] = v_window.y;
		vbo_window_coords[vbo_index + 2] = v_ndc.z;
	}
}


// TODO: Implement primative assembly.
__global__
void primitiveAssemblyKernel( float *vbo,
							  int vbosize,
							  float *cbo,
							  int cbosize,
							  int *ibo,
							  int ibosize,
							  triangle *primitives )
{

	// TODO: Convert vertices to primitives (triangles or fragments).

	int index = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int primitivesCount = ibosize / 3;
	if ( index < primitivesCount ) {

	}
}


// TODO: Implement a rasterization method, such as scanline.
__global__
void rasterizationKernel( triangle *primitives,
						  int primitivesCount,
						  fragment *depthbuffer,
						  glm::vec2 resolution )
{

	// TODO: Which pixels does a primitive cover?

	int index = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	if ( index < primitivesCount ) {

	}
}


// TODO: Implement a fragment shader.
__global__
void fragmentShadeKernel( fragment *depthbuffer,
						  glm::vec2 resolution )
{

	// TODO: How does light interact with a fragment?
	// TODO: Write pixel color to frame buffer.

	int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
	int index = x + ( y * resolution.x );
	if ( x <= resolution.x && y <= resolution.y ) {

	}
}

/*********** DANNY'S PRIMARY CONTRIBUTION - END ***********/


//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
    framebuffer[index] = depthbuffer[index].color;
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore( uchar4 *PBOpos,
						float frame,
						float *vbo,
						int vbosize,
						float *cbo,
						int cbosize,
						int *ibo,
						int ibosize,
						simpleCamera camera )
{
	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock( tileSize,
						  tileSize );
	dim3 fullBlocksPerGrid( ( int )ceil( ( float )camera.resolution.x / ( float )tileSize ),
							( int )ceil( ( float )camera.resolution.y / ( float )tileSize ) );

	// set up framebuffer
	framebuffer = NULL;
	cudaMalloc( ( void** )&framebuffer,
				( int )camera.resolution.x * ( int )camera.resolution.y * sizeof( glm::vec3 ) );
  
	// set up depthbuffer
	depthbuffer = NULL;
	cudaMalloc( ( void** )&depthbuffer,
				( int )camera.resolution.x * ( int )camera.resolution.y * sizeof( fragment ) );

	// kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
	clearImage<<< fullBlocksPerGrid, threadsPerBlock >>>( camera.resolution,
														  framebuffer,
														  glm::vec3( 0.0f, 0.0f, 0.0f ) );
  
	fragment frag;
	frag.color = glm::vec3( 0.0f, 0.0f, 0.0f );
	frag.normal = glm::vec3( 0.0f, 0.0f, 0.0f );
	frag.position = glm::vec3( 0.0f, 0.0f, -10000.0f );
	clearDepthBuffer<<< fullBlocksPerGrid, threadsPerBlock >>>( camera.resolution,
																depthbuffer,
																frag );

	//------------------------------
	// memory stuff
	//------------------------------
	primitives = NULL;
	cudaMalloc( ( void** )&primitives,
				( ibosize / 3 ) * sizeof( triangle ) );

	device_ibo = NULL;
	cudaMalloc( ( void** )&device_ibo,
				ibosize * sizeof( int ) );
	cudaMemcpy( device_ibo,
				ibo,
				ibosize * sizeof( int ),
				cudaMemcpyHostToDevice );

	device_vbo = NULL;
	cudaMalloc( ( void** )&device_vbo,
				vbosize * sizeof( float ) );
	cudaMemcpy( device_vbo,
				vbo,
				vbosize * sizeof( float ),
				cudaMemcpyHostToDevice );

	device_vbo_window_coords = NULL;
	cudaMalloc( ( void** )&device_vbo_window_coords,
				vbosize * sizeof( float ) );

	device_cbo = NULL;
	cudaMalloc( ( void** )&device_cbo,
				cbosize * sizeof( float ) );
	cudaMemcpy( device_cbo,
				cbo,
				cbosize * sizeof( float ),
				cudaMemcpyHostToDevice );

	tileSize = 32;
	int primitiveBlocks = ceil( ( ( float )vbosize / 3 ) / ( ( float )tileSize ) );

	//------------------------------
	// vertex shader
	//------------------------------

	// Define model matrix.
	// Transforms from object-space to world-space.
	glm::mat4 model_matrix( 1.0f ); // Identity matrix.
	
	// Define view matrix.
	// Transforms from world-space to camera-space.
	glm::mat4 view_matrix = glm::lookAt( camera.position,
										 camera.target,
										 camera.up );

	// Define projection matrix.
	// Transforms from camera-space to clip-space.
	glm::mat4 projection_matrix = glm::perspective( camera.fov_y,
													camera.resolution.x / camera.resolution.y,
													camera.near_clip,
													camera.far_clip );

	vertexShadeKernel<<< primitiveBlocks, tileSize >>>( device_vbo,
														vbosize,
														projection_matrix * view_matrix * model_matrix,
														camera.resolution,
														device_vbo_window_coords );
	cudaDeviceSynchronize();

	//------------------------------
	// primitive assembly
	//------------------------------
	primitiveBlocks = ceil( ( ( float )ibosize / 3 ) / ( ( float )tileSize ) );
	primitiveAssemblyKernel<<< primitiveBlocks, tileSize >>>( device_vbo,
															  vbosize,
															  device_cbo,
															  cbosize,
															  device_ibo,
															  ibosize,
															  primitives );
	cudaDeviceSynchronize();

	//------------------------------
	// rasterization
	//------------------------------
	rasterizationKernel<<< primitiveBlocks, tileSize >>>( primitives,
														  ibosize / 3,
														  depthbuffer,
														  camera.resolution );
	cudaDeviceSynchronize();

	//------------------------------
	// fragment shader
	//------------------------------
	fragmentShadeKernel<<< fullBlocksPerGrid, threadsPerBlock >>>( depthbuffer,
																   camera.resolution );
	cudaDeviceSynchronize();

	//------------------------------
	// write fragments to framebuffer
	//------------------------------
	render<<< fullBlocksPerGrid, threadsPerBlock >>>( camera.resolution,
													  depthbuffer,
													  framebuffer );
	sendImageToPBO<<< fullBlocksPerGrid, threadsPerBlock >>>( PBOpos,
															  camera.resolution,
															  framebuffer );
	cudaDeviceSynchronize();

	kernelCleanup();
	checkCUDAError("Kernel failed!");
}

void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
  cudaFree( device_vbo_window_coords );
}