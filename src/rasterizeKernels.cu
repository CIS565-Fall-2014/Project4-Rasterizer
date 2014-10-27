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
float *device_nbo;
triangle* primitives;
float *device_vbo_window_coords;
int *device_lock_buffer;

const float EMPTY_BUFFER_DEPTH = 10000.0f;

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

__global__
void clearLockBuffer( glm::vec2 resolution, int *lock_buffer )
{
	int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
	int index = x + ( y * resolution.x );
	if( x <= resolution.x && y <= resolution.y ) {
		lock_buffer[index] = 0;
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

// Convert vertices from object-space coordinates to window coordinates.
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
		int vbo_index = index * 3;

		// Create point to transform.
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


template<typename T>
__host__
__device__
void simpleSwap( T &f1, T &f2 )
{
	T tmp = f1;
	f1 = f2;
	f2 = tmp;
}


// Construct primitives from vertices.
__global__
void primitiveAssemblyKernel( float *vbo, int vbosize,
							  float *cbo, int cbosize,
							  int *ibo, int ibosize,
							  float *nbo, int nbosize,
							  float *vbo_window_coords,
							  triangle *primitives )
{
	int index = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int primitivesCount = ibosize / 3;
	if ( index < primitivesCount ) {
		// Get indices of triangle vertices.
		int ibo_index = index * 3;
		int i0 = ibo[ibo_index + 0];
		int i1 = ibo[ibo_index + 1];
		int i2 = ibo[ibo_index + 2];

		// Get positions of triangle vertices.
		int v0_index = i0 * 3;
		int v1_index = i1 * 3;
		int v2_index = i2 * 3;

		// Get screen-space positions of triangle vertices.
		glm::vec3 ssp0( vbo_window_coords[v0_index + 0], vbo_window_coords[v0_index + 1], vbo_window_coords[v0_index + 2] );
		glm::vec3 ssp1( vbo_window_coords[v1_index + 0], vbo_window_coords[v1_index + 1], vbo_window_coords[v1_index + 2] );
		glm::vec3 ssp2( vbo_window_coords[v2_index + 0], vbo_window_coords[v2_index + 1], vbo_window_coords[v2_index + 2] );

		// Check if triangle is visible.
		glm::vec3 backface_check = glm::cross( ssp1 - ssp0, ssp2 - ssp0 );
		if ( backface_check.z < 0.0f ) {
			triangle tri;
			tri.is_visible = false;
			primitives[index] = tri;
			return;
		}

		// Get positions of triangle vertices.
		glm::vec3 p0( vbo[v0_index + 0], vbo[v0_index + 1], vbo[v0_index + 2] );
		glm::vec3 p1( vbo[v1_index + 0], vbo[v1_index + 1], vbo[v1_index + 2] );
		glm::vec3 p2( vbo[v2_index + 0], vbo[v2_index + 1], vbo[v2_index + 2] );

		// Get colors of triangle vertices.
		int c0_index = ( i0 % 3 ) * 3;
		int c1_index = ( i1 % 3 ) * 3;
		int c2_index = ( i2 % 3 ) * 3;
		glm::vec3 c0( cbo[c0_index + 0], cbo[c0_index + 1], cbo[c0_index + 2] );
		glm::vec3 c1( cbo[c1_index + 0], cbo[c1_index + 1], cbo[c1_index + 2] );
		glm::vec3 c2( cbo[c2_index + 0], cbo[c2_index + 1], cbo[c2_index + 2] );

		// Get normals of triangle vertices.
		glm::vec3 n0( nbo[v0_index + 0], nbo[v0_index + 1], nbo[v0_index + 2] );
		glm::vec3 n1( nbo[v1_index + 0], nbo[v1_index + 1], nbo[v1_index + 2] );
		glm::vec3 n2( nbo[v2_index + 0], nbo[v2_index + 1], nbo[v2_index + 2] );

		// Set triangle.
		primitives[index] = triangle( p0, p1, p2,
									  ssp0, ssp1, ssp2,
									  c0, c1, c2,
									  n0, n1, n2 );
	}
}


// Scanline rasterization per triangle.
// See http://graphics.stanford.edu/courses/cs248-08/scan/scan1.html for a similar, but slightly different rasterization method.
__global__
void rasterizationKernel( triangle *primitives,
						  int primitivesCount,
						  fragment *depthbuffer,
						  glm::vec2 resolution,
						  int *lock_buffer )
{
	int index = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	if ( index < primitivesCount ) {
		triangle tri = primitives[index];

		// Only rasterize current triangle if triangle is visible (determined in primitive assembly stage).
		if ( !tri.is_visible ) {
			return;
		}

	    glm::vec3 aabb_min;
	    glm::vec3 aabb_max;
	    getAABBForTriangle( tri.ssp0, tri.ssp1, tri.ssp2, aabb_min, aabb_max );

		// TODO: Clip AABB boxes outside render resolution.

		for ( int y = ceil( aabb_min.y ); y < ceil( aabb_max.y ); ++y ) {
			for ( int x = ceil( aabb_min.x ); x < ceil( aabb_max.x ); ++x ) {

				// Compute Barycentric coordinates of current fragment in screen-space triangle.
				glm::vec3 barycentric_coordinates = calculateBarycentricCoordinate( tri.ssp0, tri.ssp1, tri.ssp2, glm::vec2( x, y ) );

				// (x, y) point is outside triangle.
				if ( barycentric_coordinates.x < 0.0f || barycentric_coordinates.y < 0.0f || barycentric_coordinates.z < 0.0f ) {
					continue;
				}

				float current_z = getZAtCoordinate( barycentric_coordinates, tri.p0, tri.p1, tri.p2 );

				fragment buffer_fragment = getFromDepthbuffer( x, y, depthbuffer, resolution );
				float buffer_z = buffer_fragment.position.z;

				// Update depth buffer atomically.
				if ( current_z < buffer_z ) {
					int current_index = ( y * resolution.x ) + x;
					bool is_waiting_to_update = true;
					while ( is_waiting_to_update ) {
						if ( atomicExch( &lock_buffer[current_index], 1 ) == 0 ) {
							fragment f;
							//f.color = ( tri.c0 * barycentric_coordinates.x ) + ( tri.c1 * barycentric_coordinates.y ) + ( tri.c2 * barycentric_coordinates.z );
							f.color = glm::vec3( 0.5f, 0.5f, 0.5f );
							f.normal = glm::normalize( ( tri.n0 * barycentric_coordinates.x ) + ( tri.n1 * barycentric_coordinates.y ) + ( tri.n2 * barycentric_coordinates.z ) );					
							f.position = ( tri.p0 * barycentric_coordinates.x ) + ( tri.p1 * barycentric_coordinates.y ) + ( tri.p2 * barycentric_coordinates.z );
							writeToDepthbuffer( x, y, f, depthbuffer, resolution );

							// Release lock.
							atomicExch( &lock_buffer[current_index], 0 );
							is_waiting_to_update = false;
						}
					}
				}
			}
		}
	}
}


// Compute light interaction with fragments.
// Write fragment colors to frame buffer.
// Diffuse Lambertian shading.
__global__
void fragmentShadeKernel( fragment *depthbuffer,
						  glm::vec2 resolution )
{
	int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
	int index = x + ( y * resolution.x );
	if ( x <= resolution.x && y <= resolution.y ) {
		fragment f = depthbuffer[index];

		glm::vec3 light_pos_1( -10.0f, 0.0f, 10.0f );
		float light_intensity_1 = 2.0f;
		glm::vec3 light_1_contribution = max( glm::dot( f.normal, glm::normalize( light_pos_1 - f.position )), 0.0f ) * depthbuffer[index].color * light_intensity_1;

		glm::vec3 light_pos_2( 10.0f, 0.0f, -10.0f );
		float light_intensity_2 = 2.0f;
		glm::vec3 light_2_contribution = max( glm::dot( f.normal, glm::normalize( light_pos_2 - f.position )), 0.0f ) * depthbuffer[index].color * light_intensity_2;
		
		depthbuffer[index].color = light_1_contribution + light_2_contribution;

		//depthbuffer[index].color = max( glm::dot( f.normal, glm::normalize( light_pos - f.position )), 0.0f ) * depthbuffer[index].color * light_intensity;
	}
}


__host__
__device__
float computeDistanceBetweenTwoColors( glm::vec3 p1, glm::vec3 p2 )
{
	return sqrt( ( p2.x - p1.x ) * ( p2.x - p1.x ) + ( p2.y - p1.y ) * ( p2.y - p1.y ) + ( p2.z - p1.z ) * ( p2.z - p1.z ) );
}


__host__
__device__
bool shouldBlurPixel( int x, int y,
					  fragment *depthbuffer,
					  glm::vec2 resolution )
{
	if ( x <= resolution.x && y <= resolution.y ) {
		const float threshold = 0.25f;

		glm::vec3 p1 = depthbuffer[x + ( y * ( int )resolution.x )].color;
		int i, j;

		// Left.
		i = x - 1;
		j = y;
		if ( i > 0 && i <= resolution.x && j > 0 && j <= resolution.y ) {
			glm::vec3 p2 = depthbuffer[i + ( j * ( int )resolution.x )].color;
			if ( computeDistanceBetweenTwoColors( p1, p2 ) > threshold ) {
				return true;
			}
		}

		// Top.
		i = x;
		j = y - 1;
		if ( i > 0 && i <= resolution.x && j > 0 && j <= resolution.y ) {
			glm::vec3 p2 = depthbuffer[i + ( j * ( int )resolution.x )].color;
			if ( computeDistanceBetweenTwoColors( p1, p2 ) > threshold ) {
				return true;
			}
		}

		// Right.
		i = x + 1;
		j = y;
		if ( i > 0 && i <= resolution.x && j > 0 && j <= resolution.y ) {
			glm::vec3 p2 = depthbuffer[i + ( j * ( int )resolution.x )].color;
			if ( computeDistanceBetweenTwoColors( p1, p2 ) > threshold ) {
				return true;
			}
		}

		// Bottom.
		i = x;
		j = y + 1;
		if ( i > 0 && i <= resolution.x && j > 0 && j <= resolution.y ) {
			glm::vec3 p2 = depthbuffer[i + ( j * ( int )resolution.x )].color;
			if ( computeDistanceBetweenTwoColors( p1, p2 ) > threshold ) {
				return true;
			}
		}
	}

	return false;
}

__global__
void antiAliasingPostProcess( fragment *depthbuffer,
							  glm::vec2 resolution )
{
	int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
	int index = x + ( y * resolution.x );
	if ( x <= resolution.x && y <= resolution.y ) {
		if ( shouldBlurPixel( x, y, depthbuffer, resolution ) ) {
			int pixel_count = 0;
			glm::vec3 sum( 0.0f, 0.0f, 0.0f );
			for ( int i = x - 1; i < x + 1; ++i ) {
				for ( int j = y - 1; j < y + 1; ++j ) {
					if ( i > 0 && i <= resolution.x && j > 0 && j <= resolution.y ) {
						sum += depthbuffer[i + ( j * ( int )resolution.x )].color;
						++pixel_count;
					}
				}
			}
			depthbuffer[index].color = glm::vec3( sum.x / pixel_count, sum.y / pixel_count, sum.z / pixel_count );
			//depthbuffer[index].color = glm::vec3( 1.0f, 0.0f, 0.0f );
		}
	}
}

/*********** DANNY'S PRIMARY CONTRIBUTION - END ***********/

// Write fragment colors to the framebuffer.
__global__
void render( glm::vec2 resolution, fragment *depthbuffer, glm::vec3 *framebuffer )
{
	int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
	int index = x + ( y * resolution.x );

	if ( x <= resolution.x && y <= resolution.y ) {
		framebuffer[index] = depthbuffer[index].color;
	}
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore( uchar4 *PBOpos,
						float frame,
						float *vbo, int vbosize,
						float *cbo, int cbosize,
						int *ibo, int ibosize,
						float *nbo, int nbosize,
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
	frag.position = glm::vec3( 0.0f, 0.0f, EMPTY_BUFFER_DEPTH );
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

	device_lock_buffer = NULL;
	cudaMalloc( ( void** )&device_lock_buffer,
				( int )camera.resolution.x * ( int )camera.resolution.y * sizeof( int ) );

	device_cbo = NULL;
	cudaMalloc( ( void** )&device_cbo,
				cbosize * sizeof( float ) );
	cudaMemcpy( device_cbo,
				cbo,
				cbosize * sizeof( float ),
				cudaMemcpyHostToDevice );

	device_nbo = NULL;
	cudaMalloc( ( void** )&device_nbo,
				nbosize * sizeof( float ) );
	cudaMemcpy( device_nbo,
				nbo,
				nbosize * sizeof( float ),
				cudaMemcpyHostToDevice );

	tileSize = 32;
	int primitiveBlocks = ceil( ( ( float )vbosize / 3 ) / ( ( float )tileSize ) );

	//------------------------------
	// initialize lock buffer
	//------------------------------
	clearLockBuffer<<< fullBlocksPerGrid, threadsPerBlock >>>( camera.resolution,
															   device_lock_buffer );
	cudaDeviceSynchronize();

	//------------------------------
	// vertex shader
	//------------------------------

	// Define model matrix.
	// Transforms from object-space to world-space.
	glm::mat4 model_matrix( 1.0f ); // Identity matrix.
	//glm::mat4 model_matrix = glm::rotate( glm::mat4( 1.0f ), frame * 2, glm::vec3( 0.0f, 1.0f, 0.0f ));
	
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

	vertexShadeKernel<<< primitiveBlocks, tileSize >>>( device_vbo, vbosize,
														projection_matrix * view_matrix * model_matrix,
														camera.resolution,
														device_vbo_window_coords );
	cudaDeviceSynchronize();

	//------------------------------
	// primitive assembly
	//------------------------------
	primitiveBlocks = ceil( ( ( float )ibosize / 3 ) / ( ( float )tileSize ) );
	primitiveAssemblyKernel<<< primitiveBlocks, tileSize >>>( device_vbo, vbosize,
															  device_cbo, cbosize,
															  device_ibo, ibosize,
															  device_nbo, nbosize,
															  device_vbo_window_coords,
															  primitives );
	cudaDeviceSynchronize();

	//------------------------------
	// rasterization
	//------------------------------
	rasterizationKernel<<< primitiveBlocks, tileSize >>>( primitives,
														  ibosize / 3,
														  depthbuffer,
														  camera.resolution,
														  device_lock_buffer );
	cudaDeviceSynchronize();

	//------------------------------
	// fragment shader
	//------------------------------
	fragmentShadeKernel<<< fullBlocksPerGrid, threadsPerBlock >>>( depthbuffer,
																   camera.resolution );
	cudaDeviceSynchronize();

	//------------------------------
	// anti-aliasing
	//------------------------------
	//antiAliasingPostProcess<<< fullBlocksPerGrid, threadsPerBlock >>>( depthbuffer,
	//																   camera.resolution );
	//cudaDeviceSynchronize();

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
  cudaFree( device_nbo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
  cudaFree( device_vbo_window_coords );
  cudaFree( device_lock_buffer );
}