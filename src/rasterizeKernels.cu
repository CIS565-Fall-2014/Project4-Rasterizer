// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
int* device_ibo;
float* device_nbo;
triangle* primitives;
glm::vec3* device_tbo;
cudaEvent_t start, stop;
float timeDuration;


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

__host__ __device__ glm::vec3 findTextureColor(glm::vec3 point, glm::vec3 * texColors, tex theTex){
	//int id = theTex.id;
    //int size = theTex.h * theTex.w;
	//printf("point: %.2f, %.2f, %.2f\n", point.r, point.g, point.b);
	glm::vec3 result;
	float u,v; 
	if( fabs( point.x - 0.1f ) < 0.002f){
		
		u = point.y + 0.5f;
		v = point.z + 0.5f;
	}
	else if( fabs( point.y - 0.1f )< 0.002f){
	
		u = point.x + 0.5f;
		v = point.z + 0.5f;
	}
	else if( fabs( point.z - 0.1f ) < 0.002f){
	
		u = point.x + 0.5f;
		v = point.y + 0.5f;
	}

	//int idx = ( (int) v * theTex.w )* theTex.h + (int)u * theTex.h;

	//result.r = texColors[idx].r/255.0f;
	//result.g = texColors[idx].g/255.0f;
	//result.b = texColors[idx].b/255.0f;
	//printf("texture color: %.2f, %.2f, %.2f\n", result.r, result.g, result.b);
	return result;
}

__host__ __device__ glm::vec3 getTextureColor(glm::vec3 &point, triangle &thePoly,  glm::vec3 * texColors, tex &theTex){
	glm::vec3 result(0.0f, 0.0f, 0.0f);
	triangle newPoly = thePoly;
	//Shift XY coordinate system (+0.5, +0.5) to match the subpixeling technique
	newPoly.p0 += glm::vec3(0.5f, 0.5f, 0.0f);
	newPoly.p1 += glm::vec3(0.5f, 0.5f, 0.0f);
	newPoly.p2 += glm::vec3(0.5f, 0.5f, 0.0f);

	//Calculate alternative 1/Z, U/Z and V/Z values which will be interpolated
	newPoly.uv0 /= newPoly.p0.z;
	newPoly.uv1 /= newPoly.p1.z;
	newPoly.uv2 /= newPoly.p2.z;

	// Sort the vertices in ascending Y order
	glm::vec3 tempf;
	#define swapPoint(m, n) tempf = m; m = n; n = tempf;
		if (newPoly.p0.y > newPoly.p1.y)  //swap p0 and p1
			swapPoint(newPoly.p0, newPoly.p1);
		if (newPoly.p0.y > newPoly.p2.y) //swap p0 and p2
			swapPoint(newPoly.p0, newPoly.p2);
		if (newPoly.p1.y > newPoly.p2.y) //swap p1 and p2
			swapPoint(newPoly.p1, newPoly.p2);
	#undef swapPoint

	float x0 = newPoly.p0.x;
	float y0 = newPoly.p0.y;
	float z0 = newPoly.p0.z;
	float x1 = newPoly.p1.x;
	float y1 = newPoly.p1.y;
	float z1 = newPoly.p1.z;
	float x2 = newPoly.p2.x;
	float y2 = newPoly.p2.y;
	float z2 = newPoly.p2.z;
	float y0i = y0;
	float y1i = y1;
	float y2i = y2;
	printf("swappded points: %.2f, %.2f, %.2f", y0, y1, y2);

	if ((y0i == y1i && y0i == y2i)
	    || ((int) x0 == (int) x1 && (int) x0 == (int) x2))
		return result;
	// Calculate horizontal and vertical increments for UV axes
	float denom = ((x2 - x0) * (y1 - y0) - (x1 - x0) * (y2 - y0));

	if (!denom)		// Skip if it's an infinitely thin line
		return result;	

	denom = 1 / denom;	
	glm::vec3 duv_dx, duv_dy;  //  (  d(1/z), d(u/z), d(v/z) )
	duv_dx.x = ((newPoly.uv2.x - newPoly.uv0.x) * (y1 - y0) - (newPoly.uv1.x - newPoly.uv0.x) * (y2 - y0)) * denom;
	duv_dx.y = ((newPoly.uv2.y - newPoly.uv0.y) * (y1 - y0) - (newPoly.uv1.y - newPoly.uv0.y) * (y2 - y0)) * denom;
	duv_dx.z = ((newPoly.uv2.z - newPoly.uv0.z) * (y1 - y0) - (newPoly.uv1.z - newPoly.uv0.z) * (y2 - y0)) * denom;
	duv_dy.x = ((newPoly.uv2.x - newPoly.uv0.x) * (x2 - x0) - (newPoly.uv2.x - newPoly.uv0.x) * (x1 - x0)) * denom;
	duv_dy.y = ((newPoly.uv2.y - newPoly.uv0.y) * (x2 - x0) - (newPoly.uv2.y - newPoly.uv0.y) * (x1 - x0)) * denom;
	duv_dy.z = ((newPoly.uv2.z - newPoly.uv0.z) * (x2 - x0) - (newPoly.uv2.z - newPoly.uv0.z) * (x1 - x0)) * denom;
	
	// Calculate X-slopes along the edges
	float dx_dy0, dx_dy1, dx_dy2;
	if (y1 > y0)
		dx_dy0 = (x1 - x0) / (y1 - y0);
	if (y2 > y0)
		dx_dy1 = (x2 - x0) / (y2 - y0);
	if (y2 > y1)
		dx_dy2 = (x2 - x1) / (y2 - y1);

	// Determine which side of the poly the longer edge is on

	int side = dx_dy1 > dx_dy0;

	if (y0 == y1)
		side = x0 > x1;
	if (y1 == y2)
		side = x2 > x1;
	return result;
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

//TODO: Implement a vertex shader
__global__ void vertexShadeKernel(float* vbo, int vbosize, float* nbo, int nbosize, cudaMat4 M_mvp, cudaMat4 M_mv_prime){

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  //Transform incoming vertex position from model to clip coordinates
	  glm::vec3 pModel(vbo[index*3], vbo[index*3 + 1],vbo[index*3 + 2]);
	  glm::vec3 pClip = multiplyMV(M_mvp, glm::vec4(pModel, 1.0f));

	  //Transform normal into clip coordinates
	  glm::vec3 nModel(nbo[index*3], nbo[index*3 + 1],nbo[index*3 + 2]);
	 
	  /*glm::vec3 nTip_OS = pModel + nModel;
	  glm::vec3 nTip_WS = multiplyMV(theCam.M_mvp, glm::vec4(nTip_OS, 1.0f));
	  glm::vec3 nClip = glm::normalize(nTip_WS - pClip);*/
	 // glm::vec3 nClip = glm::normalize( multiplyMV(theCam.M_mvp, glm::vec4(nModel, 0.0f)));
	  glm::vec3 nClip = glm::normalize( multiplyMV(M_mv_prime, glm::vec4(nModel, 0.0f)));
	  
	  vbo[index*3] = pClip.x; 
	  vbo[index*3 + 1] = pClip.y; 
	  vbo[index*3 + 2] = pClip.z;

	  nbo[index*3] = nClip.x; 
	  nbo[index*3 + 1] = nClip.y; 
	  nbo[index*3 + 2] = nClip.z;

  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, triangle* primitives, int SHADING){
		
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  //get indice number
	  int i0 = ibo[index*3];
	  int i1 = ibo[index*3+1];
	  int i2 = ibo[index*3+2];

	  //assemble primitive points
	  primitives[index].p0 = glm::vec3(vbo[i0*3], vbo[i0*3+1], vbo[i0*3+2]);
	  primitives[index].p1 = glm::vec3(vbo[i1*3], vbo[i1*3+1], vbo[i1*3+2]);
	  primitives[index].p2 = glm::vec3(vbo[i2*3], vbo[i2*3+1], vbo[i2*3+2]);

	  //assemble primitive colors
	  if(SHADING == 5){  //original cbo
		  primitives[index].c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);
		  primitives[index].c1 = glm::vec3(cbo[3], cbo[4], cbo[5]);
		  primitives[index].c2 = glm::vec3(cbo[6], cbo[7], cbo[8]);
	  }
	  else{
		  primitives[index].c0 = glm::vec3(1,1,1);
		  primitives[index].c1 = glm::vec3(1,1,1);
		  primitives[index].c2 = glm::vec3(1,1,1);
	  }

	  //assemble primitive normals;
	  glm::vec3 n0 = glm::vec3(nbo[i0*3], nbo[i0*3+1], nbo[i0*3+2]);
	  glm::vec3 n1 = glm::vec3(nbo[i1*3], nbo[i1*3+1], nbo[i1*3+2]);
	  glm::vec3 n2 = glm::vec3(nbo[i2*3], nbo[i2*3+1], nbo[i2*3+2]);
	  primitives[index].n = (n0 + n1 + n2)/3.0f;

  }
}


//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, glm::vec3* texColor, int texSize, tex theTex, 
	int LINE, int SHADING){
	
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(index<primitivesCount){
	 
	  triangle tri = primitives[index];
	  
	  if(tri.n.z <0){ // back facing triangles
		  return;
	  }

	  //get bounding box for this triangle
	  glm::vec3 minpoint, maxpoint;   // in the screen coordinates, floats
	  getAABBForTriangle(tri, minpoint, maxpoint);
	  glm::vec2 minPoint, maxPoint;   //in image coodinates, ints
	  minPoint = screenToImage( glm::vec2(minpoint.x, minpoint.y), resolution);   //viewport transform
	  maxPoint = screenToImage( glm::vec2(maxpoint.x, maxpoint.y), resolution);
	  int xMin = (int)floor(minPoint.x);
	  int xMax = (int)ceil(maxPoint.x);
	  int yMin = (int)floor(maxPoint.y);
	  int yMax = (int)ceil(minPoint.y);
	  //printf("min = %.2f, %.2f; max = %.2f, %.2f\n", minpoint.x, minpoint.y, maxpoint.x, maxpoint.y);
	 // printf("min = %.2f, %.2f; max = %.2f, %.2f\n", minPoint.x, minPoint.y, maxPoint.x, maxPoint.y);

	 // clipping
	  xMin = ( xMin > 0.0f )? xMin : 0.0f;
	  yMin = ( yMin > 0.0f) ? yMin : 0.0f;
	  xMax = ( xMax < resolution.x -1) ? xMax : resolution.x -1;
	  yMax = ( yMax < resolution.y -1) ? yMax : resolution.y -1;

	  if(xMin<0 || yMin<0 || xMin>=resolution.x || yMin>=resolution.y)
		  return;
	  if(xMax<0 || yMax<0 || xMax>=resolution.x || yMax>=resolution.y)
		  return;

	  //scanline approach
	  for(int y = yMin; y < yMax; y++){  //top to down
		for(int x = xMin; x < xMax; x++){   //left to right
			 int pixelID = x + resolution.x*y;
			 glm::vec2 screenCoord = imageToScreen(glm::vec2(x,y),resolution);  //perspective transformation
			 glm::vec3 b = calculateBarycentricCoordinate(tri, screenCoord);  //barycentric coordinate for (x,y) pixel
			 
			 if(isBarycentricCoordInBounds(b)){  //p is in the triangle bounds
				 float z = getZAtCoordinate(b, tri);  //depth

				 if(Z_TEST == 1){  //do the depth test with atomic function
					// while(atomicCAS(&depthbuffer[pixelID].tested, 1, 0) != 1);  //until current fragment is tested
				 }
				 if(z > depthbuffer[pixelID].position.z && z <= 1.0f){
					 fragment frag = depthbuffer[pixelID];
					 frag.color = interpolateColor(b,tri);
					 /*frag.position = interpolatePosition(b,tri);
					 frag.position.z = z;*/
					 glm::vec3 point(screenCoord.x, screenCoord.y, z);
					 frag.position = point;
					 frag.normal = tri.n;
					 
					 if(LINE == 1){   //shade line color
						 glm::vec3 lineColor(0.0f,0.0f,1.0f);  //blue
						 glm::vec3 p = interpolatePosition(b,tri);
						 if(fabs(glm::dot(glm::normalize(tri.p0 - p), glm::normalize(tri.p0 - tri.p1))-1.0f)<0.0001f||
							 fabs(glm::dot(glm::normalize(tri.p1 - p), glm::normalize(tri.p1 - tri.p2))-1.0f)<0.0001f ||
							 fabs(glm::dot(glm::normalize(tri.p2 - p), glm::normalize(tri.p2 - tri.p0))-1.0f)<0.0001f ){
				
								 frag.color = lineColor;
								 frag.normal = glm::vec3(0.0f, 0.0f, 1.0f);
						 }
					 }

					 if(SHADING == 4){  // perspectively correct texture map
						 //http://www.lysator.liu.se/~mikaelk/doc/perspectivetexture/
						// http://chrishecker.com/Miscellaneous_Technical_Articles
						  //glm::vec3 p = multiplyMV(M_mvp_inverse, glm::vec4(depthbuffer[index].position, 1.0f));
						 // glm::vec3 p1 = multiplyMV(M_mvp_inverse, glm::vec4(primitives[index].p1, 1.0f));
						 // glm::vec3 p2 = multiplyMV(M_mvp_inverse, glm::vec4(primitives[index].p2, 1.0f));
						  //primitives[index].c0 = findTextureColor(p0, texColor, theTex);
						  //primitives[index].c1 = findTextureColor(p1, texColor, theTex);
						 // primitives[index].c2 = findTextureColor(p2, texColor, theTex);
					 }

					 depthbuffer[pixelID] = frag;
				 }
				// atomicExch(&depthbuffer[pixelID].tested, 1);
				 
			 }
		  }

	  }
	 

	  primitives[index] = tri;  //update
  }
}

// display points
__global__ void rasterizationPointsKernel(float* vbo, int vbosize, float * nbo, int nbosize, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  //find the point
	  glm::vec3 point(vbo[index*3], vbo[index*3+1], vbo[index*3+2]);
	  glm::vec3 normal(nbo[index*3], nbo[index*3+1], nbo[index*3+2]);
	  if(normal.z < 0)
		  return;

	  //locate the pixel
	  glm::vec2 pixel = screenToImage( glm::vec2(point.x, point.y), resolution);   //viewport transform
	  if(pixel.x<0 || pixel.y<0 || pixel.x>=resolution.x || pixel.y>=resolution.y)
		  return;
	  int pixelID = pixel.x + pixel.y * resolution.x;

	  //shade the point representation
	  if(point.z > depthbuffer[pixelID].position.z ){
		  glm::vec3 pointColor(1.0f, 1.0f, 0.0f);   //yellow
		 /* depthbuffer[pixelID].position = point;
		  depthbuffer[pixelID].color = pointColor;   
		  depthbuffer[pixelID].normal = glm::vec3(0.0f, 0.0f, 1.0f);*/
		  for(int i=pixel.x-1; i<=pixel.x+1; i++){
			  for(int j=pixel.y-1; j<=pixel.y+1; j++){
				  if(i<0 || j<0 || i>=resolution.x || j>=resolution.y)
						return;
				  int newpixelID = i + j * resolution.x;
				  depthbuffer[newpixelID].position = point;
				  depthbuffer[newpixelID].color = pointColor;   
				  depthbuffer[newpixelID].normal = glm::vec3(0.0f, 0.0f, 1.0f);
				 // atomicExch(&depthbuffer[pixelID].tested, 1);
			  }
		  }
	  }

  }
}



//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, int SHADING){
	//set up light
  glm::vec3 lightPos(500.0f, 500.0f, 1000.0f);    //add a light in the scene for shading

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
	  if(depthbuffer[index].normal.z<0)
		  return;
	  if(SHADING == 0){   //shade by normal
		  depthbuffer[index].color.r = glm::clamp(depthbuffer[index].normal.x, 0.0f, 1.0f);
		  depthbuffer[index].color.g = glm::clamp(depthbuffer[index].normal.y, 0.0f, 1.0f);
		  depthbuffer[index].color.b = glm::clamp(depthbuffer[index].normal.z, 0.0f, 1.0f);
	  }
	  else if(SHADING == 1){   //shade by depth
		  depthbuffer[index].color.r = glm::clamp(depthbuffer[index].position.z/1000.0f, 0.0f, 1.0f);
		  depthbuffer[index].color.g = glm::clamp(depthbuffer[index].position.z/1000.0f, 0.0f, 1.0f);
		  depthbuffer[index].color.b = glm::clamp(depthbuffer[index].position.z/1000.0f, 0.0f, 1.0f);
	  }
	  else if(SHADING == 2){    //diffuse shade
		  glm::vec3 lightDir = glm::normalize(lightPos - depthbuffer[index].position);
		  float cosTerm = glm::clamp(glm::dot(lightDir, depthbuffer[index].normal), 0.0f, 1.0f);
		  depthbuffer[index].color = glm::clamp(cosTerm * depthbuffer[index].color, 0.0f, 1.0f);
	  }
	  else if (SHADING == 3){  //blinn-phong shade
		  float coeff = 5.0f;
		  glm::vec3 lightDir = glm::normalize(lightPos - depthbuffer[index].position);
		  float cosTerm = glm::clamp(glm::dot(lightDir, depthbuffer[index].normal), 0.0f, 1.0f);
		  depthbuffer[index].color = glm::clamp( std::pow(cosTerm,coeff) * depthbuffer[index].color, 0.0f, 1.0f);
	  }
	  else{
		  depthbuffer[index].color =glm::clamp(depthbuffer[index].color, 0.0f, 1.0f);
	  }
  }
}


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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, 
	int* ibo, int ibosize, float * nbo, int nbosize){
  
	//set up camera,
 // cam theCam = mouseCam;

	//cuda timer event
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

  //set up framebuffer
  framebuffer = NULL;
  cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3));
  
  //set up depthbuffer
  depthbuffer = NULL;
  cudaMalloc((void**)&depthbuffer, (int)resolution.x*(int)resolution.y*sizeof(fragment));

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
  
  fragment frag;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,-10000);
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

  //------------------------------
  //memory stuff
  //------------------------------
  cudaEventRecord( start, 0 );
  primitives = NULL;
  cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle));

  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

   int tbosize = 0;
   device_tbo = NULL;
  if( SHADING_MODE == 5 && textureColor.size()!=0 ){   //texture map!!!
		  //establish color vector
		  tbosize = textureColor.size();
		  glm::vec3 * tbo = new glm::vec3[tbosize];
		  for(int i=0; i< tbosize; i++){
			  tbo[i] = textureColor[i];
		  }
		  cudaMalloc((void**)&device_tbo, tbosize*sizeof(glm::vec3));
          cudaMemcpy( device_tbo, tbo, tbosize*sizeof(glm::vec3), cudaMemcpyHostToDevice);
		  delete tbo;
  }

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));
  cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &timeDuration, start, stop );
	if(PERFORMANCE_MEASURE == 1){
		printf("\n\n*****************************************************\n");
		printf("Time Taken for set up memory : %f ms\n",timeDuration);
		printf("*****************************************************\n");
	}

  //------------------------------
  //vertex shader
  //------------------------------
  cudaEventRecord( start, 0 );
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, mouseCam.M_mvp, mouseCam.M_mv_prime);
  cudaDeviceSynchronize();
  cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &timeDuration, start, stop );
	if(PERFORMANCE_MEASURE == 1){
		printf("\n\n*****************************************************\n");
		printf("Time Taken for vertex shader : %f ms\n",timeDuration);
		printf("*****************************************************\n");
	}
  //------------------------------
  //primitive assembly
  //------------------------------
	cudaEventRecord( start, 0 );
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, device_nbo, nbosize, primitives, SHADING_MODE);

  cudaDeviceSynchronize();
  cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &timeDuration, start, stop );
	if(PERFORMANCE_MEASURE == 1){
		printf("\n\n*****************************************************\n");
		printf("Time Taken for primitive assembly : %f ms\n",timeDuration);
		printf("*****************************************************\n");
	}
  //------------------------------
  //rasterization
  //------------------------------
	cudaEventRecord( start, 0 );
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, device_tbo, tbosize, textureMap, LINE_RASTER, SHADING_MODE);

  cudaDeviceSynchronize();
   cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &timeDuration, start, stop );
	if(PERFORMANCE_MEASURE == 1){
		printf("\n\n*****************************************************\n");
		printf("Time Taken for rasterization : %f ms\n",timeDuration);
		printf("*****************************************************\n");
	}
  //------------------------------
  //fragment shader
  //------------------------------
	cudaEventRecord( start, 0 );
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, SHADING_MODE);

  cudaDeviceSynchronize();
   cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &timeDuration, start, stop );
	if(PERFORMANCE_MEASURE == 1){
		printf("\n\n*****************************************************\n");
		printf("Time Taken for fragment shader : %f ms\n",timeDuration);
		printf("*****************************************************\n");
	}
  //------------------------------
  //point raster shader
  //------------------------------
  if(POINT_RASTER ==1){
	  cudaEventRecord( start, 0 );
	primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));
	rasterizationPointsKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, depthbuffer, resolution);   //render point out
	cudaDeviceSynchronize();
	 cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &timeDuration, start, stop );
	if(PERFORMANCE_MEASURE == 1){
		printf("\n\n*****************************************************\n");
		printf("Time Taken for point raster : %f ms\n",timeDuration);
		printf("*****************************************************\n");
	}
  }


  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  cudaEventRecord( start, 0 );
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();
   cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &timeDuration, start, stop );
	if(PERFORMANCE_MEASURE == 1){
		printf("\n\n*****************************************************\n");
		printf("Time Taken for render : %f ms\n",timeDuration);
		printf("*****************************************************\n");
	}

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
}

