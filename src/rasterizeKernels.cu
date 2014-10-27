// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

///////////////////features//////////////////////////////
#define BLINN_PHONG 1
#define BACKFACECULL 0 
#define ANTI_ALIASING 1
#define SCISSOR_TEST 0
//////////////////////////////////////////////////////////

#define LIGHTNUM    3
glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_cbo;
int* device_ibo;
float* device_nbo;
triangle* primitives;

//ra
light* lights;
light* device_lights;

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

//TODO: Implement a vertex shader
__global__ void vertexShadeKernel(float* vbo, int vbosize, float* nbo, glm::mat4 Mmodel, glm::mat4 Mview, glm::mat4 Mprojection, glm::vec2 resolution, float depth_near, float depth_far){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index < vbosize/3){
	  
		  ////// http://www.songho.ca/opengl/gl_transform.html //////

	  glm::vec4 oldP = glm::vec4( vbo[index*3], vbo[index*3+1], vbo[index*3+2], 1.0f);
	  glm::vec4 oldN = glm::vec4( nbo[index*3], nbo[index*3+1], nbo[index*3+2], 0.0f);

	 ///// Pclip = (Mprojection)(Mview)(Mmodel)(Pmodel) /////
	  glm::mat4 mat = Mprojection * Mview * Mmodel;
	  glm::vec4 newP = mat * oldP;	

	  glm::vec3 tempP;

	  ///// perspective division /////
	  tempP.x =  newP.x / newP.w; 
	  tempP.y = newP.y / newP.w; 
	  tempP.z = newP.z / newP.w;	 

	  /////// viewport transform /////
	  tempP.x = resolution.x/2 * (tempP.x+1);
	  tempP.y = resolution.y/2 * (tempP.y+1);
	  tempP.z = (depth_far - depth_near) * 0.5 * tempP.z + (depth_far + depth_near) * 0.5;
	  
	  vbo[index*3] = tempP.x; 
	  vbo[index*3+1] = tempP.y; 
	  vbo[index*3+2] = tempP.z;

	  

	  //// http://www.songho.ca/opengl/gl_normaltransform.html //////
	   //// normal transform ////
	  //glm::vec4 newN = glm::transpose(glm::inverse(Mview)) * oldN;
	  glm::vec4 newN = Mview * Mmodel * oldN;
	  glm::vec3 newNn = glm::normalize(glm::vec3(newN));	  
	  nbo[index*3] = newNn.x; 
	  nbo[index*3+1] = newNn.y;
	  nbo[index*3+2] = newNn.z;  
	//  if(newN.x > 0.001f || newN.y > 0.001f || newN.z > 0.001f)
	//  printf("normal: %f, %f, %f", newNn.x, newNn.y, newNn.z);
  }
}
__global__ void primitiveWorldPosKernel(float* vbo, int vbosize, int* ibo, int ibosize, triangle* primitives){
	    ////////////////save world pos to triangle/////////////////
	 int index = (blockIdx.x * blockDim.x) + threadIdx.x;
     int primitivesCount = ibosize/3;
	 if(index<primitivesCount){
		  int id0 = ibo[index*3];
		  int id1 = ibo[index*3+1];
		  int id2 = ibo[index*3+2];

		primitives[index].pw0 = glm::vec3(vbo[id0 * 3],vbo[id0 * 3 + 1],vbo[id0 * 3 + 2]);
		primitives[index].pw1 = glm::vec3(vbo[id1 * 3],vbo[id1 * 3 + 1],vbo[id1 * 3 + 2]);
		primitives[index].pw2 = glm::vec3(vbo[id2 * 3],vbo[id2 * 3 + 1],vbo[id2 * 3 + 2]);

	 }

}
//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, triangle* primitives, glm::vec3 Veye){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  int id0 = ibo[index*3];
	  int id1 = ibo[index*3+1];
	  int id2 = ibo[index*3+2];

	  primitives[index].p0 = glm::vec3(vbo[id0 * 3],vbo[id0 * 3 + 1],vbo[id0 * 3 + 2]);
	  primitives[index].p1 = glm::vec3(vbo[id1 * 3],vbo[id1 * 3 + 1],vbo[id1 * 3 + 2]);
	  primitives[index].p2 = glm::vec3(vbo[id2 * 3],vbo[id2 * 3 + 1],vbo[id2 * 3 + 2]);

	  primitives[index].c0 = glm::vec3(1,0,0);
	  primitives[index].c1 = glm::vec3(0,1,0);
	  primitives[index].c2 = glm::vec3(0,0,1);
	
	  primitives[index].n0 = glm::vec3(nbo[id0*3], nbo[id0*3+1], nbo[id0*3+2]);
	  primitives[index].n1 = glm::vec3(nbo[id1*3], nbo[id1*3+1], nbo[id1*3+2]);
	  primitives[index].n2 = glm::vec3(nbo[id2*3], nbo[id2*3+1], nbo[id2*3+2]);


	  ////////////////////////////////////
#if BACKFACECULL == 1	 

	//  printf("!!!Back Face Cull Applied!!!");
	  glm::vec3 VP0 = Veye - primitives[index].pw0;
	  glm::vec3 VP1 = Veye - primitives[index].pw1;
	  glm::vec3 VP2 = Veye - primitives[index].pw2;

	  if(glm::dot( VP0, primitives[index].n0)<0  && glm::dot(VP1, primitives[index].n1)<0  && glm::dot(VP2, primitives[index].n2)<0 )
	  {
		  primitives[index].isdead = true;
	  }
	  else
	  {
		  primitives[index].isdead = false;
	  }
#endif
	  
  }
}



//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, float depth_near, float depth_far, int mode){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){	  

#if BACKFACECULL == 1	 
		  if(primitives[index].isdead)  return;
#endif

	  ////// http://fgiesen.wordpress.com/2013/02/08/triangle-rasterization-in-practice/  /////////////
	  triangle myTri = primitives[index];

	  if(mode == 0)  // vertice mode
	   {
		   int id0 = resolution.x - (int)myTri.p0.x - 1 + (resolution.y - (int)myTri.p0.y - 1) * resolution.x;
		   int id1 = resolution.x - (int)myTri.p1.x - 1 + (resolution.y - (int)myTri.p1.y - 1) * resolution.x;
		   int id2 = resolution.x - (int)myTri.p2.x - 1 + (resolution.y - (int)myTri.p2.y - 1) * resolution.x;

	//	   printf("%d  ", id0);
		   depthbuffer[id0].color = myTri.c0;
		   depthbuffer[id1].color = myTri.c1;
		   depthbuffer[id2].color = myTri.c2;

		   return;
		 }
	 
	
	  // Compute triangle bounding box
	  glm::vec3 min(0,0,0);
	  glm::vec3 max(0,0,0);
	  getAABBForTriangle(myTri, min, max);

	  // Clip against screen bounds
	  min.x = (int) (min.x < 0) ? 0 : min.x;
	  max.x = (int) (max.x < resolution.x-1) ? max.x : resolution.x-1;
	  min.y = (int) (min.y < 0) ? 0 : min.y;
	  max.y = (int) (max.y < resolution.y-1) ? max.y : resolution.y-1;

#if SCISSOR_TEST == 1
	  glm::vec4 scissorWindow = glm::vec4(100,100,650,650);
	  min.x = (int) (min.x < scissorWindow[0]) ? scissorWindow[0] : min.x;
	  max.x = (int) (max.x < scissorWindow[2]) ? max.x : scissorWindow[2];
	  min.y = (int) (min.y < 0) ? (scissorWindow[1]) : min.y;
	  max.y = (int) (max.y < scissorWindow[3]) ? max.y : scissorWindow[3];

#endif
	  //used for edge mode
	   float offset = 0.04f;

	  // Rasterize
	  for(int j = min.y; j <= max.y; j++)
	  {
		  ////get intersections
		  //float tmin = 0; float tmax = 0;
		  //glm::vec2 jmin = glm::vec2(min.x, (float)j);
		  //glm::vec2 jmax = glm::vec2(max.x, (float)j);
		  //glm::vec2 pp0 = glm::vec2(myTri.p0);
		  //glm::vec2 pp1 = glm::vec2(myTri.p1);
		  //glm::vec2 pp2 = glm::vec2(myTri.p2);
		  //scanlineIntersection(jmin, jmax, pp0, pp1, pp2, tmin, tmax);
		  for(int i = min.x; i <= max.x; i++)
		  {
			  int depthIndex = ((resolution.y - j -1) * resolution.x) + (resolution.x - i - 1);	
			  glm::vec2 myPixel = glm::vec2(i,j);
			  			
			  glm::vec3 myBC = calculateBarycentricCoordinate(myTri, myPixel);
			 // glm::vec3 myBC_core = calculateBarycentricCoordinate(myTriCore, myPixel);

			  //check if pixel is within the triangle
			  if(isBarycentricCoordInBounds(myBC))
			  {
				   float depth = getZAtCoordinate(myBC, primitives[index]);

				   fragment myFrag;
				   myFrag.position = glm::vec3(myPixel.x, myPixel.y, depth);
				   myFrag.normal = glm::normalize((myTri.n0 + myTri.n1 + myTri.n2)/3.0f);

				  if(mode == 1) //edge mode
				  {					
						if(abs(myBC.x)  < offset || abs(myBC.y)  < offset || abs(myBC.z)  < offset)
						{
							myFrag.color = glm::vec3(0.5f, 1.0f, 1.0f);
							depthbuffer[depthIndex] = myFrag;
						}

				  }else// face mode
				  {						 				  
					  myFrag.color =  glm::vec3(0.1f,0.1f,0.1f);	

					  if (depthbuffer[depthIndex].position.z  < myFrag.position.z && myFrag.position.z < -depth_near && myFrag.position.z > -depth_far)
					  {
							depthbuffer[depthIndex] = myFrag;
					  }
				  }
			 }

		  }//x
	  }//y
  }
}


__device__ glm::vec3 myNormalize(glm::vec3 norm)
{
	if(glm::length(norm)< 0.0001f)
		return norm;
	else
		return glm::normalize(norm);
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, light* lights, glm::vec3 Veye, glm::mat4 Mmodel, glm::mat4 Mprojection, glm::mat4 Mview){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
#if BLINN_PHONG == 1	 
	  ////// Blinn-Phong  http://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model ////////////
	  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	   glm::vec3 myPos = depthbuffer[index].position;
	  ///// change back to world coordination !!! /////
	  glm::vec3 myPosWorld = glm::vec3((glm::inverse(Mprojection) * glm::inverse(Mmodel * Mview) * glm::vec4(myPos,1.0f)));

	  glm::vec3 myNorm = glm::normalize(depthbuffer[index].normal);
	
	 for(int l = 0; l < LIGHTNUM; l++)
	  {
		  glm::vec3 lightDir = myNormalize( lights[l].position - myPosWorld );
		  float dist = glm::length(lightDir) * glm::length(lightDir);
		  
		  float NdotL = glm::dot(myNorm, lightDir);
		  NdotL = (NdotL > 0) ? NdotL : 0;
		  glm::vec3 diffuseColor = NdotL * depthbuffer[index].color/dist;

	      glm::vec3 viewDir= myNormalize( Veye - myPosWorld );
		  glm::vec3 H = myNormalize( lightDir + viewDir);
		  
		  float NdotH = glm::dot( myNorm, H);
		  NdotH = ( NdotH > 0) ? NdotH : 0;
		  float intensity = pow(NdotH, 20);
		  glm::vec3 specularColor = glm::vec3(1,1,1) * intensity/dist;

		  depthbuffer[index].color += (lights[l].emit * lights[l].color * (0.7f * diffuseColor + 0.3f * specularColor))/(float)LIGHTNUM;
	  }
	//		printf("color: %f, %f, %f", depthbuffer[index].color.x, depthbuffer[index].color.y, depthbuffer[index].color.z);
#endif	  

#if BLINN_PHONG == 0
	  for(int l = 0; l < LIGHTNUM; l++)
	  {
		 depthbuffer[index].color += (lights[l].color) / (float)LIGHTNUM;
	  }
#endif
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer, int mode){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  #if ANTI_ALIASING == 1
  if(mode == 2) //only apply to face mode
	{	glm::vec3 colorSum(0,0,0);
		for(int i = -1; i<=1; i++)
		{
			for(int j = -1; j<=1; j++)
			{
				int newx = x+i; int newy = y+j;
				if(newx >= 0 && newx < resolution.x && newy >= 0 && newy < resolution.y)
				{
					int tmp = newx + newy * resolution.x;
					colorSum += depthbuffer[tmp].color;
				}						
			}
		}
		framebuffer[index] = colorSum / 9.0f;
  }
	
#endif

  if(x<=resolution.x && y<=resolution.y){
    framebuffer[index] = depthbuffer[index].color;
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, glm::vec3 Veye,
	                               glm::mat4 Mmodel, glm::mat4 Mview, glm::mat4 Mprojection, float depth_near, float depth_far, int mode){

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
  primitives = NULL;
  cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle));

  //ra
  lights = new light[LIGHTNUM];
  lights[0].position = glm::vec3(-5, 10, 8);
  lights[0].color = glm::vec3(0.9, 0.4, 0.9);
  lights[0].emit = 8.0f;
  lights[1].position = glm::vec3(7, -10, 12);
  lights[1].color = glm::vec3(1, 0.8, 0);
  lights[1].emit = 8.0f;
  lights[2].position = glm::vec3(0, 20, 0);
  lights[2].color = glm::vec3(0.2, 1, 0.3);
  lights[2].emit = 7.0f;

  device_lights = NULL;
  cudaMalloc((void**)&device_lights, LIGHTNUM*sizeof(light));
  cudaMemcpy( device_lights, lights, LIGHTNUM*sizeof(light), cudaMemcpyHostToDevice);

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



  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //save world vertex pos
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveWorldPosKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_ibo, ibosize, primitives);
  cudaDeviceSynchronize();

  //------------------------------
  //vertex shader
  //------------------------------
  primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, Mmodel, Mview, Mprojection, resolution, depth_near, depth_far);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, device_nbo, nbosize, primitives, Veye);

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, depth_near, depth_far, mode);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  if(mode == 2) {//face mode
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, device_lights, Veye, Mmodel, Mprojection, Mview);
  }

  cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer, mode);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

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
}

