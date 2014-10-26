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
float* device_nbo;//new
int* device_ibo;
triangle* primitives;

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
      f.position.z = 100000.0f; //invalid depth
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
//Gaussian Kernel
__device__ float gaussSample(float inX, float inY, float sigma){
  float xSquared = inX * inX;
  float ySquared = inY * inY;
  float sigmaSquared = sigma * sigma;
  float exponent = (xSquared + ySquared)/sigmaSquared;
  float e = powf(E,-exponent);
  return e / (2 * PI * sigmaSquared);
   
  
}

//TODO: Implement a vertex shader
__global__ void vertexShadeKernel(float* vbo, int vbosize, float* nbo, int nbosize, glm::mat4 View, glm::mat4 modelTransform){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
  //convert from model space into eye/view space
    glm::vec4 vert;
    glm::vec4 norm;
    vert = glm::vec4(vbo[(index * 3) + 0], vbo[(index * 3) + 1], vbo[(index * 3) + 2], 1.0f);//point
    norm = glm::vec4(nbo[(index * 3) + 0], nbo[(index * 3) + 1], nbo[(index * 3) + 2], 0.0f);//vector
    vert = View * modelTransform * vert;
    norm = View * modelTransform * norm;
    __syncthreads(); //This is slow, find a better way to do this
    vbo[(index * 3) + 0] = vert.x;
    vbo[(index * 3) + 1] = vert.y;
    vbo[(index * 3) + 2] = vert.z;
    nbo[(index * 3) + 0] = norm.x;
    nbo[(index * 3) + 1] = norm.y;
    nbo[(index * 3) + 2] = norm.z;
    
    
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, triangle* primitives, glm::mat4 projection){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
    //assemble the primitives!
    //At this point vertices and normals are in eye space.
    //not using ibo, not sure what problem is with ibo but it looks funky
    int vboindex = index * 9;
    int nboindex = vboindex;
    
    
    glm::vec4 p0 = glm::vec4(0,0,0,1.0f);
    glm::vec4 p1 = glm::vec4(0,0,0,1.0f);
    glm::vec4 p2 = glm::vec4(0,0,0,1.0f);
    
    glm::vec3 c0 = glm::vec3(1,0,0);
    glm::vec3 c1 = glm::vec3(0,1,0);
    glm::vec3 c2 = glm::vec3(0,0,1);
    
    glm::vec3 n0 = glm::vec3(1.0f,0   ,0   );
    glm::vec3 n1 = glm::vec3(0   ,1.0f,0   );
    glm::vec3 n2 = glm::vec3(0   ,0   ,1.0f);
    //get points
    p0.x = vbo[vboindex];
    p0.y = vbo[vboindex+ 1];
    p0.z = vbo[vboindex + 2];
    p1.x = vbo[vboindex + 3];
    p1.y = vbo[vboindex + 4];
    p1.z = vbo[vboindex + 5];
    p2.x = vbo[vboindex + 6];
    p2.y = vbo[vboindex + 7];
    p2.z = vbo[vboindex + 8];
    
    
    
    //get colors (nothing here yet)
    
    //get normals
    n0.x = nbo[nboindex];
    n0.y = nbo[nboindex + 1];
    n0.z = nbo[nboindex + 2];
    n1.x = nbo[nboindex + 3];
    n1.y = nbo[nboindex + 4];
    n1.z = nbo[nboindex + 5];
    n2.x = nbo[nboindex + 6];
    n2.y = nbo[nboindex + 7];
    n2.z = nbo[nboindex + 8];
    //leave them in clip space for shading
    
    ////////////////////
    //Back Face Culling!
    ////////////////////
    bool back0;
    bool back1;
    bool back2;
    back0 = glm::dot(n0,-glm::vec3(p0.x,p0.y,p0.z)) < 0;
    back1 = glm::dot(n1,-glm::vec3(p1.x,p1.y,p1.z)) < 0;
    back2 = glm::dot(n2,-glm::vec3(p2.x,p2.y,p2.z)) < 0;
    if(back0 && back1 && back2){
      primitives[index].culled = true;
      return;//no need to do more work
    }
    
    //transform points into clip space
    p0 = projection * p0;
    p1 = projection * p1;
    p2 = projection * p2;
    
    glm::vec3 transP0 = glm::vec3(p0.x/p0.w, p0.y/p0.w, p0.z/p0.w); 
    glm::vec3 transP1 = glm::vec3(p1.x/p1.w, p1.y/p1.w, p1.z/p1.w); 
    glm::vec3 transP2 = glm::vec3(p2.x/p2.w, p2.y/p2.w, p2.z/p2.w); 
    
    
    //place triangles
    primitives[index].p0 = transP0;
    primitives[index].p1 = transP1;
    primitives[index].p2 = transP2;
    primitives[index].c0 = c0;
    primitives[index].c1 = c1;
    primitives[index].c2 = c2;
    primitives[index].n0 = n0;
    primitives[index].n1 = n1;
    primitives[index].n2 = n2;
    primitives[index].culled = false;

  }
}


//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){ //, glm::vec3* devbbox){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
    //convert to screen coordinates
    triangle tri = primitives[index];
    if(tri.culled == true){//did I cull this triangle?
      return;
    }
    clipToScreen(tri, resolution);
    //get bounding box for triangle
    glm::vec3 bbMin;
    glm::vec3 bbMax;
    getAABBForTriangle(tri, bbMin, bbMax);
    
    /* if (index == 0){
      devbbox[0] = tri.p0;
      devbbox[1] = tri.p1;
    } */
    
    int minX, minY, maxX, maxY;
    minX = max(floor(bbMin.x), 0.0f);
    minY = max(floor(bbMin.y), 0.0f);
    maxX = min(ceil( bbMax.x), resolution.x);
    maxY = min(ceil( bbMax.y), resolution.y);
    
    glm::vec2 coords;
    glm::vec3 baryCords;
    //iterate through the box to see in or out
    for  (int i = minX; i < maxX; i++){
      for(int j = minY; j < maxY; j++){
        coords.x = float(i);
        coords.y = float(j);
        baryCords = calculateBarycentricCoordinate(tri, coords);
        if(isBarycentricCoordInBounds(baryCords)){
          fragment frag;
          frag.position.x = coords.x;
          frag.position.y = coords.y;
          frag.position.z = tri.p0.z * baryCords.x + tri.p1.z * baryCords.y + tri.p2.z * baryCords.z;
          
          int dbIndex = i + (j * resolution.x);
          fragment frag2 = depthbuffer[dbIndex];
          
          //need to figure how to do this atomically
          if(frag.position.z >= 0.0f && frag.position.z <= 1.0f && frag.position.z < frag2.position.z){
            frag.color  = tri.c0 * baryCords.x + tri.c1 * baryCords.y + tri.c2 * baryCords.z;
            frag.normal = tri.n0 * baryCords.x + tri.n1 * baryCords.y + tri.n2 * baryCords.z;
            depthbuffer[dbIndex] = frag;
          }
          
        }
      }
    }
    
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 lightPos){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
    if (depthbuffer[index].position.z == 100000.0f){
      depthbuffer[index].color = glm::vec3((float)x/resolution.x, (float)y/resolution.y, 1.0f);
    }else{
      fragment frag = depthbuffer[index];
      //Lambert
      
      glm::vec3 fragPos  = glm::vec3((frag.position.x / resolution.x) * 2 - 1, (frag.position.y / resolution.y) * 2 - 1, frag.position.z);
      glm::vec3 lightDir = glm::normalize(lightPos - fragPos);
      float diffuse = max(glm::dot(frag.normal,lightDir), 0.0);
      depthbuffer[index].color = frag.color * diffuse;
      
      //Depth Render
      //depthbuffer[index].color = frag.position.z * glm::vec3(1,1,1);
    }
  }
}



//MODIFIED  Downsamples and then prints
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
    float dbX = x * 3 + 3;
    float dbY = y * 3 + 3;
    glm::vec2 dbRes = resolution;
    dbRes.x = resolution.x * 3 + 6;
    dbRes.y = resolution.y * 3 + 6;
    glm::vec3 color = glm::vec3(0,0,0);
    for(int i = int(dbX) - 3; i < int(dbX) + 4; i++){
      for(int j = int(dbY) - 3; j < int(dbY) + 4; j ++){
        if(i == int(dbX) && j == int(dbY)){
          float gauss = gaussSample(EPSILON, EPSILON, 1.0f); //sigma = 1
          color += depthbuffer[i + (j * int(dbRes.x))].color * gauss;
        }else{
          color += depthbuffer[i + (j * int(dbRes.x))].color * gaussSample((float(i) - dbX), (float(j) - dbY), 1.0f); //sigma = 1
        }
      }
    }
    framebuffer[index] = color;
  }
}



// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, camera cam){

  //set up framebuffer
  framebuffer = NULL;
  cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3));
  
  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));
  
  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
  
  //superscale:
  glm::vec2 screenRes = resolution;
  resolution.x = resolution.x * 3 + 6;
  resolution.y = resolution.y * 3 + 6;

  // set up crucial magic
  tileSize = 8;
  threadsPerBlock = dim3(tileSize, tileSize);
  fullBlocksPerGrid = dim3((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

  
  

  //set up depthbuffer
  depthbuffer = NULL;
  cudaMalloc((void**)&depthbuffer, (int)resolution.x*(int)resolution.y*sizeof(fragment));
  

  
  
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

  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  //normal buffer
  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));
  
  //Stuff I'll need in the process/
  // view matrix
  glm::mat4 View = glm::lookAt(cam.position, cam.forward, cam.up);
  
  // projection matrix
  float aspectRatio = (float)resolution.x / (float)resolution.y;
  glm::mat4 Projection = glm::perspective(cam.fovy, aspectRatio, cam.nearClip, cam.farClip);
  
  // model transform matrix?
  glm::vec3 translation  = glm::vec3(0.0f,0.0f,0.0f);
  glm::vec3 rotation     = glm::vec3(0.0f,frame,0.0f);
  glm::vec3 scale        = glm::vec3(1.0f,1.0f,1.0f);
  glm::mat4 modelTransform = utilityCore::buildTransformationMatrix(translation, rotation, scale);
  
  //Light  Make this editable
  glm::vec4 Light = glm::vec4(2,5,0,1);
  Light = View * Light;
  glm::vec3 lightPos = glm::vec3(Light.x, Light.y, Light.z); //in clip space
  
  /*
  std::cout << "vbo length" << vbosize << "\n";
  std::cout << "cbo length" << cbosize << "\n";
  std::cout << "ibo length" << ibosize << "\n";
  for(int i = 0; i < ibosize; i ++){
    std::cout << ibo[i] << " , ";
  }
  std::cout << "\n";
  exit(0);
  */


  //------------------------------
  //vertex shader 
  // convert to clip coordinates
  // displacement mapping
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_nbo, nbosize, View, modelTransform);
  cudaDeviceSynchronize();
  
  //------------------------------
  //primitive assembly
  // assign points & colors to triangle vertices
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, device_nbo, nbosize, primitives, Projection);
  cudaDeviceSynchronize();


  /*
  triangle* primitives2 = new triangle[(ibosize/3)];
  std::cout << "trying to memcopy \n";
  cudaMemcpy(primitives2, primitives, (ibosize/3)*sizeof(triangle), cudaMemcpyDeviceToHost);
  utilityCore::printVec3(primitives2[0].p0);
  utilityCore::printVec3(primitives2[0].p1);
  utilityCore::printVec3(primitives2[0].p2);
  //utilityCore::printVec3(primitives2[0].n0);
  //utilityCore::printVec3(primitives2[0].n1);
  //utilityCore::printVec3(primitives2[0].n2);
  
  triangle tri = primitives2[0];
  triangle newTri = primitives2[0];
  
  newTri.p0.x = ((tri.p0.x + 1) / 2) * resolution.x;
  newTri.p0.y = ((tri.p0.y + 1) / 2) * resolution.y;
  
  newTri.p1.x = ((tri.p1.x + 1) / 2) * resolution.x;
  newTri.p1.y = ((tri.p1.y + 1) / 2) * resolution.y;
  
  newTri.p2.x = ((tri.p2.x + 1) / 2) * resolution.x;
  newTri.p2.y = ((tri.p2.y + 1) / 2) * resolution.y; 
  
  std::cout << "Comverted to screen coords \n";
  std::cout << "Resolution" << resolution.x << " , " << resolution.y << "\n";
  
  utilityCore::printVec3(newTri.p0);
  utilityCore::printVec3(newTri.p1);
  utilityCore::printVec3(newTri.p2);
  
  
  
  //make temp memory for tests
  glm::vec3* boundBox = new glm::vec3[2];
  glm::vec3* dev_bbox;
  cudaMalloc((void**)&dev_bbox, 2*sizeof(glm::vec3));
  */
  
  /////////////////////////
  //clip stage goes here!!!  measure performance gains when addded
  /////////////////////////
  
  //------------------------------
  //rasterization
  //------------------------------
  //std::cout << "Try to Rasterize \n";
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);
  cudaDeviceSynchronize();
  
  
  /*
  cudaMemcpy(boundBox, dev_bbox, 2*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  std::cout << "Bounding Box coords \n";
  utilityCore::printVec3(boundBox[0]);
  utilityCore::printVec3(boundBox[1]);
  */



  //------------------------------
  //fragment shader
  //------------------------------
//std::cout << "Try to shade \n";
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, lightPos);

  cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  //Downscale:
  resolution = screenRes;
  tileSize = 8;
  fullBlocksPerGrid = dim3((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();

  kernelCleanup();

  //exit(0);

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

