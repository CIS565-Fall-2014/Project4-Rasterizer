// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"   //to used 'lookAt' & 'perspective'
#include "cudaMat4.h"
#include "utilities.h"


#define SHADING_MODE 2  //0-shading based on normal, 1-shade based on depth, 2-diffuse

void kernelCleanup();
void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, 
	int* ibo, int ibosize, float * nbo, int nbosize);

class cam{
public:
	float fovy;   //vertical field of view
	float aspect;   //aspect ratio = reso_x/reso_y
	glm::vec3 eye;    //location of camera
	glm::vec3 up;     //camera up vector
	glm::vec3 center;   //where camera is looking at

	cudaMat4 M_mvp;   //model-view-projection matrix
	cudaMat4 M_mvp_prime;  // transpose(inverse(M_mvp))
	//constructor
	cam(float fovyi, float aspecti, glm::vec3 eyei, glm::vec3 upi, glm::vec3 centeri){
		fovy = fovyi;
		eye = eyei;
		up = glm::normalize(upi);
		center = centeri;
		aspect = aspecti;

		//establish model matrix
		glm::mat4 M_model( glm::vec4(1.0f, 0.0f, 0.0f, 0.0f),
						glm::vec4(0.0f, 1.0f, 0.0f, 0.0f),
						glm::vec4(0.0f, 0.0f, 1.0f, 0.0f),
						glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
		

		//establish view matrix =  inverse of camera transform matrix
		glm::mat4 M_view = glm::lookAt(eye,center,up);
		
		//establish projection matrix
		float near = 1.0f;
		float far = 10000.0f;
		glm::mat4 M_projection = glm::perspective(fovy,aspect,near,far);

		M_mvp =  utilityCore::glmMat4ToCudaMat4( M_projection * M_view * M_model );
		M_mvp_prime = utilityCore::glmMat4ToCudaMat4(glm::transpose(glm::inverse(utilityCore::cudaMat4ToGlmMat4(M_mvp))));
	}

};

//fovy, aspect, eye, up, lookat


#endif //RASTERIZEKERNEL_H
