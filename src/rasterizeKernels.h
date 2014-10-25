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


#define SHADING_MODE 3  //0-shading based on normal, 1-shade based on depth, 2-diffuse, 3-blinn

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

	float theta;
	float phi;
	float r;

	glm::mat4 M_model, M_view, M_projection;
	//glm::mat4 M_camera;
	cudaMat4 M_mvp;   //model-view-projection matrix
	cudaMat4 M_mv_prime;  // transpose(inverse(modelView))

	cam(){}

	//constructor
	cam(float fovyi, float aspecti, glm::vec3 eyei, glm::vec3 upi, glm::vec3 centeri){
		fovy = fovyi;
		eye = eyei;
		up = glm::normalize(upi);
		center = centeri;
		aspect = aspecti;
		
		theta = 90.0f;
		phi = 0.0f;
		r = glm::length (eye - center);
		calculateMVP();
	}

	//copy assignment operator
	cam& operator= ( const cam& other ){
		fovy = other.fovy;
		eye = other.eye;
		up = glm::normalize(other.up);
		center = other.center;
		aspect = other.aspect;
		r = other.r;
		theta = other.theta;
		phi = other.phi;

		//M_camera = other.M_camera;
		M_mvp = other.M_mvp;
		M_mv_prime = other.M_mv_prime;
		return *this;
	}

	void calculateCamPos(){
		eye.x = r * sin( theta * PI /180.0f ) * sin( phi * PI /180.0f ) + center.x;
		eye.y = r * cos( theta * PI /180.0f ) + center.y;
		eye.z = r * sin( theta * PI /180.0f ) * cos( phi * PI /180.0f ) + center.z;
		calculateMVP();
	}

	void calculateMVP(){

		//establish model matrix
		//M_model = glm::mat4( glm::vec4(1.0f, 0.0f, 0.0f, 0.0f), glm::vec4(0.0f, 1.0f, 0.0f, 0.0f),glm::vec4(0.0f, 0.0f, 1.0f, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
		M_model = utilityCore::buildTransformationMatrix(glm::vec3(0.0f, -0.3f, 0.0f), glm::vec3(0.0f), glm::vec3(0.8f));
		
		//establish view matrix =  inverse of camera transform matrix
		M_view = glm::lookAt(eye,center,up);
		
		//establish projection matrix
		//make things closer appear bigger
		float near = 0.1f;
		float far = 1000.0f;
		M_projection = glm::perspective(fovy,aspect,near,far);
		
		M_mvp =  utilityCore::glmMat4ToCudaMat4( M_projection * M_view * M_model );
		//M_mvp_prime = utilityCore::glmMat4ToCudaMat4(glm::transpose(glm::inverse(M_projection * M_view * M_model )));
		M_mv_prime = utilityCore::glmMat4ToCudaMat4(glm::transpose(glm::inverse(M_view * M_model)));

		//printf("the mvp matrix:\n");
		//utilityCore::printCudaMat4(M_mvp);

	}
};

extern cam mouseCam;   //used for mouse events


#endif //RASTERIZEKERNEL_H
