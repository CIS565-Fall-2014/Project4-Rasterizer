#ifndef camera_h
#define camera_h


#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "glm/glm.hpp"
#include "cudaMat4.h"


class cam{
	float fovy;   //vertical field of view
	glm::vec3 eye;    //location of camera
	glm::vec3 up;     //camera up vector
	glm::vec3 view;   
	cudaMat4 model;
	cudaMat4 view;
	cudaMat4 projection;
	cam(float fovyi, glm::vec3 eyei, glm::vec3 upi, glm::vec3 viewi){
		fovy = fovyi;
		eye = eyei;
		up = upi;
		view = viewi;
		//model = new cudaMat4
	}

};



#endif