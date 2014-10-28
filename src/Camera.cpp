#include "Camera.h"


Camera::Camera()
{
	float fovy = 60.0f;
	float zNear = 0.1f;
	float zFar = 100.0f;
	width = 800;
	height = 600;

	cameraPos = glm::vec3(0,0.2,0.5);
	viewDirection = glm::vec3(0,0,-1);
	lookAtPos = glm::vec3(0, 0, 0);
	projectionMatrix = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
	viewMatrix = glm::lookAt(cameraPos, cameraPos + viewDirection, glm::vec3(0, 1, 0));
}


Camera::~Camera()
{
}
