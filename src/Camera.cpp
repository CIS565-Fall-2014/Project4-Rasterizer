#include "Camera.h"


Camera::Camera()
{
	float fovy = 60.0f;
	float zNear = 0.1f;
	float zFar = 100.0f;
	width = 800;
	height = 600;
	lightPos = glm::vec3(5, 10, 10);
	lightColor = glm::vec3(1, 1, 1);
	cameraPos = glm::vec3(0,1,1);
	lookAtPos = glm::vec3(0.0f, 0.3f, 0.0f);
	cameraUp = glm::vec3(0, 1, 0);
	viewDirection = glm::normalize(lookAtPos - cameraPos);
	projectionMatrix = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
	viewMatrix = glm::lookAt(cameraPos, lookAtPos,cameraUp);
}


Camera::~Camera()
{
}

void Camera::UpdatePosition(float rotationX, float rotationY)
{
	glm::mat4 Transform = utilityCore::buildTransformationMatrix(glm::vec3(0.0f), glm::vec3(-rotationY, rotationX, 0.0f), glm::vec3(1.0f));
	glm::vec4 cameraPos4(cameraPos, 1.0f);
	glm::vec4 up4(cameraUp, 0.0f);
	glm::vec4 newPos = Transform * cameraPos4;
	glm::vec4 newUp = Transform * cameraPos4;
	cameraPos = glm::vec3(newPos.x, newPos.y, newPos.z);
	cameraUp = glm::vec3(newUp.x, newUp.y, newUp.z);
	viewDirection = glm::normalize(lookAtPos - cameraPos);
	viewMatrix = glm::lookAt(cameraPos, lookAtPos, cameraUp);
}
void Camera::UpdatePosition(float zOffset)
{
	glm::vec3 translate = viewDirection * zOffset;
	glm::mat4 Transform = utilityCore::buildTransformationMatrix(translate, glm::vec3(0), glm::vec3(1.0f));

	glm::vec4 cameraPos4(cameraPos, 1.0f);
	glm::vec4 up4(cameraUp, 0.0f);
	glm::vec4 newPos = Transform * cameraPos4;
	glm::vec4 newUp = Transform * cameraPos4;
	if (glm::length(newPos) <= 0.5) return;

	cameraPos = glm::vec3(newPos.x, newPos.y, newPos.z);
	cameraUp = glm::vec3(newUp.x, newUp.y, newUp.z);
	viewDirection = glm::normalize(lookAtPos - cameraPos);
	viewMatrix = glm::lookAt(cameraPos, lookAtPos, cameraUp);
}