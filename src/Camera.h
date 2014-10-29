#pragma once
#include "glm\glm.hpp"
#include "utilities.h"

class Camera
{
public:
	float fovy = 60.0f;
	float zNear = 0.1f;
	float zFar = 100.0f;
	int width = 800; int height = 800;

	glm::vec3 lightPos;
	glm::vec3 lightColor;
	glm::vec3 cameraPos;
	glm::vec3 viewDirection;
	glm::vec3 lookAtPos;
	glm::vec3 cameraUp;
	glm::mat4 projectionMatrix;
	glm::mat4 viewMatrix;

	Camera();
	~Camera();
	void UpdatePosition(float rotationX, float rotationY, float zOffset);
};

