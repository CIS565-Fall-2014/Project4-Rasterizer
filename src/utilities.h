//UTILITYCORE- A Utility Library by Yining Karl Li
//This file is part of UTILITYCORE, Coyright (c) 2012 Yining Karl Li

#ifndef Pathtracer_utilities_h
#define Pathtracer_utilities_h

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include "cudaMat4.h"

const float PI                          =3.1415926535897932384626422832795028841971;
const float TWO_PI                      =6.2831853071795864769252867665590057683943;
const float SQRT_OF_ONE_THIRD           =0.5773502691896257645091487805019574556476;
const float E                           =2.7182818284590452353602874713526624977572;
const float EPSILON                     =.000000001;
const float ZERO_ABSORPTION_EPSILON     =0.00001;
const float RAY_BIAS_AMOUNT             =0.0002;

namespace utilityCore {
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str); 
    extern cudaMat4 glmMat4ToCudaMat4(glm::mat4 a);
    extern glm::mat4 cudaMat4ToGlmMat4(cudaMat4 a);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern void printCudaMat4(cudaMat4 m);
    extern std::string convertIntToString(int number);

    //-----------------------------
    //-------GLM Printers----------
    //-----------------------------
    extern void printMat4(glm::mat4);
    extern void printVec4(glm::vec4);
    extern void printVec3(glm::vec3);
}
#endif
