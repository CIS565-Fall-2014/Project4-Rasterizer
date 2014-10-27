-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
In this project, I implemented a simplified CUDA based implementation of a standard rasterized graphics pipeline, similar to the OpenGL pipeline; including vertex shading, primitive assembly, perspective transformation, rasterization, fragment shading,  fragments to a frame buffering. 

-------------------------------------------------------------------------------
Basic Features:
-------------------------------------------------------------------------------
I have implemented the following stages of the graphics pipeline and basic features:

* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through either a scanline or a tiled approach
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* A simple lighting/shading scheme, such as Lambert or Blinn-Phong, implemented in the fragment shader

-------------------------------------------------------------------------------
Demo
-------------------------------------------------------------------------------
[![ScreenShot](https://raw.github.com/GabLeRoux/WebMole/master/ressources/WebMole_Youtube_Video.png)](http://youtu.be/vt5fpE0bzSY)
-------------------------------------------------------------------------------
Extra Features:
-------------------------------------------------------------------------------

* Back-face culling

Back-face culling removes the primitives which are not invisible. The process makes rendering objects quicker and more efficient by reducing the number of polygons. 
To determine wheteher the face is invisible, just to calculate the dot multiplication of the view direction and the normal of face. If the value is greate than zero, it means it is to be culled.

* Correct color interpolation between points on a primitive

For each primitive, only the color of each vertices are given. We can use interpolation to get the color of a certain point inside the primitive. 
The mehtod here is simple. As barycentric coordinate of each point is already calculated and the sum of each barycentric coordinate value is equal to 1. So we can use coordinate as interpolation weight of each vertex color.

Following the the color interpolation result for a triangle primitive.

![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/master/result/color%20interpolate.PNG)

* Anti-aliasing

A simple anti-aliasing method is to sum the color of the neighbouring positions and set the average value as the anti-aliasing result. In the program, I used the original pixel and its 8 surronding neighbours to perform anti aliasing.

The left image is without anti-aliasing and the right one is after anti aliasing. It makes the edge smoother but also makes the other part of the image a little blurred.

![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/master/result/color%20interpolate.PNG)
![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/master/result/antialiasing2.PNG)

![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/master/result/beforeAnti.PNG)
![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/master/result/antialias.PNG)
-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
While using multithreads to rasterize the objects, there may exist read and write conflict. Back-face culling helps solve some conflictions, but it's not guarantee for it.
