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
[![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/1ebcda9d585eceb58bd538cd547539c2d57c92e0/result/video.JPG)](http://youtu.be/Kxfwf9KqjOw)
-------------------------------------------------------------------------------
Extra Features:
-------------------------------------------------------------------------------

* Back-face culling

Back-face culling removes the primitives which are not invisible. The process makes rendering objects quicker and more efficient by reducing the number of polygons. 
To determine whether the face is invisible, just to calculate the dot multiplication of the view direction and the normal of face. If the value is greater than zero, it means it is to be culled. And use string compaction to kill the threads with back-face primitive.

* Correct color interpolation between points on a primitive

For each primitive, only the color of each vertices are given. We can use interpolation to get the color of a certain point inside the primitive. 
The mehtod here is simple. As barycentric coordinate of each point is already calculated and the sum of each barycentric coordinate value is equal to 1. So we can use coordinate as interpolation weight of each vertex color.

Following the the color interpolation result for a triangle primitive.

![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/master/result/color%20interpolate.PNG)

* Anti-aliasing

A simple anti-aliasing method is to sum the color of the neighbouring positions and set the average value as the anti-aliasing result. In the program, I used the original pixel and its 8 surrounding neighbours to perform anti aliasing.

The left image is without anti-aliasing and the right one is after anti aliasing. It makes the edge smoother but also makes the other part of the image a little blurred.

![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/master/result/antialiasing2.PNG)

![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/master/result/antialias.PNG)

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------

* Visual Performance


When using multi threads to raster the objects, there may exist read and write conflict.

Here is a cube without lock. In the image, some back-face color substitute the front-face color.

![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/master/result/orgCube.PNG)

Back-face culling helps solve some conflict ions, but it's not guarantee for it. Because for the face with perpendicular normal to the view direction, it cannot determine whether it is visible or not.

Here is a cube with back-face culling. From the image, we can see that the top and bottom base substitute the front-face color.

![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/master/result/backFace.PNG)

So, the lock or atomic function should be implemented. In my program, I add an attribute isLocked in fragment. But my function is not guaranteed to lock each pixel, so my cube is still not totally correct.

![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/master/result/lockCube.PNG)

* Time Efficiency
The histogram shows the timing and FPS. Back-face culling helps improve the time efficiency a little. And the anti-aliasing function increases timing a lot.

![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/master/result/table.JPG)

![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/master/result/chart.JPG)


-------------------------------------------------------------------------------
Rendering Result
-------------------------------------------------------------------------------

In the following image, color represents the face normal:

![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/master/result/normal.PNG)

Add shading to it:

![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/master/result/shading.PNG)

Add color to each vertices:

![ScreenShot](https://github.com/liying3/Project4-Rasterizer/blob/master/result/withut%20light.PNG)