=======================================
CIS565: Project 4: CUDA Rasterizer
======================================
Fall 2014 <br />
Bo Zhang

![Alt text](https://github.com/wulinjiansheng/Project4-Rasterizer/blob/master/Pics/Running.bmp)


##Overview

This is a simplified CUDA based implementation of a standard rasterized graphics pipeline, similar to the OpenGL pipeline.

Basic features:

* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through a scanline approach
* A depth buffer for storing and depth testing fragments
* Fragment Shading
* Fragment to framebuffer writing
* Blinn-Phong lighting/shading scheme

Extra features:
 * Correct color interpolation between points on a primitive
 * Back-face culling
 * Anti-aliasing
 * MOUSE BASED interactive camera support
 * Lines mode and points mode

##Progress
###1.Basic Features:
#####(1) Vertex Shading,Primitive Assembly with support for triangle VBOs/IBOs,Perspective Transformation
   These three steps converts the vertices' position and normal from world coordinate system to the screen coordinate system. And then we save these new position and normal into primitives. (triangles in this projects)<br />
   
For vertices' position:<br />
 * World to clip: Pclip = (Mprojection)(Mview)(Mmodel)(Pmodel)<br />
 * Perspective division: Pndc = (Pclip).xyz / (Pclip).w<br />
 * Viewport transform: Pwindow = (Mviewport-transform)(Pndc)<br />
 
For vertices' normal:<br />
 * World to window: Nwindow = transpose(inverse(Mview*Mmodel))(Nmodel)<br />
 
#####(2) Rasterization through a scanline approach,A depth buffer for storing and depth testing fragments
I use a per primitive method to rasterize the triangles. For each triangle, I firstly calculate the bounding box of the triangle to get its x and y range. Then, I implement triangle rasterization by barycenteric coordinates. That for each scan point in the bounding box range, we convert the it into barycentric coordinate. Only if the barycentric coordinate is within the range of [0,1], we retain the point. Then, we use the barycentric coordinates to interpolate the depth of triangle vertices. For each interpolated depth, we compare it with the corresponding depth we store in the depth buffer. And we only keep and show the points nearest to the eye position in z axis.<br />

Here is a sketch map which shows how to do Barycentric Interpolation:
![Alt text](https://github.com/wulinjiansheng/Project4-Rasterizer/blob/master/Pics/Barycentric%20Coords%20and%20Interpolation.png)
<br />
<br />
<br />
And here is the rasterized bunny.obj (No lights):
![Alt text](https://github.com/wulinjiansheng/Project4-Rasterizer/blob/master/Pics/Nolight.bmp)
<br />
<br />

#####(3) Fragment Shading, Fragment to framebuffer writing, Blinn-Phong lighting/shading scheme
After rasterization, I add light into the scene and calculate diffuse and specular color based on Blinn-Phong model.(http://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model) <br />
Notice: The vertices' position should be transformed back to world coordinate system.

Here is the result with diffuse color added:
![Alt text](https://github.com/wulinjiansheng/Project4-Rasterizer/blob/master/Pics/Only%20Diffuse.bmp)
<br />
<br />
Here is the result with both diffuse color and specular color:
![Alt text](https://github.com/wulinjiansheng/Project4-Rasterizer/blob/master/Pics/DIffuse%20and%20specular.bmp)
<br />
<br />

###2.Extra Features:
#####(1)Correct color interpolation between points on a primitive
We can use the barycentric coordinates to interpolate other attributes of the triangle vertices in the same way we interpolate vertices' depth. 

Here are the results with(right) and without(left) color interpolation(Three vertices on each triangle has the color red,blue and green):
<br />The one without color interpolation all has the same color (1/3,1/3,1/3), while the other one is colorful.
![Alt text](https://github.com/wulinjiansheng/Project4-Rasterizer/blob/master/Pics/With%26Without%20CI.bmp)
<br />
<br />


Here are the results with(right) and without(left) normal interpolation(The one with normal interpolation is smoother):
![Alt text](https://github.com/wulinjiansheng/Project4-Rasterizer/blob/master/Pics/With%26Without%20NI.bmp)
<br />
<br />


#####(2)Back-face culling
Before the scene is rendered, polygons (or parts of polygons) which will not be visible in the final scene will be removed.Back-face culling is that we remove backfaces (faces turned away from the camera). As the triangles in obj files usually are defined with counter-clockwise winding when viewed from the outside, we can use the signed triangle area method to judge which faces are backfaces. If we measure the area of such a triangle and find it to be <0, then we know we are looking at a backface which does not have to be drawn. And after deciding which faces to discard, I use the thrust stream compaction to remove the dicarded faces, and send the rest faces to do rasterization.

Here are the details about how to compute signed triangle area:
![Alt text](https://github.com/wulinjiansheng/Project4-Rasterizer/blob/master/Pics/Signed%20Triangle.bmp)
<br />
<br />
Here are the results with(right) and without(left) Back-face culling(The left one is without Back-face culling and we can see some artifacts, and the right one does not have these artifacts):
![Alt text](https://github.com/wulinjiansheng/Project4-Rasterizer/blob/master/Pics/With%26Without%20BC.bmp)
<br />
<br />

#####(3)Anti-aliasing
I use a simple anti-aliasing method that each pixel's color is averaged from the sum of its color and its eight neighbor pixels' color. This make the result smoother and here are the results with(right) and without(left) anti-aliasing:
![Alt text](https://github.com/wulinjiansheng/Project4-Rasterizer/blob/master/Pics/With%26Without%20AA.bmp)
<br />
<br />
 * MOUSE BASED interactive camera support
 I use   glfwSetMouseButtonCallback,  glfwSetCursorEnterCallback and glfwSetCursorPosCallback to implement MOUSE BASED interactive camera:
Use left button to rotate, middle button to pan and right button to zoom in(move top right) or out(mvoe bottom left). See more details in video link.
 
 * Lines mode and points mode
I change the rasterization function to support line mode and point mode to show the edges or points of the obj.<br />

Lines mode:<br />
![Alt text](https://github.com/wulinjiansheng/Project4-Rasterizer/blob/master/Pics/Line%20mode.bmp)
<br />
<br />
Points mode:<br />
![Alt text](https://github.com/wulinjiansheng/Project4-Rasterizer/blob/master/Pics/Point%20mode.bmp)
<br />
<br />




## PERFORMANCE EVALUATION
Here are the results of cow.obj and bunny.obj, mainly records their FPS under different settings:
![Alt text](https://github.com/wulinjiansheng/Project4-Rasterizer/blob/master/Pics/Perform1.bmp)

![Alt text](https://github.com/wulinjiansheng/Project4-Rasterizer/blob/master/Pics/Perform2.bmp)
<br />
<br />
From the plots, we can see that the FPS drops down a bit when I set backface culling. It makes me confused and I think the only possible reason is that the stream compaction's part takes more time than rasterizing these culling faces. And I need to optimize the stream compaction to solve this problem. Besides, we can see the FPS drops greatly when I add anti-aliasing. It makes sense as we need to compute the averaged color for each pixel, which slows down the rendering progress. 


##Video Link
http://youtu.be/WiQ_etQEv6U
