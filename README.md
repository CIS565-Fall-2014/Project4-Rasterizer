=======================================
CIS565: Project 4: CUDA Rasterizer
======================================
Fall 2014 <br />
Bo Zhang

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

And here is the rasterized bunny.obj (No lights):


#####(3) Fragment Shading, Fragment to framebuffer writing, Blinn-Phong lighting/shading scheme
After rasterization, I add light into the scene and calculate diffuse and specular color based on Blinn-Phong model.(http://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model) <br />
Notice: The vertices' position should be transformed back to world coordinate system.

Here is the result with diffuse color added:

Here is the result with both diffuse color and specular color:



###2.Extra Features:
#####(1)Correct color interpolation between points on a primitive
We can use the barycentric coordinates to interpolate other attributes of the triangle vertices in the same way we interpolate vertices' depth. 

Here are the results with(right) and without(left) color interpolation(Three vertices on each triangle has the color red,blue and green):
<br />The one without color interpolation all has the same color (1/3,1/3,1/3), while the other one is colorful.


Here are the results with(right) and without(left) normal interpolation(The one with normal interpolation is smoother):


#####(2)Back-face culling
Before the scene is rendered, polygons (or parts of polygons) which will not be visible in the final scene will be removed.Back-face culling is that we remove backfaces (faces turned away from the camera). As the triangles in obj files usually are defined with counter-clockwise winding when viewed from the outside, we can use the signed triangle area method to judge which faces are backfaces. If we measure the area of such a triangle and find it to be <0, then we know we are looking at a backface which does not have to be drawn.

Here are the details about how to compute signed triangle area:

Here are the results with(right) and without(left) Back-face culling(The left one is without Back-face culling and we can see some artifacts, and the right one does not have these artifacts):


#####(3)Anti-aliasing
I use a simple anti-aliasing method that each pixel's color is averaged from the sum of its color and its eight neighbor pixels' color. This make the result smoother and here are the results with(right) and without(left) anti-aliasing:


 * MOUSE BASED interactive camera support
 I use   glfwSetMouseButtonCallback,  glfwSetCursorEnterCallback and glfwSetCursorPosCallback to implement MOUSE BASED interactive camera:
Use left button to rotate, middle button to pan and right button to zoom in(move top right) or out(mvoe bottom left). See more details in personal video.
 
 * Lines mode and points mode
I change the rasterization function to support line mode and point mode to show the edges or points of the obj.<br />

Lines mode:

Points mode:





-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
The performance evaluation is where you will investigate how to make your CUDA
programs more efficient using the skills you've learned in class. You must have
performed at least one experiment on your code to investigate the positive or
negative effects on performance. 

We encourage you to get creative with your tweaks. Consider places in your code
that could be considered bottlenecks and try to improve them. 

Each student should provide no more than a one page summary of their
optimizations along with tables and or graphs to visually explain any
performance differences.


SUBMISSION
---
As with the previous project, you should fork this project and work inside of
your fork. Upon completion, commit your finished project back to your fork, and
make a pull request to the master repository.  You should include a README.md
file in the root directory detailing the following

* A brief description of the project and specific features you implemented
* At least one screenshot of your project running.
* A link to a video of your raytracer running.
* Instructions for building and running your project if they differ from the
  base code.
* A performance writeup as detailed above.
* A list of all third-party code used.
* This Readme file edited as described above in the README section.
