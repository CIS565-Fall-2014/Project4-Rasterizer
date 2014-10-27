-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2014
-------------------------------------------------------------------------------
Due Monday 10/27/2014 @ 12 PM
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
This project is a simplified CUDA based implementation of a standard rasterized graphics pipeline, similar to the OpenGL pipeline. 

Basic Features Implemented:
* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through either a scanline or a tiled approach
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* A simple lighting/shading scheme, such as Lambert or Blinn-Phong, implemented in the fragment shader

Extra Features:
* Displacement Mapping
* Blending
* Correct color interpolation between points on a primitive
* Texture mapping
* Support for additional primitices. Each one of these can count as HALF of a feature.
   * Points
* MOUSE BASED interactive camera support. Interactive camera support based only on the keyboard is not acceptable for this feature.
	Left button - rotate
	Right button - zoom in/out

Video Link : https://www.youtube.com/watch?v=UqCF5kZ2ZAc


-------------------------------------------------------------------------------
THIRD PARTY SOFTWARE
-------------------------------------------------------------------------------

bitmap_image.hpp - A simple bitmap reader that reads BMP files.

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
According to the round table, we could observe that it took much longer time in the pipeline of rasterization and fragment shader. The reason for that is that we have an atomic function in the rasterization to make sure that we only store nearest primitives into depth buffer. The waiting time for other threads to access same memory costs a lot.

Secondly, as we can see from table 1, the performance of rasterization dramastically decrease if the primitives are closer to the view port or are larger, which is because each primitive checks with more pixels it overlaps. To increase the performance, we could use polygon fill method hierarchically instead of scanline in Rasterization pipeline. 

Scale	        0.2	0.5	1	2	4
Displacement	0.136	0.158	0.136	0.147	0.136
Vertex Shader	0.184	0.183	0.2	0.187	0.188
Primitive Assem	0.659	0.658	0.658	0.659	0.659
Rasterization	0.36	0.475	0.989	3.48	14.179
Fragment Shader	1.213	1.198	1.198	1.2	1.123
Blending	0.165	0.166	0.173	0.238	0.701
Rendering	0.551	0.591	0.519	0.597	0.6026


Finally, I tested rasterizer with two meshes, bunny and dragon. The bunny has 2503 vertices and 4968 faces, and dragon has 50000 Vertices and 1000000 Faces. According to table 2, the rasterization for dragon took longer time on displacement, vertex shader, primitive assembly stage due to higher frequency on global memory access. For rasterization and fragment, the atomic function took the majority latency. As the result there is not a huge different. 

Pipeline	Bunny	Dragon
Displacement	0.038	0.18211
Vertex Shader	0.0484	0.233
Primitive Assem	0.0953	1.116
Rasterization	1.96	2.684
Fragment Shader	1.213	1.08
Blending	0.384	0.178
Rendering	0.613	0.548


