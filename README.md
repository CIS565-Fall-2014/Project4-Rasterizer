-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer [Fall 2014]
-------------------------------------------------------------------------------

video demo: http://youtu.be/kjV3lIKWtjY

![alt tag](https://github.com/radiumyang/Project4-Rasterizer/blob/master/buddha_blinn_0.JPG)
![alt tag](https://github.com/radiumyang/Project4-Rasterizer/blob/master/dragon_blinn_0.JPG)
![alt tag](https://github.com/radiumyang/Project4-Rasterizer/blob/master/cow_blinn_1.JPG)
-------------------------------------------------------------------------------
INTRODUCTION
-------------------------------------------------------------------------------
In this project, I implemented a simplified CUDA based implementation of a standard rasterized graphics pipeline, including vertex shading, primitive assembly, perspective transformation, rasterization, fragment shading, testing and rendering. This project is based on the basecode provided at class.


-------------------------------------------------------------------------------
FEATURES
-------------------------------------------------------------------------------
I implemented following stages of the graphics pipeline and features:

* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through either a scanline or a tiled approach
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* A simple lighting/shading scheme, such as Lambert or Blinn-Phong, implemented in the fragment shader

Extra features:
   * Back-face culling
   * Scissor test
   * Support for additional primitices
	 * Lines
	 * Points
   * Anti-aliasing

-------------------------------------------------------------------------------
RUNNING THE CODE
-------------------------------------------------------------------------------
Right click project -> properties -> debugging -> put the obj file here. (obj files must contain v, vn, f)

Screen interation:
* Press "1": Point Mode (show vertices only)
* Press "2": Wireframe Mode (show edges only)
* Press "3": Fragment Shader Mode (show everything)

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
###PRIMITICES RENDERING MODE
* Blinn-Phong Shader
![alt tag](https://github.com/radiumyang/Project4-Rasterizer/blob/master/buddha_blinn_1.JPG)
* Flat
![alt tag](https://github.com/radiumyang/Project4-Rasterizer/blob/master/dragon_flat_0.JPG)
* LINE
![alt tag](https://github.com/radiumyang/Project4-Rasterizer/blob/master/cow_wire_0.JPG)

* POINT
![alt tag](https://github.com/radiumyang/Project4-Rasterizer/blob/master/buddha_dot_0.JPG)
To implement Point Mode, when rasterizing the triangle, just show the correspondent triangle vertices, no need to check whether one pixel is inside or outside the triangle. Thus, Point Mode has the highest FPS.

FPS Comparison between different modes:

![alt tag](https://github.com/radiumyang/Project4-Rasterizer/blob/master/chart_modes.JPG)

###BACK-FACE CULLING
without backcull:

![alt tag](https://github.com/radiumyang/Project4-Rasterizer/blob/master/back00.JPG)

with backcull:

![alt tag](https://github.com/radiumyang/Project4-Rasterizer/blob/master/back01.JPG)


###ANTI-ALIASING
without anti-aliasing:

![alt tag](https://github.com/radiumyang/Project4-Rasterizer/blob/master/antia_03.JPG)

with anti-aliasing:

![alt tag](https://github.com/radiumyang/Project4-Rasterizer/blob/master/antia_04.JPG)

For pixel(i,j), I use the average color value of points from pixel(i-2, j-2) to pixel(i+2, j+2) as the color of current pixel(i,j). 

###SCISSOR TEST

The Scissor Test is a Per-Sample Processing operation that discards Fragments that fall outside of a certain rectangular portion of the screen.

When rasterizing the triangles, I set the scissor window as bounding box to limit the pixels that we need to process, which can help save efficiency on checking useless out-of-bound points.

![alt tag](https://github.com/radiumyang/Project4-Rasterizer/blob/master/scissor.JPG)




-------------------------------------------------------------------------------
References
-------------------------------------------------------------------------------
* Perspective/ Viewport Transformation: http://www.songho.ca/opengl/gl_transform.html 
* Normal Transformation: http://www.songho.ca/opengl/gl_normaltransform.html 
* Triangle Rasterization: http://fgiesen.wordpress.com/2013/02/08/triangle-rasterization-in-practice/ 
* Blinn-Phong Fragment Shader: http://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model

---
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

