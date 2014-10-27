-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2014
-------------------------------------------------------------------------------

video demo: http://youtu.be/kjV3lIKWtjY

![alt tag](https://github.com/radiumyang/Project4-Rasterizer/blob/master/buddha_blinn_0.JPG)
![alt tag](https://github.com/radiumyang/Project4-Rasterizer/blob/master/dragon_blinn_0.JPG)
![alt tag](https://github.com/radiumyang/Project4-Rasterizer/blob/master/cow_blinn_1.JPG)
-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
In this project, I implemented a simplified CUDA based implementation of a standard rasterized graphics pipeline, including vertex shading, primitive assembly, perspective transformation, rasterization, fragment shading, testing and rendering. This project is based on the basecode provided at class.


-------------------------------------------------------------------------------
FEATURES:
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
The performance evaluation is where you will investigate how to make your CUDA
programs more efficient using the skills you've learned in class. You must have
performed at least one experiment on your code to investigate the positive or
negative effects on performance. 

We encourage you to get creative with your tweaks. Consider places in your code
that could be considered bottlenecks and try to improve them. 

Each student should provide no more than a one page summary of their
optimizations along with tables and or graphs to visually explain any
performance differences.

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

