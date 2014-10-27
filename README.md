-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2014
-------------------------------------------------------------------------------
Jiatong He
-------------------------------------------------------------------------------
Base code by Karl Li
-------------------------------------------------------------------------------

Implemented Features:
---------------------
### Vertex Shader

### Primitive Assembly for triangles

### Backface Culling and Clipping

### Rasterization

#### Color & Normal Interpolation

### Fragment Shading (Blinn-Phong Shader)

-------------------------------------------------------------------------------
REQUIREMENTS:
-------------------------------------------------------------------------------
In this project, you are given code for:

* A library for loading/reading standard Alias/Wavefront .obj format mesh files and converting them to OpenGL style VBOs/IBOs
* A suggested order of kernels with which to implement the graphics pipeline
* Working code for CUDA-GL interop

You will need to implement the following stages of the graphics pipeline and features:

* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through either a scanline or a tiled approach
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* A simple lighting/shading scheme, such as Lambert or Blinn-Phong, implemented in the fragment shader

You are also required to implement at least 3 of the following features:

* Additional pipeline stages. Each one of these stages can count as 1 feature:
   * Geometry shader
   * Transformation feedback
   * Back-face culling
   * Scissor test
   * Stencil test
   * Blending

IMPORTANT: For each of these stages implemented, you must also add a section to your README stating what the expected performance impact of that pipeline stage is, and real performance comparisons between your rasterizer with that stage and without.

* Correct color interpolation between points on a primitive
* Texture mapping WITH texture filtering and perspective correct texture coordinates
* Support for additional primitices. Each one of these can count as HALF of a feature.
   * Lines
   * Line strips
   * Triangle fans
   * Triangle strips
   * Points
* Anti-aliasing
* Order-independent translucency using a k-buffer
* MOUSE BASED interactive camera support. Interactive camera support based only on the keyboard is not acceptable for this feature.

-------------------------------------------------------------------------------
README
-------------------------------------------------------------------------------
All students must replace or augment the contents of this Readme.md in a clear 
manner with the following:

* A brief description of the project and the specific features you implemented.
* At least one screenshot of your project running.
* A 30 second or longer video of your project running.  To create the video you
  can use http://www.microsoft.com/expression/products/Encoder4_Overview.aspx 
* A performance evaluation (described in detail below).

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

