-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------
Due Thursday 10/31/2012
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
NOTE:
-------------------------------------------------------------------------------
This project requires an NVIDIA graphics card with CUDA capability! Any card with CUDA compute capability 1.1 or higher will work fine for this project. For a full list of CUDA capable cards and their compute capability, please consult: http://developer.nvidia.com/cuda/cuda-gpus. If you do not have an NVIDIA graphics card in the machine you are working on, feel free to use any machine in the SIG Lab or in Moore100 labs. All machines in the SIG Lab and Moore100 are equipped with CUDA capable NVIDIA graphics cards. If this too proves to be a problem, please contact Patrick or Karl as soon as possible.

-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
In this project, you will implement a simplified CUDA based implementation of a standard rasterized graphics pipeline, similar to the OpenGL pipeline. In this project, you will implement vertex shading, primitive assembly, perspective transformation, rasterization, fragment shading, and write the resulting fragments to a framebuffer. More information about the rasterized graphics pipeline can be found in the class slides and in your notes from CIS560.

The basecode provided includes an OBJ loader and much of the mundane I/O and bookkeeping code. The basecode also includes some functions that you may find useful, described below. The core rasterization pipeline is left for you to implement.

You MAY NOT use ANY raycasting/raytracing AT ALL in this project, EXCEPT in the fragment shader step. One of the purposes of this project is to see how a rasterization pipeline can generate graphics WITHOUT the need for raycasting! Raycasting may only be used in the fragment shader effect for interesting shading results, but is absolutely not allowed in any other stages of the pipeline.

Also, you MAY NOT use OpenGL ANYWHERE in this project, aside from the given OpenGL code for drawing Pixel Buffer Objects to the screen. Use of OpenGL for any pipeline stage instead of your own custom implementation will result in an incomplete project.

Finally, note that while this basecode is meant to serve as a strong starting point for a CUDA rasterizer, you are not required to use this basecode if you wish, and you may also change any part of the basecode specification as you please, so long as the final rendered result is correct.

-------------------------------------------------------------------------------
CONTENTS:
-------------------------------------------------------------------------------
The Project4 root directory contains the following subdirectories:
	
* src/ contains the source code for the project. Both the Windows Visual Studio solution and the OSX makefile reference this folder for all source; the base source code compiles on OSX and Windows without modification.
* objs/ contains example obj test files: cow.obj, cube.obj, tri.obj.
* renders/ contains an example render of the given example cow.obj file with a z-depth fragment shader. 
* PROJ4_WIN/ contains a Windows Visual Studio 2010 project and all dependencies needed for building and running on Windows 7.
* PROJ4_OSX/ contains a OSX makefile, run script, and (hopefully) all dependencies needed for building and running on Mac OSX 10.8. 
* PROJ4_NIX/ contains a makefile tested to work on Ubuntu 12.04

The Windows and OSX versions of the project build and run exactly the same way as in Project0, Project1, and Project2.

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
BASE CODE TOUR:
-------------------------------------------------------------------------------
You will be working primarily in two files: rasterizeKernel.cu, and rasterizerTools.h. Within these files, areas that you need to complete are marked with a TODO comment. Areas that are useful to and serve as hints for optional features are marked with TODO (Optional). Functions that are useful for reference are marked with the comment LOOK.

* rasterizeKernels.cu contains the core rasterization pipeline. 
	* A suggested sequence of kernels exists in this file, but you may choose to alter the order of this sequence or merge entire kernels if you see fit. For example, if you decide that doing has benefits, you can choose to merge the vertex shader and primitive assembly kernels, or merge the perspective transform into another kernel. There is not necessarily a right sequence of kernels (although there are wrong sequences, such as placing fragment shading before vertex shading), and you may choose any sequence you want. Please document in your README what sequence you choose and why.
	* The provided kernels have had their input parameters removed beyond basic inputs such as the framebuffer. You will have to decide what inputs should go into each stage of the pipeline, and what outputs there should be. 

* rasterizeTools.h contains various useful tools, including a number of barycentric coordinate related functions that you may find useful in implementing scanline based rasterization...
	* A few pre-made structs are included for you to use, such as fragment and triangle. A simple rasterizer can be implemented with these structs as is. However, as with any part of the basecode, you may choose to modify, add to, use as-is, or outright ignore them as you see fit.
	* If you do choose to add to the fragment struct, be sure to include in your README a rationale for why. 

You will also want to familiarize yourself with:

* main.cpp, which contains code that transfers VBOs/CBOs/IBOs to the rasterization pipeline. Interactive camera work will also have to be implemented in this file if you choose that feature.
* utilities.h, which serves as a kitchen-sink of useful functions

-------------------------------------------------------------------------------
SOME RESOURCES:
-------------------------------------------------------------------------------
The following resources may be useful for this project:

* High-Performance Software Rasterization on GPUs
	* Paper (HPG 2011): http://www.tml.tkk.fi/~samuli/publications/laine2011hpg_paper.pdf
	* Code: http://code.google.com/p/cudaraster/ Note that looking over this code for reference with regard to the paper is fine, but we most likely will not grant any requests to actually incorporate any of this code into your project.
	* Slides: http://bps11.idav.ucdavis.edu/talks/08-gpuSoftwareRasterLaineAndPantaleoni-BPS2011.pdf
* The Direct3D 10 System (SIGGRAPH 2006) - for those interested in doing geometry shaders and transform feedback.
	* http://133.11.9.3/~takeo/course/2006/media/papers/Direct3D10_siggraph2006.pdf
* Multi-Fragment Eﬀects on the GPU using the k-Buﬀer - for those who want to do a k-buffer
	* http://www.inf.ufrgs.br/~comba/papers/2007/kbuffer_preprint.pdf
* FreePipe: A Programmable, Parallel Rendering Architecture for Efficient Multi-Fragment Effects (I3D 2010)
	* https://sites.google.com/site/hmcen0921/cudarasterizer
* Writing A Software Rasterizer In Javascript:
	* Part 1: http://simonstechblog.blogspot.com/2012/04/software-rasterizer-part-1.html
	* Part 2: http://simonstechblog.blogspot.com/2012/04/software-rasterizer-part-2.html

-------------------------------------------------------------------------------
NOTES ON GLM:
-------------------------------------------------------------------------------
This project uses GLM, the GL Math library, for linear algebra. You need to know two important points on how GLM is used in this project:

* In this project, indices in GLM vectors (such as vec3, vec4), are accessed via swizzling. So, instead of v[0], v.x is used, and instead of v[1], v.y is used, and so on and so forth.
* GLM Matrix operations work fine on NVIDIA Fermi cards and later, but pre-Fermi cards do not play nice with GLM matrices. As such, in this project, GLM matrices are replaced with a custom matrix struct, called a cudaMat4, found in cudaMat4.h. A custom function for multiplying glm::vec4s and cudaMat4s is provided as multiplyMV() in intersections.h.

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

-------------------------------------------------------------------------------
THIRD PARTY CODE POLICY
-------------------------------------------------------------------------------
* Use of any third-party code must be approved by asking on Piazza.  If it is approved, all students are welcome to use it.  Generally, we approve use of third-party code that is not a core part of the project.  For example, for the ray tracer, we would approve using a third-party library for loading models, but would not approve copying and pasting a CUDA function for doing refraction.
* Third-party code must be credited in README.md.
* Using third-party code without its approval, including using another student's code, is an academic integrity violation, and will result in you receiving an F for the semester.

-------------------------------------------------------------------------------
SELF-GRADING
-------------------------------------------------------------------------------
* On the submission date, email your grade, on a scale of 0 to 100, to Liam, liamboone+cis565@gmail.edu, with a one paragraph explanation.  Be concise and realistic.  Recall that we reserve 30 points as a sanity check to adjust your grade.  Your actual grade will be (0.7 * your grade) + (0.3 * our grade).  We hope to only use this in extreme cases when your grade does not realistically reflect your work - it is either too high or too low.  In most cases, we plan to give you the exact grade you suggest.
* Projects are not weighted evenly, e.g., Project 0 doesn't count as much as the path tracer.  We will determine the weighting at the end of the semester based on the size of each project.

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

