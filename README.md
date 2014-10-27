![AntiAliasing Level 3](https://raw.githubusercontent.com/RTCassidy1/Project4-Rasterizer/master/renders/AALevel3.png)
-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2014
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
This is a simplified CUDA based implementation of a standard rasterized graphics pipeline, similar to the OpenGL pipeline. This project implements vertex shading, primitive assembly, perspective transformation, rasterization, fragment shading, and writes the resulting fragments to a framebuffer. 


-------------------------------------------------------------------------------
CONTENTS:
-------------------------------------------------------------------------------
The Project4 root directory contains the following subdirectories:
	
* src/ contains the source code for the project. Both the Windows Visual Studio solution and the OSX makefile reference this folder for all source; the base source code compiles on OSX and Windows without modification.
* objs/ contains example obj test files: cow.obj, cube.obj, tri.obj.
* renders/ contains 3 videos of the rasterizer in action.

-------------------------------------------------------------------------------
ADDITIONAL FEATURES:
-------------------------------------------------------------------------------
* Correct color interpolation between points on a primitive
* Back-Face Culling
* Anti-aliasing

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
ADDITIONAL FEATURES TOUR:
-------------------------------------------------------------------------------
* Correct color interpolation between points on a primitive
	* my Triangles have support for per-vertex color.  In the Rasterization kernel I use the Barycentric coordinates of the triangle to apply the correct color value to a fragment based on its distance from the three vertices.
	* As an aside, when first implementing this I had a sign error and implemented "Front-Face culling" While not a desirable feature, it made a funny video that can be found here: https://www.youtube.com/watch?v=q9GIzXXPtGc&feature=youtu.be
* Back-Face Culling
	* I included Back-Face culling in my primitive assembly and not as a separate feature.  I augmented my triangle struct to have a field indicating whether or not it had been culled.  
	* In the Rasterization Kernel if a triangle has been culled the Kernel returns immediately.
* Anti-Aliasing
	* This is the most in depth feature.  In the cudaRasterizeCore() method there is a variable where you can set the Anti-Aliasing level.  If you leave this as 1 the rasterizer will act as normal, however setting this to a value larger than one will supersample the entire rasterization process by that many times in both the x and y direction.  When it comes to the Render Kernel it will downsample the fragments back to the given resolution using a gaussian distribution.

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
The biggest performance hit comes from Anti-Aliasing. Without anti-aliasing my rasterizer was able to render the cow at 60fps which I used as a baseline.  When I set the Antialiasing to 2x that dropped to 10 fps, at 3x it was 4-5fps, and at 5x it was 1-2fps.  
* I had hoped that back-face culling would help with this, but I actually did not get any performance gains.  However I think this is because of my implementation.  When I culled the triangle, it still was submitted to the rasterization Kernel where it failed fast, but because of the warp size, there were probably very few warps that had ONLY back facing triangles.  
	* What I need to do at a future iteration is to use string compaction to remove the culled triangles so they don't reach the rasterization kernel at all.
* There is also room to improve the AntiAliasing as well. Currently I supersampled the entire image, but if I were to only superscale the edges I would produce a lot fewer fragments. 
* Another quick improvement would be in the downsampling algorithm.  Currently I calculate the gaussian weight for every subpixel on every pixel. The weights could be computed ahead of time and passed to the kernels so they just read them instead of computing them on every frame for every subpixel.
* I also could try to move the subpixel fragments color values to shared memory. With the antialiasing, multiple pixels will sample the same fragments, so if I set this up correctly I could probably reduce a lot of calls out to memory.
I have two more videos of the renderer, one with AntiAliasing on 3x: https://www.youtube.com/watch?v=uRSzpbR4ZaQ&feature=youtu.be
and one without AntiAliasing: https://www.youtube.com/watch?v=J8bXx7zOvN0&feature=youtu.be
