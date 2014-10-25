-------------------------------------------------------------------------------
Software Rasterizer implemented using CUDA
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
This is a simplified CUDA based implementation of a standard rasterized graphics pipeline, very similar to the OpenGL pipeline.
Implemented features: vertex shading, primitive assembly, perspective transformation, rasterization, fragment shading, and write the resulting fragments to a framebuffer. 

###Pipe-line stages:

* Vertex Shading
* Primitive Assembly
* Back-Face Culling
* Scanline rasterization
* Fragment Shading
* Render

###Other features:

*MOUSE BASED interactive camera support.

Click and drag to rotate camera, scroll to zoom in and out. 
W,A,S,D to move camera position.

Used GLFW glfwGetMouseButton,glfwGetCursorPos for getting necessary inputs for setting up the camera according to mouse movement.

*Mesh View

*Point View

*Color interpolation

###Performance Analysis
Graphics Card: NVIDIA GeForce GTX 660




