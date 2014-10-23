-------------------------------------------------------------------------------
Software Rasterizer implemented using CUDA
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
This is a simplified CUDA based implementation of a standard rasterized graphics pipeline, very similar to the OpenGL pipeline.
Implemented features: vertex shading, primitive assembly, perspective transformation, rasterization, fragment shading, and write the resulting fragments to a framebuffer. 

Features:

* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through either a scanline or a tiled approach
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* A simple lighting/shading scheme, such as Lambert or Blinn-Phong, implemented in the fragment shader

Extra features:

*MOUSE BASED interactive camera support.

Click and drag to rotate camera, scroll to zoom in and out. 
W,A,S,D to move camera position.

Used GLFW glfwGetMouseButton,glfwGetCursorPos for getting necessary inputs for setting up the camera according to mouse movement.



