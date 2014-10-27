CIS 565 Project 4: CUDA Rasterizer
==================================

* Kai Ninomiya (Arch Linux/Windows 8, Intel i5-4670, GTX 750)


Base Code Features
------------------

* A library for loading/reading standard Alias/Wavefront .obj format mesh files
  and converting them to OpenGL style VBOs/IBOs
* A suggested order of kernels with which to implement the graphics pipeline
* Working code for CUDA-GL interop


Features Implemented
--------------------

* Vertex shader
    * Model-view-projection transformation
* Primitive Assembly with support for triangle VBOs/IBOs
* **Backface culling**
* Basic scanline rasterization into a fragment buffer
    * Depth-testing
    * Barycentric **interpolation of vertex data**
        * Color: not visible on uncolored model
        * Normals: visible with suzanne.obj model
        * World-space position: used in lighting calculations
* Fragment shading
    * Lambert diffuse per-fragment lighting
* Fragment to framebuffer writing

(Extras in **bold**.)
