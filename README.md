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


Feature Performance
-------------------

| Feature           | Frame time | Added time | Added time | Notes
|:-------           | ----------:| ----------:| ----------:|:-----
| Nothing           |    4.17 ms |            |            | Base code
| Prim asm          |    4.22 ms |    0.05 ms |      1.20% | Copying data, handling IBO
| Rast+render       |    4.77 ms |    0.55 ms |     13.03% | No locking
| Normal buffer     |    4.84 ms |    0.07 ms |      1.47% | Using normals from mesh
| Basic frag shad   |    5.80 ms |    0.96 ms |     19.83% | Renders model normals
| Backface cull     |    5.76 ms |   -0.04 ms |     -0.69% | 6.13ms using stream compaction to remove backfaces
| Vert/frag structs |    5.78 ms |    0.02 ms |      0.35% | Small performance change
| World-space pos   |    7.21 ms |    1.43 ms |     24.74% | Extra fragment input, extra interpolation of that input
| Depth buf optim   |    7.13 ms |   -0.08 ms |     -1.11% | Remove some unnecessary depth checks
| VS transforms     |    7.77 ms |    0.64 ms |      8.98% | Note that the change in screen size of the model affects the performance
| Lambert shading   |    8.29 ms |    0.52 ms |      6.69% |
| Geometry shader   |    8.82 ms |    0.53 ms |      6.39% | Maximum 4 output tris per input tri
| Tessellation GS   |         ms |         ms |          % | Splits each tri into 3 tris, colors one red
| Backface GS       |         ms |         ms |          % | Moved backface culling to inside the GS
