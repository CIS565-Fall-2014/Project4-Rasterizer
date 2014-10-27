CIS 565 project 04 : CUDA software rasterizer
===================

## INTRODUCTION

This project is an implementation of a simplified, CUDA-based rasterized graphics pipeline, similar to OpenGL's pipeline. I implemented vertex shading, primitive assembly, perspective transformation, rasterization, and fragment shading, and wrote the resulting fragments to a framebuffer for display.

With the exception of the lighting calculations in my fragment shader, no other raycasting/raytracing is used in this project to generate graphics. The purpose of this project is to see how a rasterization pipeline can generate graphics without the use of raycasting.

Additionally, with the exception of  drawing pixels to the screen after all stages of my rasterization pipeline have completed execution, OpenGL is not used anywhere else in this project. Again, the purpose of this project is to see how a rasterization pipeline generates graphics. The purpose is not to generate graphics using OpenGL.

## BASECODE

The basecode provided to me included an OBJ loader and most of the I/O and bookkeeping code. The basecode also included a few useful helper functions for things like computing Barycentric coordinates and AABB bounding boxes of triangles. The core rasterization pipeline was my responsibility to implement.

## VERTEX SHADING

My vertex shader is responsible for transforming vertices from object-space into screen-space. This transformation requires three matrices to complete. First, a model matrix must be computed that transforms vertices from object-space into world-space. Second, a view matrix must be computed that transforms vertices from world-space into camera-space. Finally, a projection matrix must be computed that transforms vertices from camera-space into clip-space.

The wrapper function that calls my vertex shader kernel computes the model, view, and projection matrices, multiplies them together, and then passes the composite matrix on to my vertex shader. (I used glm's built-in lookAt() and perspective() functions to compute my view and projection matrices.) My vertex shader then multiplies every vertex with this passed-in composite matrix to convert the vertex from object-space to clip-space. Once in clip space, a perspective transformation is performed by dividing the x-, y-, and z-components by the w-component (the transformed vertex is homogeneous). This division puts the vertex in NDC-space (normalized device coordinate). This NDC vertex is of the range [-1, 1]. I then transform the x- and y-value ranges from [-1, 1] to [0, 1] by adding 1 and dividing by 2. I want x and y in the [0, 1] range so they can be easily converted to pixels (window coordinates). This conversion is performed by multiplying x by the rendered image's width, and y by the rendered image's height.

I wanted to keep the model-space vertices around for lighting and depth computations later in the pipeline, so I decided to store my newly computed screen-space vertices to a second vertex buffer object.

## PRIMITIVE ASSEMBLY

My primitive assembly kernel is responsible for creating triangle primitives from the passed-in index, vertex, color, and normal buffers. My triangle objects ended up containing a bit more information than I would have liked, but the information included served my purposes well. Each triangle maintained its object-space vertex positions, screen-space vertex positions, vertex colors, vertex normals, and a flag that marked whether or not the triangle should be rendered in the rasterization stage.

Indices in the index buffer were extracted based on the index of the curerent triangle primitive. These extracted indices were then used to index into the vertex, color, and normal buffers to form the three points that comprise a correct triangle. This pipeline stage is quite simple, and a quick glance at my code will elucidate things far better than a paragraph in a README can. I encourage you to look at my code.

Inside the primitive assembly stage is also where I perform a simple computation to check if the triangle is facing toward or away from the camera. The result of this computation sets the visibility flag in the triangle which is used in the rasterization stage to cull backfaces. My method for backface culling is described in greater detail below in the section labeled "Baclface culling".

## RASTERIZATION

Coming soon.

## FRAGMENT SHADING

Coming soon.

## BARYCENTRIC COLOR INTERPOLATION

Coming soon.

## BACKFACE CULLING

Coming soon.

## ANTI-ALIASING AS A POST-PROCESS

Coming soon.

## VIDEO DEMO

[Video demo.](https://vimeo.com/110144028)

## PERFORMANCE ANALYSIS

Coming soon.

## ROOM FOR IMPROVEMENT

Coming soon.

## SPECIAL THANKS

I want to give a quick shout-out to Patrick Cozzi who led the fall 2014 CIS 565 course at Penn and Harmony Li who was the TA for the same course. Thanks guys!