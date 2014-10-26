CIS 565 Project4 : CUDA Rasterizer
===================

Fall 2014

Author: Dave Kotfis

##Overview

This is a GPU graphics pipeline implemented using CUDA. The pipeline is made up of the following stages:

- Vertex Shader -> transforms points/normals using the Model-View-Projection matrix.
- Primitive Assembly -> Constructs triangles from vertices and normals.
- Culling -> Removes primitives that face away from the screen, or are entirely out of the field of view.
- Geometry Shader -> Provides tesselation of primitives.
- Rasterization -> Uses a scanline algorithm to turn primitives into fragments.
- Fragment Shader -> Colors fragments using Phong shading.
- Rendering -> Takes the front fragment from the depth buffer and stores in the frame buffer to show onscreen.


##Progress

The first feature that I could demonstrate was the rasterization step. I implemented a scanline algorithm based on http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html. This breaks up each triangle in 2, one with a flat bottom and the other with a flat top. Each of these triangles can be rasterized by iterating through each horizontal line, starting with the tip. To simplify the pipleline at this point, I rendered a single triangle with a single color. The result looked like this:

<img src="https://raw.github.com/dkotfis/Project4-Rasterizer/master/renders/raster-triangle.png" "Rasterization Kernel">


##Performance Analysis



##Future


