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

The first feature that I could demonstrate was the rasterization step. I implemented a scanline algorithm based on http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html. This breaks up each triangle in 2, one with a flat bottom and the other with a flat top. Each of these triangles can be rasterized by iterating through each horizontal line, starting with the tip. To simplify the pipeline at this point, I rendered a single triangle with a single color. The result looked like this:

<img src="https://raw.github.com/dkotfis/Project4-Rasterizer/master/renders/raster-triangle.png" "Rasterization Kernel">

Next, I added the vertex shader, primitive assembly, and culling steps so I could render models. My setup keeps all models at the origin of world coordinates, and has a camera that points at the origin. This made it straightforward to later move the camera with mouse controls. The primitive assembly stage loads in a single normal for each triangle from the NBO. This normal would later be used in the fragment shader, so was also passed along through the rasterization stage onto the fragments. I added backface culling through Thrust by performing stream compaction on the primitives by checking the winding order of the vertices in image coordinates. The result of adding these steps in rendering a cow object looked like this:

<img src="https://raw.github.com/dkotfis/Project4-Rasterizer/master/renders/cow_rasterize.png" "Vertex+Primitives">

I then added phong shading to make the models appear 3 dimensional. This adds an ambient, diffuse, and specular component using a light source and a normal vector for each fragment. The result looks like this:

<img src="https://raw.github.com/dkotfis/Project4-Rasterizer/master/renders/cow_phong.png" "Phong Shading">

I next added camera control using the mouse with GLFW so the 3D models can be explored and examined from the application. The mouse control scheme treats the camera's position in spherical coordinates. Clicking and dragging the mouse in the X direction results in incrementally changing the azimuth angle, and the Y direction does the same for the polar angle. To avoid this exploration scheme from breaking down, the polar angle is bound between 0 and PI. Using the scroll wheel on the mouse moves the radial coordinate of the camera incrementally. I added speed factors that were determined experimentally to maximize usability. A video demonstrating camera control can be found in renders/cow_video.wmv.

In cases where a single primitive is very large on screen, the rasterization step for the primitive can take a much longer time to rasterize into fragments. This seemed to be to primary performance bottleneck that I was running into. To solve this, I added a tesselation step after culling that can reduce a triangle into smaller sub-triangles. This can even out the workload for rasterization so that more triangles can be working on drawing the same region of space. I did this by tesselating the triangle by splitting it into 3 subtriangles using a new vertex at the center of the triangle. Additional tessellation is supported by recursing additional layers (implemented as a stack). Thus, a single triangle can be split into 3^N triangles.

##Performance Analysis

Backface Culling - I've found that removing backfacing primitives before rasterization has made very little impact on the rendering speed. For a standard cow object test case, the original 5,804 triangles are culled down by nearly half to 3,014 triangles. However, the performance impact is hardly noticeable, running at 40 FPS independently of whether or not this feature is turned on.

Tesselation - I've compared the run speeds for rendering the cow model at different distances (same viewing angle). For all cases, the speed decreases when getting closer to the model since the rasterization becomes more work for each primitive. 1 level of tesselation increases the number of triangles by 3, which helps reduce the amount of work per triangle at close distances, but hurts at larger distances since more triangles must be executed and each does very little work. Tesselating 2 levels to create 27 times more triangles runs slowly at all distances tested.

<img src="https://raw.github.com/dkotfis/Project4-Rasterizer/master/renders/tesselation_impact.png" "Tesselation Performance">

##Future

- I've notices edge cases where my rasterization algorithm creates bleeding scanlines. This happens when one too many scanlines are generated, due to rounding, and when the triangle slopes are very high.
- I don't currently support loading in and using materials. The color and reflection properties are just hard coded into the fragment shader.
- Writing to the depth buffer within rasterization isn't actually thread-safe. I have minimized the distance between the read and write steps, but am not using any locks or atomic functions. I've found that the cases where this causes an issue are rare enough that it has been a low priority to solve.
- Adaptive level of detail for the tesselation process would be extremely beneficial. Right now, a constant tesselation level applied to all triangles reduces the performance while running with a model far away to boost the performance of the model up close. To get the best of both worlds, it would be better to determine the onscreen area associated with a triangle, and tesselate until the sizes are under an appropriate threshold. This would be complicated to implement, since I would have to do more dynamic GPU memory allocation from within kernels, or allocate extra space up-front and use stream compaction to reduce it afterwards.
- More sophisticated tesselation methods would be valuable. Mine is simple and fast, but it can only tesselate in powers of 3. There are places where more granularity will be necessary to achieve good performance.

