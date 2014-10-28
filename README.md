CIS 565 project 04 : CUDA software rasterizer
===================

![alt tag](https://raw.githubusercontent.com/drerucha/Project4-Rasterizer/master/renders/cow_flat_colored.jpg)

## INTRODUCTION

This project is an implementation of a simplified, CUDA-based rasterized graphics pipeline, similar to OpenGL's pipeline. I implemented vertex shading, primitive assembly, perspective transformation, rasterization, and fragment shading, and wrote the resulting fragments to a framebuffer for display.

With the exception of the lighting calculations in my fragment shader, no other raycasting/raytracing is used in this project to generate graphics. Additionally, with the exception of drawing pixels to the screen after all stages of my rasterization pipeline have completed execution, OpenGL is not used anywhere else in this project. The purpose of this project is to see how a rasterization pipeline can generate graphics without the use of raycasting or OpenGL.

## BASECODE

The basecode provided to me included an OBJ loader and most of the I/O and bookkeeping code. The basecode also included a few helper functions for things like computing Barycentric coordinates and AABB bounding boxes for triangles. I implemented the core rasterization pipeline.

## VERTEX SHADING

My vertex shader is responsible for transforming vertices from object-space into screen-space. This transformation requires three matrices to complete. First, a model matrix is used to transform vertices from object-space to world-space. Second, a view matrix is used to transform vertices from world-space to camera-space. Finally, a projection matrix is used to transform vertices from camera-space to clip-space.

The wrapper function that calls my vertex shader kernel computes the model, view, and projection matrices, multiplies them together, and then passes the composite matrix on to my vertex shader. (I used glm's built-in lookAt() and perspective() functions to compute my view and projection matrices.) My vertex shader then multiplies every vertex with this passed-in composite matrix to convert the vertex from object-space to clip-space.

Once in clip space, a perspective transformation is performed by dividing the x-, y-, and z-components of the vertex by the w-component (the transformed vertex is homogeneous, and thus has four components instead of three). This division puts the vertex in NDC-space (normalized device coordinate). This NDC vertex has range [-1, 1]. Next, I remap the x- and y-value ranges from [-1, 1] to [0, 1] by adding 1 and dividing by 2. I want x and y in the [0, 1] range to facilitate conversion to pixels (window coordinates). This conversion is performed by multiplying x by the rendered image's width, and y by the rendered image's height.

I keep the world-space vertices around for lighting and depth computations later in the pipeline, so I store my newly computed screen-space vertices to a second vertex buffer object.

## PRIMITIVE ASSEMBLY

My primitive assembly kernel is responsible for creating triangle primitives from the passed-in index, vertex, color, and normal buffers. Each triangle maintains its world-space vertex positions, screen-space vertex positions, vertex colors, vertex normals, and a flag that marks whether or not the triangle will be rendered in the rasterization stage.

Indices in the index buffer are extracted based on the index of the current triangle primitive. These extracted indices are then used to index into the vertex, color, and normal buffers to create the three points that comprise a correct triangle. This pipeline stage is quite simple, and a quick glance at my code will elucidate things far better than a paragraph in a README can, so I encourage you to look at my code.

Inside the primitive assembly stage, I also perform a simple computation to check if the triangle is facing toward or away from the camera. The result of this computation sets the visibility flag in the triangle which is used in the rasterization stage during backface culling. My method for backface culling is described in greater detail below in the section labeled "Baclface culling".

## RASTERIZATION

The rasterization stage is responsible for determining which fragments (a pixel-sized piece of a triangle primitive) are visible to the camera. This process is completed using a per-primitive scanline conversion algorithm.

First, my method checks the visibility flag of the current triangle (set in the primitive assembly stage). If it is false, then no further processing is done for that triangle.

My method then computes an AABB bounding box for the current triangle in screen-space. I iterate through all the pixels inside this bounding box and compute the Barycentric coordinates for the current pixel against the triangle in screen-space. If negative Barycentric coordinates are detected, then the pixel is outside the bounds of the triangle, and the fragment can be discarded.

If the Barycentric coordinates are not negative, then the depth of the current fragment in world-space is compared against the fragment depth already stored in the depth buffer at the current pixel location. Here, it is probably important to mention that a fragment is basically a pixel before it becomes a pixel. So, each fragment stored in the depth buffer maps to one pixel in the eventual output image.

If the current fragment is determined to be closer to the camera than the fragment already stored in the depth buffer, then the current fragment replaces the fragment in the depth buffer. The world-space position, color, and normal for the new fragment are computed by using the Barycentric coordinates computed in screen-space to interpolate over the triangle in world-space.

This method is parallelized per primitive rather than per scanline, so each thread processes all fragments for a single triangle and updates the depth buffer accordingly.

## FRAGMENT SHADING

My pipeline applies a simple diffuse Lambertian shading to each fragment stored in my depth buffer. Each fragment knows its world-space position, color, and normal. Those three pieces of information, along with a light position and light intensity, are all that are needed to compute a diffuse lighting coefficient and perform Lambertian shading.

Below, you can see the results of adding light contribution to the rendered image using Lambertian shading. Without any lighting, the image lacks all depth.

![alt tag](https://raw.githubusercontent.com/drerucha/Project4-Rasterizer/master/renders/cow_flat_with_aa.jpg)

![alt tag](https://raw.githubusercontent.com/drerucha/Project4-Rasterizer/master/renders/cow_diffuse_with_aa.jpg)

## BARYCENTRIC COLOR INTERPOLATION

![alt tag](https://raw.githubusercontent.com/drerucha/Project4-Rasterizer/master/renders/tri_gray_smaller.jpg) ![alt tag](https://raw.githubusercontent.com/drerucha/Project4-Rasterizer/master/renders/tri_colored_smaller.jpg)

## BACKFACE CULLING

Backface culling is a process where triangles facing away from the camera are not rasterized. For closed, 3D objects, culling backfaces can result in "throwing out" as much as 50% of triangles. As a result, for complex scenes, backface culling can result in a significant performance boost.

My backface culling method is very simple. The direction a triangle is facing is determined by the order of that triangle's vertices. If the order of vertices is counter-clockwise, then the triangle is facing toward the camera. If the order of vertices is clockwise, then the triangle is facing away from the camera. I compute the order of vertices by taking the cross product of the two triangle vectors that originate from the first triangle vertex. So, if the triangle has vertices p1, p2, and p3, I compute ( p2 - p1 ) X ( p3 - p1 ), where 'X' is the cross product. Once I have the result of this cross product, I leverage the right-hand rule to determine vertex ordering.

To perform the right-hand rule, using your right hand, with an open hand, line your fingers up with the first vector, and curl your fingers toward the second vector. Note the direction of your thumb. If your thumb is pointing toward you, then the z-component of the vector cross product result will be positive. If your thumb is pointing away from you, then the z-component of the vector cross product result will be negative. After performing the cross product, I check this z-value to determine triangle visibility.

An performance analysis comparing frame rates with and without backface culling is included below in the section named "Performance analysis".

## ANTI-ALIASING AS A POST-PROCESS

I implemented a simple post-process anti-aliasing scheme. After the fragment shader stage (after lighting computations), I detected edge pixels by computing the color distances between neighboring pixels. If the difference in colors between adjacent pixels exceeded a predefined threshold, then that pixel was marked as an edge. Then, for all edge pixels, I applied a simple uniform blurring by averaging the edge pixel with its eight neighbors.

In the first image below, the red pixels indicate edge pixels that are to be blurred. In the next images, you can see a closeup of the blurring results. The first image does not use anti-aliasing. Anti-aliasing has been applied to the second image.

This was an interesting exercise, and it was very simple to implement and understand, but I am not impressed with the results, and suspect there are improved alternative anti-aliasing methods I should explore. An performance analysis comparing frame rates with and without anti-aliasing is included below in the section named "Performance analysis".

![alt tag](https://raw.githubusercontent.com/drerucha/Project4-Rasterizer/master/renders/cow_diffuse_outlined.jpg)

![alt tag](https://raw.githubusercontent.com/drerucha/Project4-Rasterizer/master/renders/cow_diffuse_no_aa_zoomed_02.jpg) ![alt tag](https://raw.githubusercontent.com/drerucha/Project4-Rasterizer/master/renders/cow_diffuse_with_aa_zoomed_02.jpg)

## VIDEO DEMO

[Video demo.](https://vimeo.com/110144028)

## PERFORMANCE ANALYSIS

[Add text.]

cow.obj with backface culling: 42 fps.
cow.obj without backface culling: 38 fps.

cow.obj without anti-aliasing: 42 fps.
cow.obj with anti-aliasing: 36 fps.

[Add graphs.]

## ROOM FOR IMPROVEMENT

Currently, I have a bug in my rasterization stage when computing fragment depths, so occluded geometry is sometimes rendered in front of the geometry occluding it. My immediate next steps for this project include locating and correcting that error. After that, I would like to interpolate normal values across triangle faces to give the illusion of smoothed geometry even in low polygon objects. After that, I think it would be interesting to try a per-scanline parallelization scheme in place of my per-primitive parallelization scheme. There is a lot of room for performance improvements in my rasterization stage that I think are worth exploring.

## SPECIAL THANKS

I want to give a quick shout-out to Patrick Cozzi who led the fall 2014 CIS 565 course at Penn and Harmony Li who was the TA for the same course. Thanks guys!