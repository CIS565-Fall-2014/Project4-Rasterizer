-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------


-------------------------------------------------------------------------------
NOTE:
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Overview
-------------------------------------------------------------------------------
In this project I implementED a simplified CUDA based standard rasterized graphics pipeline, similar to the OpenGL pipeline, including vertex shading, primitive assembly, perspective transformation, rasterization, fragment shading, and writing the resulting fragments to a framebuffer.
![alt tag](https://github.com/XJMa/Project4-Rasterizer/blob/master/screenshots/demo3.gif)

I have implemented the following features:
* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through either a scanline
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* A Lambert lighting/shading scheme in the fragment shader

Additional features:
* Back-face culling
* Correct color interpolation between points on a primitive
* Anti-aliasing
* MOUSE BASED interactive camera 

-------------------------------------------------------------------------------
Basic Pipeline Implementation
-------------------------------------------------------------------------------
My first step is implementing vertax shader, which is the first stage of the pipeline. This step is mainly transforming input vertices into clipping space. Next step is primitive assembly, which collects 3 vertices in the vertex buffer and assign them with triangle primitives. The thired stage I implemented is rasterization, which is a little more complicated than the first two. This step is parallelized by primitives( it can also be parallelized by pixel I guess) and I did the following things:
1. use AABB method get the bound box of triangle
2. set scanline in Y axis and get intersection points for each scan line
3. fill the pixels between each intersection points with primitive color it is the front pixel
After these 3 steps I can get a plain filled scene like this
![alt tag](https://raw.githubusercontent.com/XJMa/Project4-Rasterizer/master/screenshots/paintfill.jpg)

Then I implemented a Lambert shading model in fregmant shader
diffuse model:

![alt tag](https://raw.githubusercontent.com/XJMa/Project4-Rasterizer/master/screenshots/diffuss-light.jpg)

specular model:

![alt tag](https://raw.githubusercontent.com/XJMa/Project4-Rasterizer/master/screenshots/spec.jpg)

Normal debug scene:

![alt tag](https://raw.githubusercontent.com/XJMa/Project4-Rasterizer/master/screenshots/normal.jpg)

-------------------------------------------------------------------------------
Color Interpolation
-------------------------------------------------------------------------------
To achieve proper color interpolation, I converted each pixel coordinate back to barycentric coordinates relative to the triangle primitive's three vertices. Then I can get properly interpolated color gradients within each face by following barycentric interpolation model

![alt tag](https://raw.githubusercontent.com/XJMa/Project4-Rasterizer/master/screenshots/color%20interpolation.jpg)

-------------------------------------------------------------------------------
Mouse Based Interactive Camera
-------------------------------------------------------------------------------
Left Button: Camera rotate

Middle Button: Camera Pan

Right Button: Zoom

Interactive camera demo: https://www.youtube.com/watch?v=Q0boU6VKco4

-------------------------------------------------------------------------------
Anti-aliasing
-------------------------------------------------------------------------------
Anti-aliasing is achieved by rasterizing in another depthbuffer with doubled width and height, and convert the big buffer into normal size with simple interpolation, the result is pretty good:

without antianliazing:

![alt tag](https://raw.githubusercontent.com/XJMa/Project4-Rasterizer/master/screenshots/anti_no.jpg)

with antianliazing:

![alt tag](https://raw.githubusercontent.com/XJMa/Project4-Rasterizer/master/screenshots/anti.jpg)
-------------------------------------------------------------------------------
Back-face culling
-------------------------------------------------------------------------------
Each face of a sold polyhedron has two sides, a frontface on the outside and a backface on the inside. We can only see frontfaces, and normally, we can only see about half of the front faces. If we take one polygon in isolation, and find that we are looking at it's backface, we can cull that face (remove that face from the list of faces to be rendered).
This article gives clear explaination of the idea: http://glasnost.itcarlow.ie/~powerk/GeneralGraphicsNotes/HSR/backfaceculling.html

To achieve that I determined if the face is visible in primitive assembly stage by checking the angle between eye direction and face normal. And cull out the invisible face using thrust remove(like we did in stream compaction).

-------------------------------------------------------------------------------
Performance Analysis
-------------------------------------------------------------------------------
![alt tag](https://raw.githubusercontent.com/XJMa/Project4-Rasterizer/master/screenshots/performance.jpg)

In my implementation the rasterization is parallerized by permitives, so it is no suprise that the FPS rates drop when we have a bigger mesh(with more faces). And the antialiasing process is computationally expensive too, since it require extra computation in a 4X depthbuffer. 
For the back-face culling, since normally we can only see about half of the front faces, I expected it will speed up the FPS by a factor of 2. But in fact it does not have that obvious influence. When tested with cow the back-face culling did not speed up the rasterization process at all. I think it is because the overhead of thrust::remove_if operation for each primitive(similar to the result of stream compaction in path tracer). For the bunny mesh the culling face process did pretty good. But surprisingly the back-face culling did not speed up the dragon mesh as much as the bunny, given the dragon has more faces the result is quite different from stream compaction. I think this may because the removeif loop is also a hot spot in computaion, since we are not really get rid of the backfaces like we did with the ray in pathtracer. Although we can save time in rasterization, increase mesh faces greatly increase the culling loop time.   


