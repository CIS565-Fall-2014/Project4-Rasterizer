-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2014
-------------------------------------------------------------------------------
Jiatong He
-------------------------------------------------------------------------------
Base code by Karl Li

Implemented Features:
---------------------
### Vertex Shader
Implemented a basic vertex shader that takes the vertices, and a model-view-projection matrix, and transforms each vertex into clip space.

### Primitive Assembly for Triangles
Assembles triangles from vertex, color, and normal buffer objects using the index buffer.  Triangles store color, normal, and position for each vertex.  The position is in clip space, but the normal is in world space.

### Backface Culling and Clipping
Simple triangle backface culling using a calculated normal from the vertices (since the stored normal is in world space), and clipping that removes triangles that are outside of the (1, 1) to (-1, -1) box.

### Rasterization
Rasterization implemented as a scanline algorithm.  This section currently takes the most time, and large triangles (in screen space) will slow down the program significantly or even crash it.  For every triangle, we begin by sorting the vertices from top to bottom.  Then, starting from the top, we render it in two steps--top to middle, and middle to bottom (note that either of these may have 0 height, if the top and middle, or middle and bottom are at the same height).

#### Color & Normal Interpolation
I use double linear interpolation to calculate the appropriate depth, color, and normal for each fragment.  I did not use the provided code for barycentric coordinates.  Instead, I LERP first along the edges to find a left fragment and right fragment, then LERP between them to fill in the shape.  I am fairly certain that this method gives the correct color, though it might favor colors horizontally.  I will have to check later.  Normal interpolation comes for free as well, but the OBJ's I am using have uniform normals on each face, so it doesn't change anything.

### Fragment Shading (Blinn-Phong Shader)
Simple fragment shader that takes in a light, fragments, and the inverse of the model-view-projection matrix.  This inverse is multiplied with the position of the fragment in order to get the position in world-space.  The world-space coordinate is then used, along with the normal and light position, to calculate shading using a Blinn-Phong shader.  Objects do not obscure each other yet.  So long as a plane has a normal towards the light, it will be lit.

### Mouse Control
Use the mouse to rotate, pan, and zoom the camera.  LMB rotates, RMB pans, and middle mouse button zooms.  Note that zooming too far in will cause the code to crash due to excessively large triangles.

### Key Control
Pressing 'z' will draw faces.  Pressing 'c' will draw only vertices.  'x' is supposed to draw a wireframe but it's not done yet.
Pressing 'a' will use Blinn-Phong lighting.  's' will color by normals.  'd' will color by depth (not really working).

Missing Features
----------------
### Depth buffer testing
I did not have time to do proper depth checking, so you can see depth errors such as in the following image, where the cow's tail is visible through its body.  Each fragment does have a depth value, it's just a matter of setting up atomics and locking the fragment properly.

Back-Face Culling and Clipping Performance Analysis
--------------------------------------
### Expectations
I don't actually expect that this increases runtime too much.  I don't think that the bottleneck is due to too many triangles. Rather, it's caused by large triangles.  Large triangles cause a single thread to take longer, and mess everything up.  There should be a small performance impact, but not huge.

However, since I did not implement a proper z-buffer, having back-face culling means that there will be less incorrect z-fighting due to race conditions, since there won't be any back face to fight with the front.

### Performance Impact
#####Without Culling and Clipping: 24-25 fps
#####With Culling: 25-26 fps
#####With Culling and Clipping (flank of cow shown): 27-28 fps
#####With Culling and Clipping (head of cow shown): 29-31 fps

So as you can see, it does produce some level of speedup, but not much.  In addition, clipping gives more speedup if I hide the body offscreen and keep the head (which has more triangles, but they are smaller), which leads me back to the point I was making about triangle size causing the bottleneck.

Performance Evaluation--A Better Linear Interpolation?
------------------------------------------------------
The current bottleneck in the code is rasterizationKernel (though the fragment shader and clearDepthBuffer take up considerable time as well).  When a single triangle takes up a significant part of the screen (maybe 10%), the program slows to a crawl and can crash.  This is caused by a single thread trying to process a large amount of fragments.  The image below shows an example of the runtime of my code while zoomed out, and zoomed in.
[image]

As such, I will be addressing the rasterization kernel for improving performance.

As mentioned above, I use linear interpolation to calculate coordinates/interpolate color&normals for my geometry rather than using barycentric coordinates.  However, I am recalculating the interpolation every fragment.  Since it's linear, each step should have a constant change.  What if I replaced the calculations with dNorm, dCol, dPos values, and added those to the current left, right, or center points?  This would add several variables to the kernel, but should require fewer calculations per triangle and speed up the processing time for large triangles.

