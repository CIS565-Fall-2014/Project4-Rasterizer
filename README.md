-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2014
-------------------------------------------------------------------------------
Due Monday 10/27/2014 @ 12 PM
-------------------------------------------------------------------------------
![alt tag](https://github.com/jianqiaol/Project4-Rasterizer/blob/master/one_more_light.png)
-------------------------------------------------------------------------------
Project Overview:
-------------------------------------------------------------------------------
In this project, I implemented a standard rasterized graphics pipeline. Following stages of the graphics pipeline are implemented as basic features:
* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through either a scanline or a tiled approach
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* A simple lighting/shading scheme, such as Lambert or Blinn-Phong, implemented in the fragment shader

Following features are implemented as additional features:
* Back-face culling
* Correct color interpolation between points on a primitive
* MOUSE BASED interactive camera support. Interactive camera support based only on the keyboard is not acceptable for this feature.

-------------------------------------------------------------------------------
Progress:
-------------------------------------------------------------------------------
I started this project by first implement the standard vertex shader. I find this blog really helpful for explaining the whole vertex operation and primitive assembly operation, especially for the projection transformation: http://www.songho.ca/opengl/gl_transform.html

After the vertex shader and primitiveAssemblyKernel are done, here is the rasterized cow without light:
![alt tag](https://github.com/jianqiaol/Project4-Rasterizer/blob/master/without_light.png)

By setting the color of the fragment as its normal, I get this debug picture:
![alt tag](https://github.com/jianqiaol/Project4-Rasterizer/blob/master/debug_normal.png)

After I implemented the fragment shader with to add light effect. The diffuse and specular color are calculated based on Blinn-Phong mode:http://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model. The cow now looks like this:
![alt tag](https://github.com/jianqiaol/Project4-Rasterizer/blob/master/with_light.png)

By using the Barycentric interpolation, I get this correct color interpolation between points on a primitive by setting the vertices' color to be red, green and blue in each triangle:
![alt tag](https://github.com/jianqiaol/Project4-Rasterizer/blob/master/color_interpolation.png)

I then added the mouse based interactive camera. The model is rotating slowly by default.You can change the view angle by holding left button. You can change the distance between the camera and the object by holding right button. A demo can be found here:https://www.youtube.com/watch?v=6bU3c5rq9xY&list=UUUFqyNxl-EHZ0yESB2gGJ-w&spfreload=10

At last I implemented the backface culling by calculate the dot product between the eye direction and the face normal. I also used thrust::remove_if as I did in the last project. In theory the performance should improve a lot, since about half primitives should be invisible. However for the cow the FPS doesn't change too much. Before backface culling the FPS is about 15, and after using backface culling the FPS is still around 16.

Future work:
I would like to try Anti-aliasing and also smoothing the normals for adjacent primitives. 
