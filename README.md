Software Rasterizer in CUDA
============
Features
--------

Here's a quick down break down of the features:
- Vertex Shading
- Primitive Assembly with support for triangle VBOs/IBOs
- Perspective Transformation
- Rasterization through either a scanline or a tiled approach
- Fragment Shading
- A depth buffer for storing and depth testing fragments
- Fragment to framebuffer writing
- A simple lighting/shading scheme, such as Lambert or Blinn-Phong, implemented in the fragment shader
- Back face culling
- Mouse based interaction
- Color interpolation

Here is a render!
![Base PT Image][base pt image]

I'm going to talk about the performance of the system briefly.  All times are measured with the dragon file, which is 100,000 triangles.

-Base code (io really) 0.007 ms total per frame
-With vertex shader 0.008 ms total per frame 
-With assembly 0.0095 ms total per frame (+18.75%)
-Rasterization step 0.012 ms total per frame (+26%)
-Rasterization step with back face culling 0.011 ms total per frame (-19%)
-Fragment shader 0.013 ms total per frame (+18%)
-Fragment shader w/color interpolation 0.013 ms total per frame (+18%)
-Render step 0.014 ms total per frame (+7%)

So, culling unseen faces save you on render pretty substantially!  And color interpolation didn't affect things in a noticeable way.

![ee][e]
Blue is with back face culling, red is without.

And mouse based scrolling obviously doesn't affect anything. in terms of performance.

http://youtu.be/rjKDOhJbNLc

[base pt image]:http://2.bp.blogspot.com/-5HfoVl3K_CE/VFQRZcV1bNI/AAAAAAAACd8/BAUZy5cwuqY/s1600/render.png
[e]:http://3.bp.blogspot.com/-Hl2XbNgf__w/VFQTiHrPO1I/AAAAAAAACeI/YlHoSi2JyZA/s1600/image.png