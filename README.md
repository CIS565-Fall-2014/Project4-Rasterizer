-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2014

Graphics pipeline and features implemented:

* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through either a scanline or a tiled approach
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* A simple lighting/shading scheme, such as Lambert or Blinn-Phong, implemented in the fragment shader

Extra features:

   * Back-face culling
   * Texture mapping WITH texture filtering and perspective correct texture coordinates
   * Support for additional primitices. Each one of these can count as HALF of a feature.
	   * Lines
	   * Points

-------------------------------------------------------------------------------
CONTROL
-------------------------------------------------------------------------------
Press 'left' and 'right' to rotate the camera

Press 'A' to enable and disable line drawing

Press 'S' to enable and disable points drawing

Press 'D' to enable and disable back face culling

-------------------------------------------------------------------------------
RESULTS
-------------------------------------------------------------------------------
![](https://github.com/DiracSea3921/Project4-Rasterizer/blob/master/cow.png)

Texture mapping with diffuse lighting shading

![](https://github.com/DiracSea3921/Project4-Rasterizer/blob/master/cow2.png)

Lines Rasterization

![](https://github.com/DiracSea3921/Project4-Rasterizer/blob/master/cow3.png)

Points Rasterization

![](https://github.com/DiracSea3921/Project4-Rasterizer/blob/master/texture.bmp)
![](https://github.com/DiracSea3921/Project4-Rasterizer/blob/master/cow4.png)

Texture mapping with texture filtering, bilinear interpolation

-------------------------------------------------------------------------------
PERFORMANCE ANALYSIS
-------------------------------------------------------------------------------

![](https://github.com/DiracSea3921/Project4-Rasterizer/blob/master/chart.png)

With back face culling enabled the fps can be improved from 30 to 40. The expected performance impact is reduce half of the rasterization stage. The test result has proven this assumption.

![](https://github.com/DiracSea3921/Project4-Rasterizer/blob/master/chart2.png)

With different camera distances the fps will change dramatically. This is because in the rasterization stage each thread needs to rasterize one triangle. If the distance is too short, the triangle will be very large in the screen space. So each thread will need more time to resterize that triangle.
 


