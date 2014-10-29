-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2014
-------------------------------------------------------------------------------

Completed Features:
-------------------------------------------------------------------------------
You will need to implement the following stages of the graphics pipeline and features:
Required:
* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through either a scanline or a tiled approach
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* A simple lighting/shading scheme, such as Lambert or Blinn-Phong, implemented in the fragment shader

Additional:
* Additional pipeline stages.
   * Back-face culling
* Correct color interpolation between points on a primitive
* Support for additional primitices. 
   * Lines
   * Points
* MOUSE BASED interactive camera support. Interactive camera support based only on the keyboard is not acceptable for this feature.

#How to use:

Use #define RENDER_MODE_0 in rasterizeKernels.cu to use different shading mode
0 for blinn-phong, 1 for wire-frame, 2 for interpolated color, 3 for vertices
Hold and drag mouse to rotate the camera. Use mouse scroll to zoom in and out.

#Result:
![Alt text](/img/phong.PNG "Blinn-Phong shading")
Blinn-Phong shading
![Alt text](/img/interp.PNG)
Color interpolation
![Alt text](/img/wire.PNG)
Wire- Frame
![Alt text](/img/vertices.PNG)
Vertices
#Performance:
Firstly I tried to ananysize the relationship between tile size and one frame runtime. However the result shows they are almost constant against tile size. (See the following chart)
![Alt text](/img/tilezise.PNG)

Then I used the mouse to zoom in and out and I found that the larger one triangle covers the screen the slower the program runs. Here is the result.
![Alt text](/img/small.PNG)
![Alt text](/img/medium.PNG)
![Alt text](/img/large.PNG)
![Alt text](/img/huge.PNG)
![Alt text](/img/extra huge.PNG)

