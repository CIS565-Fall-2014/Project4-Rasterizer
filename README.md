-------------------------------------------------------------------------------
Software Rasterizer implemented using CUDA
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
This is a CUDA based software implementation of a standard rasterized graphics pipeline, very similar to the OpenGL pipeline.
The following is a quick overview of the structure and features of my rasterizor. Implementation details will be explained later.
###Pipe-line stages:

* Vertex Shading
* Primitive Assembly
* Back-Face Culling
* Scanline rasterization
* Fragment Shading
* Render

###Other features:

*Mouse-based interactive camera.
*Mesh View
*Vertices View
*Color interpolation

###Reulsts
Tyra (200,000 faces)
![]tyra1.jpg

Cow (5,804 faces) - flat shading
![] cowFlat.jpg

Cow (5,804 faces) - phong shading
![] cowPhong.jpg

Donut - phong
![] donutPhong.jpg

Donut - mesh
![] donutMesh.jpg

Armadillo (212,000 faces) - camera space normal
![]armaNormal.jpg

Dragon (100,000 faces) - vertices
![]dragonCloud.jpg

###Implementation Details
* Vertex Shading

gather

* Primitive Assembly
* Back-Face Culling
* Scanline rasterization
* Fragment Shading
* Render

###Other features:

*Mouse-based interactive camera.
*Mesh View
*Vertices View
*Color interpolation
Click and drag to rotate camera, scroll to zoom in and out. 
W,A,S,D to move camera position.

Used GLFW glfwGetMouseButton,glfwGetCursorPos for getting necessary inputs for setting up the camera according to mouse movement.







###Performance Analysis
Graphics Card: NVIDIA GeForce GTX 660






