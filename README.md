-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2014
Bo Zhang

##Overview

This is a simplified CUDA based implementation of a standard rasterized graphics pipeline, similar to the OpenGL pipeline.

Basic features:

* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through a scanline approach
* A depth buffer for storing and depth testing fragments
* Fragment Shading
* Fragment to framebuffer writing
* Blinn-Phong lighting/shading scheme

Extra features:
 * Correct color interpolation between points on a primitive
 * Back-face culling
 * Anti-aliasing
 * MOUSE BASED interactive camera support
 * Lines mode and points mode

##Progress
###1.Basic Features:
#####(1) Vertex Shading,Primitive Assembly with support for triangle VBOs/IBOs,Perspective Transformation
   These three steps converts the vertices' position and normal from world coordinate system to the screen coordinate system. And then we save these new position and normal into primitives. (triangles in this projects)<br />
   
For vertices' position:<br />
 * World to clip: Pclip = (Mprojection)(Mview)(Mmodel)(Pmodel)<br />
 * Perspective division: Pndc = (Pclip).xyz / (Pclip).w<br />
 * Viewport transform: Pwindow = (Mviewport-transform)(Pndc)<br />
 
For vertices' normal:<br />
 * World to window: Nwindow = transpose(inverse(Mview*Mmodel))(Nmodel)<br />
 
#####(2) Rasterization through a scanline approach,A depth buffer for storing and depth testing fragments
I use a per primitive method to rasterize the triangles. For each triangle, I firstly calculate the bounding box of the triangle to get its x and y range. Then, I implement triangle rasterization by barycenteric coordinates. That for each scan point in the bounding box range, we convert the it into barycentric coordinate. Only if the barycentric coordinate is within the range of [0,1], we retain the point. Then, we use the barycentric coordinates to interpolate the depth of triangle vertices. For each interpolated depth, we compare it with the corresponding depth we store in the depth buffer. And we only keep and show the points nearest to the eye position in z axis.<br />

Here is a sketch map which shows how to do Barycentric Interpolation:



-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
The performance evaluation is where you will investigate how to make your CUDA
programs more efficient using the skills you've learned in class. You must have
performed at least one experiment on your code to investigate the positive or
negative effects on performance. 

We encourage you to get creative with your tweaks. Consider places in your code
that could be considered bottlenecks and try to improve them. 

Each student should provide no more than a one page summary of their
optimizations along with tables and or graphs to visually explain any
performance differences.


-------------------------------------------------------------------------------
SELF-GRADING
-------------------------------------------------------------------------------
* On the submission date, email your grade, on a scale of 0 to 100, to Liam, harmoli+cis565@seas.upenn.edu, with a one paragraph explanation.  Be concise and realistic.  Recall that we reserve 30 points as a sanity check to adjust your grade.  Your actual grade will be (0.7 * your grade) + (0.3 * our grade).  We hope to only use this in extreme cases when your grade does not realistically reflect your work - it is either too high or too low.  In most cases, we plan to give you the exact grade you suggest.
* Projects are not weighted evenly, e.g., Project 0 doesn't count as much as the path tracer.  We will determine the weighting at the end of the semester based on the size of each project.

---
SUBMISSION
---
As with the previous project, you should fork this project and work inside of
your fork. Upon completion, commit your finished project back to your fork, and
make a pull request to the master repository.  You should include a README.md
file in the root directory detailing the following

* A brief description of the project and specific features you implemented
* At least one screenshot of your project running.
* A link to a video of your raytracer running.
* Instructions for building and running your project if they differ from the
  base code.
* A performance writeup as detailed above.
* A list of all third-party code used.
* This Readme file edited as described above in the README section.

