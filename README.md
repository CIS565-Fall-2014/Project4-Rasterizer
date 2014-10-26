-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------


-------------------------------------------------------------------------------
NOTE:
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Overview
-------------------------------------------------------------------------------
In this project I will implement a simplified CUDA based implementation of a standard rasterized graphics pipeline, similar to the OpenGL pipeline, including vertex shading, primitive assembly, perspective transformation, rasterization, fragment shading, and writing the resulting fragments to a framebuffer.
![alt tag](https://github.com/XJMa/Project4-Rasterizer/blob/master/screenshots/diffuss-light.jpg)
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
![alt tag](https://raw.githubusercontent.com/XJMa/Project4-Rasterizer/master/screenshots/light.jpg)
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

To achieve that I determined if the face is visible in primitive assembly stage by checking the angle between eye direction and face normal. And cull out the invisible face using trust move(like we did in stream compaction).

-------------------------------------------------------------------------------
Performance Analysis
-------------------------------------------------------------------------------


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
THIRD PARTY CODE POLICY
-------------------------------------------------------------------------------
* Use of any third-party code must be approved by asking on Piazza.  If it is approved, all students are welcome to use it.  Generally, we approve use of third-party code that is not a core part of the project.  For example, for the ray tracer, we would approve using a third-party library for loading models, but would not approve copying and pasting a CUDA function for doing refraction.
* Third-party code must be credited in README.md.
* Using third-party code without its approval, including using another student's code, is an academic integrity violation, and will result in you receiving an F for the semester.

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

