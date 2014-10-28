-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
All required features are implemented:

	* Vertex Shading
	
	* Primitive Assembly with support for triangle VBOs/IBOs
	
	* Perspective Transformation
	
	* Rasterization through either a scanline or a tiled approach
	
	* Fragment Shading
	
	* A depth buffer for storing and depth testing fragments
	
	* Fragment to framebuffer writing
	
	* A simple lighting/shading scheme, such as Lambert or Blinn-Phong, implemented in the fragment shader

3 additional features are:

	* Back-face culling

	* Correct color interpolation between points on a primitive

	* MOUSE BASED interactive camera support. Interactive camera support based only on the keyboard is not acceptable for this feature.
	
---------------------------------------------------------------------------------
Performance Analysis
---------------------------------------------------------------------------------
All the data are listed in the Performance.xls file. 
At the beginning, I think after the backface culling phases, as the face number is only half to 1/3 of the original model, I think the FPS should at least 20% to 40% higher. 

However, the actually improvement is subtle. I performed a compaction of the premitives which removed ones with reverse normal. It might be that the stream compaction costs larger than rendering the back faces. 

The render time firstly reduced as the number of premitives increases and then the time increases. For the first decreasing, I think it is because a number of thread will simultaneously visit the same triangle and result in the low performance. And for the increasing, it is because that the number of premitives to test is increasing. 

Because of the memory is limited, I cannot test cases with even more premitives. 

![Alt text](https://github.com/chiwsy/Project4-Rasterizer/blob/master/renders/CullingPersentage.png)
Culling Scale

![Alt text](https://github.com/chiwsy/Project4-Rasterizer/blob/master/renders/Performance.png)
Performance

---------------------------------------------------------------------------------
Render Result
---------------------------------------------------------------------------------
![Alt text](https://github.com/chiwsy/Project4-Rasterizer/blob/master/renders/ColorInterpolation.png)
Color interpolation

![Alt text](https://github.com/chiwsy/Project4-Rasterizer/blob/master/renders/NoAA.png)
Moiré pattern

![Alt text](https://github.com/chiwsy/Project4-Rasterizer/blob/master/renders/cow.png)
![Alt text](https://github.com/chiwsy/Project4-Rasterizer/blob/master/renders/cowHighPoly.png)
cow test case