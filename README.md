#Implement features:
* Mouse control
* Back-face culling
* Anti-aliasing
* Blending
* Correct color interpolation
* Scissor test
* Draw line and point

#Mouse control:
* Dragging mouse with the left mouse button pressed to change the view direction.
* Pressing the "Ctrl" key and dragging mouse with the left mouse button pressed to change the view port position.
* Pressing the "Alt" key and dragging mouse with the left mouse button pressed to change the light direction.
* Scroll mouse middle button to change the view scale.
![ResultImage](mouse control.bmp)

#Back-face culling:
Using back-face culling could effectively reduce the required time when doing the rasterize. 
Since we could skip the process of computing the barycentric coordinate and doing the interpolation.
![ResultImage](back culling illustration.bmp)
![ResultImage](back culling_resize.bmp)

#Scissor test:
* Using "S" to switch On/Off the scissor test. The scissor area is a 200x200 size window in the center of the screen.
In the scissor test screen, the fragment shader process will be skipped. With the scissor test, the FPS should be higher than which without the scissor test. 
The save time depends on the scissor test screen size. The larger area of the scissor test is, the more time could be saved.
![ResultImage](scissortest.bmp)
![ResultImage](scissor test chart_resize.bmp)

#Anti-Aliasing:
* Super sampling antialiasing:
In order to do the anti-aliasing, I use super sampling to pre-rasterize a 1600 *1600 screen size image and then average the color every 4 pixels.
Because of the super sampling, the cost time on this pipe line is 4 times more than which without the anti-aliasing pipe line.
![ResultImage](supersampling.bmp)
![ResultImage](antialiasing compare.bmp)


#Blending:
* Using "A" to switch On/Off the color blending effect.
* Using "Q" and "W" to Increase/Decrease the alpha value when blending.  
I create a white & black grid background to do the alpha blending with my object image. I create a kernel to deal with the color blending pixel by pixel.
![ResultImage](color blending1.bmp)
![ResultImage](color blending2.bmp)
![ResultImage](color blending chart.bmp)

#Correct color
This function is doing the color interpolation between the vertices. The required time of doing this process is almost nothing because we already have the barycentric coordinate when we doing the depth interpolation. 
What we need to do here is just using the barycentric coordinate as the weight value to multiply with the colors of each vertex.
![ResultImage](correct color.bmp)
![ResultImage](correct color2.bmp)

#Draw line:
* Using "D" to switch the different display mode. 
* Solid
![ResultImage](solidline1.bmp)
* Real line
![ResultImage](realline.bmp)
* Point
![ResultImage](point.bmp)
![ResultImage](drawline chart.bmp)

#Performance:
![ResultImage](tilesize1.bmp)
![ResultImage](tilesize1_chart.bmp)
![ResultImage](tilesize2.bmp)
![ResultImage](tilesize2_chart.bmp)

http://youtu.be/22JkxHzivGE

