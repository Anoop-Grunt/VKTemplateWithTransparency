#version 450

layout(input_attachment_index = 0, binding = 0) uniform subpassInput inputColour; // Colour output from subpass 1 --> opaque
layout(input_attachment_index = 1, binding = 1) uniform subpassInput inputDepth;  // Depth output from subpass 1
layout(input_attachment_index = 2, binding = 2) uniform subpassInput inputColour2;  // color output from subpass 2 --> accum
layout(input_attachment_index = 3, binding = 3) uniform subpassInput inputColour3;

layout(location = 0) out vec4 colour;



void main()
{
	int xHalf = 1920/2;
	if(gl_FragCoord.x < xHalf)
	{
		
		colour = vec4(subpassLoad(inputColour2).rgb, 1.0f);
	}
	else
	{
		colour = subpassLoad(inputColour).rgba;
		//A guess as to why the weird transparency thing happens--> those black(now white) pixels, are actually red pixles with alpha = 0, and they get aggressively blended (cuz a = 0 means transparent)with the background color of the swapchain image(not the intermediate image, I tested this)
		//When we put a = 1, this blending is avoided, so the originall red is seen
		
	}
	
}