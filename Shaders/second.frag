#version 450

layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput inputColour2;
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inputColour;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput inputDepth;
layout(input_attachment_index = 3, set = 0, binding = 3) uniform subpassInput inputColour3;

layout(location = 0) out vec4 colour;

const float EPSILON = 0.00001f;

// calculate floating point numbers equality accurately
bool isApproximatelyEqual(float a, float b)
{
    return abs(a - b) <= (abs(a) < abs(b) ? abs(b) : abs(a)) * EPSILON;
}

// get the max value between three values
float max3(vec3 v)
{
    return max(max(v.x, v.y), v.z);
}


void main()
{
    
    vec4 opaqueColor = subpassLoad(inputColour);
	float revealage = subpassLoad(inputColour3).r;
	
	
    
	
	vec4 accum_total = subpassLoad(inputColour2);
	vec3 accum_color = accum_total.rgb;
	float accum_weight = accum_total.a;

	
	//supressing overeflow
	if (isinf(max3(abs(accum_color))))
        accum_color = vec3(accum_weight);
	
	if (!isApproximatelyEqual(revealage, 1.0f)){
		colour = vec4((1-revealage)*(accum_color/max(accum_weight, EPSILON)) + (revealage * opaqueColor.rgb), 1.f);
	}
	else
	{
		colour = opaqueColor;
	}

	
    
}
