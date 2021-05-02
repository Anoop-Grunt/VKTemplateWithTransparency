#version 450

layout(set = 1, binding = 0) uniform sampler2D textureSampler;

layout(location = 0) out vec4 outColour;
layout(location = 1) in vec2 fragTex;
layout(location = 1) out vec4 revealage;
layout(location = 0) in vec3 fragCol;

void main()
{
    vec4 texColour = texture(textureSampler, fragTex);
	outColour = texColour;
    revealage = vec4(texColour.a, 0.f, 0.f, 1.f);
}

