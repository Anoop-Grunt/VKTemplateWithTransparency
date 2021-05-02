#version 450

layout(set = 1, binding = 0) uniform sampler2D textureSampler;

layout(location = 0) out vec4 accum;
layout(location = 1) in vec2 fragTex;
layout(location = 1) out vec4 revealage;
layout(location = 0) in vec3 fragCol;

void main()
{
    vec4 texColour = texture(textureSampler, fragTex);
    float weight = clamp(pow(min(1.0, texColour.a * 10.0) + 0.01, 3.0) * 1e8 * 
                         pow(1.0 - gl_FragCoord.z * 0.9, 3.0), 1e-2, 3e3);
	accum = vec4(texColour.rgb * texColour.a, texColour.a) * weight;
    revealage = vec4(texColour.a, 0.f, 0.f, 1.f);
}

