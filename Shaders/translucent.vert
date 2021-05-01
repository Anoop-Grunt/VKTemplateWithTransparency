#version 450 		

layout(location = 0) in vec3 pos; //Both are at binding zero for ins 
layout(location = 1) in vec3 col;
layout(location = 2) in vec2 tex;

layout(set = 0, binding = 0) uniform UboViewProjection{      //The uniform binding zero, not the in binding zero
	mat4 projection;
	mat4 view;
	

}uboViewProjection;


//No longer using the dynamic uniform buffers for transform(actually just the model) data, instead we use push constants, the following  binding is left just for reference 
layout(set = 0,binding = 1) uniform UboModel{      //The uniform binding zero, not the in binding zero
	mat4 model;

}uboModel;

layout(push_constant) uniform PushModel{
	mat4 model;
} pushModel;


layout(location = 0) out vec3 fragCol;
layout(location = 1) out vec2 fragTex;

void main() {
	gl_Position = uboViewProjection.projection * uboViewProjection.view * pushModel.model * vec4(pos, 1.0);
	
	fragCol = col;
	fragTex = tex;
}