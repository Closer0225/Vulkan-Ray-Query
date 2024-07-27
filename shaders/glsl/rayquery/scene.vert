#version 460
layout (location = 0) in vec3 inPos;

layout (binding = 0) uniform UBO 
{
	mat4 projection;
	mat4 view;
	mat4 model;
	vec3 lightPos;
} ubo;

layout (location = 0) out vec3 outViewVec;
layout (location = 1) out vec3 outLightVec;
layout (location = 2) out vec3 outWorldPos;

void main() 
{
	gl_PointSize = 1.0;
	gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPos.xyz, 1.0);
    vec4 pos = ubo.model * vec4(inPos, 1.0);
	outWorldPos = vec3(ubo.model * vec4(inPos, 1.0));
    outLightVec = normalize(ubo.lightPos - inPos);
    outViewVec = -pos.xyz;
}

