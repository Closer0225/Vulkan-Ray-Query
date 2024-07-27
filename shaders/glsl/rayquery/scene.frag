#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable
layout (binding = 1, set = 0) uniform accelerationStructureEXT topLevelAS;
layout (location = 0) in vec3 inViewVec;
layout (location = 1) in vec3 inLightVec;
layout (location = 2) in vec3 inWorldPos;

layout (location = 0) out vec4 outFragColor;

#define ambient 0.1

void main() 
{	
	vec3 diffuse =  vec3(1.0, 0.0, 0.0);
	outFragColor = vec4(diffuse, 1.0);
}
