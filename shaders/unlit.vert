
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_multiview : require

layout(binding = 0) uniform CameraUbo {
    mat4 camera[2];
};

layout(binding = 1) uniform Animation {
    float anim;
};

layout(push_constant) uniform Model {
    mat4 model;
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = camera[gl_ViewIndex] * model * vec4(inPosition, 1.0);
    vec3 color = normalize(model[0].xyz);
    if (color.z < 0.) {
        color.gr = color.rb;
    }
    fragColor = abs(color);
}

