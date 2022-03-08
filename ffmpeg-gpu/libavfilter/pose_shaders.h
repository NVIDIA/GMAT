const char *srcVertexShader_face = R"###(
#version 330 core
layout (location = 0) in vec4 aPos;
out vec2 texPos;

uniform float W, H;
uniform mat4 model, projection;

void main() {
    vec4 projected = projection * model * aPos;
    gl_Position = vec4(projected.x / projected.z / (W/2) - 1.0, projected.y / projected.z / (H/2) - 1.0, projected.z / 1000.0, 1.0f);
    texPos = vec2(aPos.x, aPos.y);
}
)###";

const char *srcFragmentShader_face = R"###(
#version 330 core
in vec2 texPos;
out vec4 FragColor;

uniform sampler2D tex0;

void main() {
    // FragColor = vec4(1.0f, 1.0f, 1.0f, 0.0f);
    FragColor = texture(tex0, texPos);
}
)###";

const char *srcVertexShader_pic = R"###(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexPos;
out vec2 texPos;

void main() {
    gl_Position = vec4(aPos.x, aPos.y, 0.999, 1.0);
    texPos = aTexPos;
}
)###";

const char *srcFragmentShader_pic = R"###(
#version 330 core
in vec2 texPos;
out vec4 FragColor;

uniform sampler2D tex0;

void main() {
    FragColor = texture(tex0, texPos);
}
)###";

const char *srcVertexShader_face_mask = R"###(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNorm;
layout (location = 2) in vec2 aTexPos;
out vec2 texPos;

uniform float W, H;
uniform mat4 model, projection;

void main() {
    vec4 projected = projection * model * vec4(aPos.x, -aPos.y, -(aPos.z + 0.07), 1.0);
    gl_Position = vec4(projected.x / projected.z / (W/2) - 1.0, projected.y / projected.z / (H/2) - 1.0, projected.z / 1000.0, 1.0f);
    texPos = aTexPos;
}
)###";

const char *srcFragmentShader_face_mask = R"###(
#version 330 core
in vec2 texPos;
out vec4 FragColor;

uniform sampler2D tex_diffuse0;

void main() {
    // FragColor = vec4(1.0f, 1.0f, 1.0f, 0.0f);
    FragColor = texture(tex_diffuse0, texPos);
}
)###";

const char *srcVertexShader_pic_mask = R"###(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexPos;
out vec2 texPos;

void main() {
    gl_Position = vec4(aPos.x, aPos.y, 0.999, 1.0);
    texPos = aTexPos;
}
)###";

const char *srcFragmentShader_pic_mask = R"###(
#version 330 core
in vec2 texPos;
out vec4 FragColor;

uniform sampler2D tex0;

void main() {
    FragColor = texture(tex0, texPos);
}
)###";
