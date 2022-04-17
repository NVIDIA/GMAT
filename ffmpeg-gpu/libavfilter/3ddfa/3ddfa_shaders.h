/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

const char *tddfaVertexShader_face = R"###(
#version 330 core
layout (location = 0) in vec3 aPos;
out vec2 texPos;

uniform float W, H;
uniform mat4 projection;

void main() {
    // vec4 p = projection * vec4(aPos, 1.0f);
    // gl_Position = vec4(p.x, -p.y, p.z, 1.0f);
    gl_Position = vec4(aPos.x / W * 2 - 1.0f, -aPos.y / H * 2 + 1.0f, aPos.z / 10000.0f, 1.0f);
    texPos = vec2((gl_Position.x + 1.0f) / 2, (gl_Position.y + 1.0f) / 2);
}
)###";

const char *tddfaFragmentShader_face = R"###(
#version 330 core
in vec2 texPos;
out vec4 FragColor;

uniform sampler2D tex0;

void main() {
    // FragColor = texture(tex0, texPos);
    FragColor = vec4(1.0f, 1.0f, 1.0f, 0.3f);
}
)###";

const char *tddfaVertexShader_pic = R"###(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexPos;
out vec2 texPos;

void main() {
    gl_Position = vec4(aPos.x, aPos.y, 0.99f, 1.0f);
    texPos = aTexPos;
}
)###";

const char *tddfaFragmentShader_pic = R"###(
#version 330 core
in vec2 texPos;
out vec4 FragColor;

uniform sampler2D tex0;

void main() {
    FragColor = texture(tex0, texPos);
}
)###";
