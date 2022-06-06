#include <iostream>
#include <vector>
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
// #include <assimp/Importer.hpp>
// #include <assimp/scene.h>
// #include <assimp/postprocess.h>
#include "stb_image.h"

GLenum glCheckError_(const char *file, int line)
{
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        std::string error;
        switch (errorCode)
        {
            case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
            case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
            case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
            // case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
            // case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
            case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
        }
        std::cout << error << " | " << file << " (" << line << ")" << std::endl;
    }
    return errorCode;
}
#define glCheckError() glCheckError_(__FILE__, __LINE__)

class Shader {
private:
    unsigned int m_shaderProgram;
public:
    Shader(char const *vertexShaderSource, char const *fragmentShaderSource) {
        unsigned int vertexShader;
        vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
        glCompileShader(vertexShader);
        check_shader_error(vertexShader, "Vertex");
    glCheckError();

        unsigned int fragmentShader;
        fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
        glCompileShader(fragmentShader);
        check_shader_error(fragmentShader, "Fragment");
    glCheckError();

        // link shaders
        m_shaderProgram = glCreateProgram();
        glAttachShader(m_shaderProgram, vertexShader);
        glAttachShader(m_shaderProgram, fragmentShader);
    glCheckError();
        glLinkProgram(m_shaderProgram);
        check_program_error(m_shaderProgram);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }
    void SetUniform(char const *name, float x) {
        glUniform1f(glGetUniformLocation(m_shaderProgram, name), x);
    }
    void SetUniform(char const *name, int x) {
        glUniform1i(glGetUniformLocation(m_shaderProgram, name), x);
    }
    void SetUniform(char const *name, unsigned int x) {
        glUniform1ui(glGetUniformLocation(m_shaderProgram, name), x);
    }
    void SetUniform(char const *name, glm::mat4 x) {
        glUniformMatrix4fv(glGetUniformLocation(m_shaderProgram, name), 1, GL_FALSE, glm::value_ptr(x));
    }

    void Use() {
        glUseProgram(m_shaderProgram);
    }

private:
    static void check_shader_error(unsigned int shader, char const *ShaderType) {
        int success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if(!success) {
            char infoLog[512];
            glGetShaderInfoLog(shader, 512, NULL, infoLog);
            std::cerr << ShaderType << " shader compilation error:\n" << infoLog << std::endl;
        }
    }

    static void check_program_error(unsigned int program) {
        int success;
        glGetProgramiv(program, GL_COMPILE_STATUS, &success);
        if(!success) {
            char infoLog[512];
            glGetProgramInfoLog(program, 512, NULL, infoLog);
            std::cerr << "Shader program building error:\n" << infoLog << std::endl;
        }
    }
};

inline unsigned int LoadTexture(char const *szImagePath, bool bFlip) {
    stbi_set_flip_vertically_on_load(bFlip);
    int width, height, nChannel;
    unsigned char *data = stbi_load(szImagePath, &width, &height, &nChannel, 0);
    if (!data) {
        std::cout << "No texture file " << szImagePath << std::endl;
        return false;
    }
    unsigned int tex = -1;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, nChannel == 4 ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(data);
    return tex;
}

class Camera {
private:
    glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f,  3.0f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f,  0.0f);

    float yaw = -90.0f, pitch = 0.0f;
    float fov = 45.0f;

public:
    void Move(float frontDelta, float rightDelta) {
        cameraPos += frontDelta * cameraFront + rightDelta * glm::normalize(glm::cross(cameraFront, cameraUp));
    }
    void Rotate(float yawDelta, float pitchDelta) {
        yaw += yawDelta;
        pitch += pitchDelta;
        if (pitch > 89.0f) {
            pitch = 89.0f;
        }
        if (pitch < -89.0f) {
            pitch = -89.0f;
        }
    }
    void Zoom(float fovDelta) {
        fov -= (float)fovDelta;
        if (fov < 1.0f) {
            fov = 1.0f;
        }
        if (fov > 75.0f) {
            fov = 75.0f;
        }
    }
    glm::mat4 GetProjection() {
        return glm::perspective(glm::radians(fov), 800.0f / 600.0f, 0.1f, 100.0f);
    }
    glm::mat4 GetView() {
        glm::vec3 direction;
        direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction.y = sin(glm::radians(pitch));
        direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraFront = glm::normalize(direction);
        return glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    }
};

// class Material {
//     struct Texture {
//         unsigned id;
//     };
//     std::vector<Texture> m_vTex;
//     int m_nTex = 0;
//     static void Parse(aiMaterial *mat, aiTextureType type, std::vector<Texture> &vTex) {
//         for (unsigned i = 0; i < mat->GetTextureCount(type); i++) {
//             aiString str;
//             mat->GetTexture(type, i, &str);
//             std::cout << str.C_Str() << std::endl;

//             Texture tex;
//             tex.id = LoadTexture(str.C_Str(), false);
//             vTex.push_back(tex);
//         }
//     }

// public:
//     void SetTextures(Shader &shader) {
//         if (m_vTex.size()) {
//             int iTex = 0;
//             glActiveTexture(GL_TEXTURE0 + iTex);
//             glBindTexture(GL_TEXTURE_2D, m_vTex[0].id);
//             shader.SetUniform("tex_diffuse0", iTex);
//         }
//     }
//     static Material Parse(aiMaterial *mat) {
//         Material m;
//         Parse(mat, aiTextureType_DIFFUSE, m.m_vTex);
//         Parse(mat, aiTextureType_SPECULAR, m.m_vTex);
//         Parse(mat, aiTextureType_NORMALS, m.m_vTex);
//         Parse(mat, aiTextureType_HEIGHT, m.m_vTex);
//         return m;
//     }
// };

// class Mesh {
//     struct Vertex {
//         glm::vec3 pos;
//         glm::vec3 norm;
//         glm::vec2 tex;
//     };

//     std::vector<Vertex> m_vVertex;
//     std::vector<unsigned> m_vIndex;
//     Material *m_pMaterial = nullptr;
//     unsigned m_VAO = 0, m_VBO = 0, m_EBO = 0;

//     void Setup() {
//         glGenVertexArrays(1, &m_VAO);
//         glGenBuffers(1, &m_VBO);
//         glGenBuffers(1, &m_EBO);

//         glBindVertexArray(m_VAO);
//         glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
//         glBufferData(GL_ARRAY_BUFFER, m_vVertex.size() * sizeof(m_vVertex[0]), m_vVertex.data(), GL_STATIC_DRAW);

//         glEnableVertexAttribArray(0);
//         glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos));
//         glEnableVertexAttribArray(1);
//         glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, norm));
//         glEnableVertexAttribArray(2);
//         glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tex));

//         glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
//         glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_vIndex.size() * sizeof(m_vIndex[0]), m_vIndex.data(), GL_STATIC_DRAW);

//         glBindBuffer(GL_ARRAY_BUFFER, 0);
//         glBindVertexArray(0);
//     }

// public:
//     void Draw(Shader &shader) {
//         if (m_pMaterial) {
//             m_pMaterial->SetTextures(shader);
//         }
//         glBindVertexArray(m_VAO);
//         glDrawElements(GL_TRIANGLES, m_vIndex.size(), GL_UNSIGNED_INT, 0);
//         glBindVertexArray(0);
//     }

//     static Mesh Parse(const aiMesh *mesh, std::vector<Material> &vMaterial) {
//         Mesh m;
//         for (unsigned i = 0; i < mesh->mNumVertices; i++) {
//             Vertex vertex;
//             auto const
//                 &v = mesh->mVertices[i],
//                 &n = mesh->mNormals[i];
//             vertex.pos = glm::vec3(v.x, v.y, v.z);
//             vertex.norm = glm::vec3(n.x, n.y, n.z);
//             vertex.tex = glm::vec2(0, 0);
//             if (mesh->mTextureCoords[0]) {
//                 vertex.tex = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
//             }
//             m.m_vVertex.push_back(vertex);
//         }
//         for (unsigned i = 0; i < mesh->mNumFaces; i++) {
//             auto const &f = mesh->mFaces[i];
//             for (unsigned j = 0; j < f.mNumIndices; j++) {
//                 m.m_vIndex.push_back(f.mIndices[j]);
//             }
//         }
//         if (mesh->mMaterialIndex < vMaterial.size()) {
//             m.m_pMaterial = &vMaterial[mesh->mMaterialIndex];
//         } else {
//             std::cout << "Parse error: mesh->mMaterialIndex >= vMaterial.size()" << std::endl;
//         }
//         std::cout << "vVertex.size()=" << m.m_vVertex.size() << ", vIndex.size()=" << m.m_vIndex.size() << std::endl;
//         m.Setup();
//         return m;
//     }
// };

// class Model {
//     std::vector<Mesh> m_vMesh;
//     std::vector<Material> m_vMaterial;

// public:
//     void Draw(Shader &shader) {
//         shader.Use();
//         for (auto &m : m_vMesh) {
//             m.Draw(shader);
//         }
//     }
//     static Model Parse(const aiScene *scene) {
//         Model m;
//         for (unsigned i = 0; i < scene->mNumMaterials; i++) {
//             m.m_vMaterial.push_back(Material::Parse(scene->mMaterials[i]));
//         }
//         for (unsigned i = 0; i < scene->mNumMeshes; i++) {
//             m.m_vMesh.push_back(Mesh::Parse(scene->mMeshes[i], m.m_vMaterial));
//         }
//         std::cout << "vMesh.size()=" << m.m_vMesh.size() << ", vMaterials.size()=" << m.m_vMaterial.size() << std::endl;
//         return m;
//     }
// };
