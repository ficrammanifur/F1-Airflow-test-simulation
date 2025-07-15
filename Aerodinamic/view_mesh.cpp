#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <thread>
#include <cmath>
#include <memory>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Vertex shader untuk partikel
const char* particleVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aVel;
layout(location = 2) in float aAge;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float time;

out vec3 fragColor;
out float alpha;

void main() {
    vec3 pos = aPos + aVel * time * 0.1;
    gl_Position = projection * view * model * vec4(pos, 1.0);
    
    // Warna berdasarkan kecepatan
    float speed = length(aVel);
    fragColor = mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), speed / 100.0);
    alpha = 1.0 - (aAge / 100.0);
    
    gl_PointSize = 3.0;
}
)";

// Fragment shader untuk partikel
const char* particleFragmentShader = R"(
#version 330 core
in vec3 fragColor;
in float alpha;
out vec4 FragColor;

void main() {
    FragColor = vec4(fragColor, alpha);
}
)";

// Vertex shader untuk geometri F1
const char* geometryVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragPos;
out vec3 normal;

void main() {
    fragPos = vec3(model * vec4(aPos, 1.0));
    normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(fragPos, 1.0);
}
)";

// Fragment shader untuk geometri F1
const char* geometryFragmentShader = R"(
#version 330 core
in vec3 fragPos;
in vec3 normal;
out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 objectColor;

void main() {
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    
    // Ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
    
    // Diffuse
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;
    
    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
)";

struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    float age;
    bool active;
    
    Particle() : position(0.0f), velocity(0.0f), age(0.0f), active(false) {}
};

struct F1Geometry {
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    GLuint VAO, VBO, EBO;
    glm::vec3 color;
    
    F1Geometry(glm::vec3 col) : color(col) {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
    }
    
    ~F1Geometry() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
    }
};

class F1AirflowSimulator {
private:
    GLFWwindow* window;
    int windowWidth, windowHeight;
    
    // Shaders
    GLuint particleShaderProgram;
    GLuint geometryShaderProgram;
    
    // Simulation parameters
    float windSpeed = 50.0f;
    float windAngle = 0.0f;
    float airDensity = 1.225f;
    int gridSize = 100;
    
    // Particle system
    std::vector<Particle> particles;
    int maxParticles = 2000;
    GLuint particleVAO, particleVBO;
    
    // Velocity field
    std::vector<std::vector<glm::vec3>> velocityField;
    std::vector<std::vector<float>> pressureField;
    
    // F1 geometry
    std::vector<std::unique_ptr<F1Geometry>> f1Parts;
    
    // Camera
    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 5.0f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    float cameraYaw = -90.0f;
    float cameraPitch = 0.0f;
    
    // Mouse and keyboard
    bool firstMouse = true;
    float lastX = 0.0f;
    float lastY = 0.0f;
    bool keys[1024];
    
    // Time
    float deltaTime = 0.0f;
    float lastFrame = 0.0f;
    
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
    
public:
    F1AirflowSimulator(int width = 1200, int height = 800) 
        : windowWidth(width), windowHeight(height), rng(std::chrono::steady_clock::now().time_since_epoch().count()), dist(0.0f, 1.0f) {
        
        initializeOpenGL();
        initializeShaders();
        initializeGeometry();
        initializeParticles();
        initializeVelocityField();
        
        // Set callbacks
        glfwSetWindowUserPointer(window, this);
        glfwSetKeyCallback(window, keyCallback);
        glfwSetCursorPosCallback(window, mouseCallback);
        glfwSetScrollCallback(window, scrollCallback);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
    
    ~F1AirflowSimulator() {
        cleanup();
    }
    
    void initializeOpenGL() {
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        
        window = glfwCreateWindow(windowWidth, windowHeight, "F1 Airflow Simulator", NULL, NULL);
        if (!window) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            exit(1);
        }
        
        glfwMakeContextCurrent(window);
        glewInit();
        
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_PROGRAM_POINT_SIZE);
        
        glViewport(0, 0, windowWidth, windowHeight);
    }
    
    GLuint compileShader(const char* source, GLenum type) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, NULL);
        glCompileShader(shader);
        
        GLint success;
        GLchar infoLog[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 512, NULL, infoLog);
            std::cerr << "Shader compilation failed: " << infoLog << std::endl;
        }
        
        return shader;
    }
    
    GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource) {
        GLuint vertexShader = compileShader(vertexSource, GL_VERTEX_SHADER);
        GLuint fragmentShader = compileShader(fragmentSource, GL_FRAGMENT_SHADER);
        
        GLuint program = glCreateProgram();
        glAttachShader(program, vertexShader);
        glAttachShader(program, fragmentShader);
        glLinkProgram(program);
        
        GLint success;
        GLchar infoLog[512];
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(program, 512, NULL, infoLog);
            std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        }
        
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        
        return program;
    }
    
    void initializeShaders() {
        particleShaderProgram = createShaderProgram(particleVertexShader, particleFragmentShader);
        geometryShaderProgram = createShaderProgram(geometryVertexShader, geometryFragmentShader);
    }
    
    void initializeGeometry() {
        // Create F1 car body
        auto carBody = std::make_unique<F1Geometry>(glm::vec3(0.8f, 0.1f, 0.1f));
        
        // Simple car body vertices (box-like shape)
        carBody->vertices = {
            // Front face
            -1.0f, -0.2f,  0.5f,  0.0f,  0.0f, 1.0f,
             1.0f, -0.2f,  0.5f,  0.0f,  0.0f, 1.0f,
             1.0f,  0.2f,  0.5f,  0.0f,  0.0f, 1.0f,
            -1.0f,  0.2f,  0.5f,  0.0f,  0.0f, 1.0f,
            
            // Back face
            -1.0f, -0.2f, -0.5f,  0.0f,  0.0f, -1.0f,
             1.0f, -0.2f, -0.5f,  0.0f,  0.0f, -1.0f,
             1.0f,  0.2f, -0.5f,  0.0f,  0.0f, -1.0f,
            -1.0f,  0.2f, -0.5f,  0.0f,  0.0f, -1.0f,
            
            // Top face
            -1.0f,  0.2f,  0.5f,  0.0f,  1.0f,  0.0f,
             1.0f,  0.2f,  0.5f,  0.0f,  1.0f,  0.0f,
             1.0f,  0.2f, -0.5f,  0.0f,  1.0f,  0.0f,
            -1.0f,  0.2f, -0.5f,  0.0f,  1.0f,  0.0f,
            
            // Bottom face
            -1.0f, -0.2f,  0.5f,  0.0f, -1.0f,  0.0f,
             1.0f, -0.2f,  0.5f,  0.0f, -1.0f,  0.0f,
             1.0f, -0.2f, -0.5f,  0.0f, -1.0f,  0.0f,
            -1.0f, -0.2f, -0.5f,  0.0f, -1.0f,  0.0f
        };
        
        carBody->indices = {
            0, 1, 2,  2, 3, 0,    // Front face
            4, 5, 6,  6, 7, 4,    // Back face
            8, 9, 10, 10, 11, 8,  // Top face
            12, 13, 14, 14, 15, 12 // Bottom face
        };
        
        glBindVertexArray(carBody->VAO);
        glBindBuffer(GL_ARRAY_BUFFER, carBody->VBO);
        glBufferData(GL_ARRAY_BUFFER, carBody->vertices.size() * sizeof(float), 
                     carBody->vertices.data(), GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, carBody->EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, carBody->indices.size() * sizeof(unsigned int), 
                     carBody->indices.data(), GL_STATIC_DRAW);
        
        // Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        f1Parts.push_back(std::move(carBody));
        
        // Create front wing
        auto frontWing = std::make_unique<F1Geometry>(glm::vec3(0.1f, 0.1f, 0.8f));
        frontWing->vertices = {
            // Front wing (flat rectangular shape)
            -0.3f, -0.3f,  0.6f,  0.0f,  1.0f,  0.0f,
             0.3f, -0.3f,  0.6f,  0.0f,  1.0f,  0.0f,
             0.3f, -0.3f,  0.8f,  0.0f,  1.0f,  0.0f,
            -0.3f, -0.3f,  0.8f,  0.0f,  1.0f,  0.0f
        };
        
        frontWing->indices = {
            0, 1, 2,  2, 3, 0
        };
        
        glBindVertexArray(frontWing->VAO);
        glBindBuffer(GL_ARRAY_BUFFER, frontWing->VBO);
        glBufferData(GL_ARRAY_BUFFER, frontWing->vertices.size() * sizeof(float), 
                     frontWing->vertices.data(), GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, frontWing->EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, frontWing->indices.size() * sizeof(unsigned int), 
                     frontWing->indices.data(), GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        f1Parts.push_back(std::move(frontWing));
        
        // Create rear wing
        auto rearWing = std::make_unique<F1Geometry>(glm::vec3(0.1f, 0.8f, 0.1f));
        rearWing->vertices = {
            // Rear wing (elevated rectangular shape)
            -0.4f,  0.2f, -0.6f,  0.0f,  1.0f,  0.0f,
             0.4f,  0.2f, -0.6f,  0.0f,  1.0f,  0.0f,
             0.4f,  0.2f, -0.8f,  0.0f,  1.0f,  0.0f,
            -0.4f,  0.2f, -0.8f,  0.0f,  1.0f,  0.0f
        };
        
        rearWing->indices = {
            0, 1, 2,  2, 3, 0
        };
        
        glBindVertexArray(rearWing->VAO);
        glBindBuffer(GL_ARRAY_BUFFER, rearWing->VBO);
        glBufferData(GL_ARRAY_BUFFER, rearWing->vertices.size() * sizeof(float), 
                     rearWing->vertices.data(), GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rearWing->EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, rearWing->indices.size() * sizeof(unsigned int), 
                     rearWing->indices.data(), GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        f1Parts.push_back(std::move(rearWing));
    }
    
    void initializeParticles() {
        particles.resize(maxParticles);
        
        // Generate particle data
        std::vector<float> particleData;
        for (int i = 0; i < maxParticles; ++i) {
            // Position
            particleData.push_back(-5.0f + dist(rng) * 2.0f); // x
            particleData.push_back(-2.0f + dist(rng) * 4.0f); // y
            particleData.push_back(-2.0f + dist(rng) * 4.0f); // z
            
            // Velocity
            particleData.push_back(windSpeed * std::cos(windAngle * M_PI / 180.0f)); // vx
            particleData.push_back(windSpeed * std::sin(windAngle * M_PI / 180.0f)); // vy
            particleData.push_back(0.0f); // vz
            
            // Age
            particleData.push_back(dist(rng) * 100.0f);
        }
        
        glGenVertexArrays(1, &particleVAO);
        glGenBuffers(1, &particleVBO);
        
        glBindVertexArray(particleVAO);
        glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
        glBufferData(GL_ARRAY_BUFFER, particleData.size() * sizeof(float), 
                     particleData.data(), GL_DYNAMIC_DRAW);
        
        // Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // Velocity attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        // Age attribute
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);
    }
    
    void initializeVelocityField() {
        velocityField.resize(gridSize);
        pressureField.resize(gridSize);
        
        for (int i = 0; i < gridSize; ++i) {
            velocityField[i].resize(gridSize);
            pressureField[i].resize(gridSize);
        }
        
        calculateVelocityField();
    }
    
    void calculateVelocityField() {
        float baseVelX = windSpeed * std::cos(windAngle * M_PI / 180.0f);
        float baseVelY = windSpeed * std::sin(windAngle * M_PI / 180.0f);
        
        for (int i = 0; i < gridSize; ++i) {
            for (int j = 0; j < gridSize; ++j) {
                float x = -5.0f + (10.0f * i) / gridSize;
                float y = -2.5f + (5.0f * j) / gridSize;
                
                glm::vec3 velocity(baseVelX, baseVelY, 0.0f);
                
                // Check collision with F1 car body
                if (x >= -1.0f && x <= 1.0f && y >= -0.2f && y <= 0.2f) {
                    velocity = glm::vec3(0.0f, 0.0f, 0.0f);
                }
                // Flow modification around front wing
                else if (x >= -0.3f && x <= 0.3f && y >= -0.3f && y <= -0.2f) {
                    velocity.y -= 20.0f; // Downforce effect
                    velocity.x *= 0.8f;  // Drag effect
                }
                // Flow modification around rear wing
                else if (x >= -0.4f && x <= 0.4f && y >= 0.2f && y <= 0.3f) {
                    velocity.y -= 30.0f; // Stronger downforce
                    velocity.x *= 0.6f;  // More drag
                }
                // Wake effect behind car
                else if (x > 1.0f && x < 3.0f && y >= -0.5f && y <= 0.5f) {
                    velocity.x *= 0.3f + 0.7f * (x - 1.0f) / 2.0f; // Gradual recovery
                    velocity.y += 10.0f * std::sin((x - 1.0f) * M_PI); // Turbulence
                }
                
                velocityField[i][j] = velocity;
                
                // Calculate pressure using Bernoulli's equation
                float velocityMagnitude = glm::length(velocity);
                pressureField[i][j] = 101325.0f - 0.5f * airDensity * velocityMagnitude * velocityMagnitude;
            }
        }
    }
    
    void updateParticles(float deltaTime) {
        std::vector<float> particleData;
        
        for (int i = 0; i < maxParticles; ++i) {
            Particle& p = particles[i];
            
            if (!p.active) {
                // Spawn new particle
                p.position = glm::vec3(-5.0f + dist(rng) * 2.0f, 
                                      -2.0f + dist(rng) * 4.0f, 
                                      -2.0f + dist(rng) * 4.0f);
                p.velocity = glm::vec3(windSpeed * std::cos(windAngle * M_PI / 180.0f),
                                      windSpeed * std::sin(windAngle * M_PI / 180.0f),
                                      0.0f);
                p.age = 0.0f;
                p.active = true;
            }
            
            // Update particle position
            p.position += p.velocity * deltaTime * 0.1f;
            p.age += deltaTime * 10.0f;
            
            // Interpolate velocity from field
            int gridX = static_cast<int>((p.position.x + 5.0f) / 10.0f * gridSize);
            int gridY = static_cast<int>((p.position.y + 2.5f) / 5.0f * gridSize);
            
            if (gridX >= 0 && gridX < gridSize && gridY >= 0 && gridY < gridSize) {
                p.velocity = velocityField[gridX][gridY];
            }
            
            // Reset particle if it's too old or out of bounds
            if (p.age > 100.0f || p.position.x > 10.0f || p.position.y > 5.0f || p.position.y < -5.0f) {
                p.active = false;
            }
            
            // Add to render data
            particleData.push_back(p.position.x);
            particleData.push_back(p.position.y);
            particleData.push_back(p.position.z);
            particleData.push_back(p.velocity.x);
            particleData.push_back(p.velocity.y);
            particleData.push_back(p.velocity.z);
            particleData.push_back(p.age);
        }
        
        // Update VBO
        glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, particleData.size() * sizeof(float), particleData.data());
    }
    
    void processInput() {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        
        float cameraSpeed = 2.5f * deltaTime;
        
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            cameraPos += cameraSpeed * cameraFront;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            cameraPos -= cameraSpeed * cameraFront;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
        
        // Wind control
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
            windSpeed += 10.0f * deltaTime;
            calculateVelocityField();
        }
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
            windSpeed -= 10.0f * deltaTime;
            windSpeed = std::max(0.0f, windSpeed);
            calculateVelocityField();
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
            windAngle -= 30.0f * deltaTime;
            calculateVelocityField();
        }
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
            windAngle += 30.0f * deltaTime;
            calculateVelocityField();
        }
    }
    
    void render() {
        glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Create transformation matrices
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 
                                               (float)windowWidth / (float)windowHeight, 
                                               0.1f, 100.0f);
        
        // Render F1 geometry
        glUseProgram(geometryShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(geometryShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(geometryShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(geometryShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        
        glm::vec3 lightPos(5.0f, 5.0f, 5.0f);
        glUniform3fv(glGetUniformLocation(geometryShaderProgram, "lightPos"), 1, glm::value_ptr(lightPos));
        glUniform3fv(glGetUniformLocation(geometryShaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos));
        
        for (const auto& part : f1Parts) {
            glUniform3fv(glGetUniformLocation(geometryShaderProgram, "objectColor"), 1, glm::value_ptr(part->color));
            glBindVertexArray(part->VAO);
            glDrawElements(GL_TRIANGLES, part->indices.size(), GL_UNSIGNED_INT, 0);
        }
        
        // Render particles
        glUseProgram(particleShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(particleShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(particleShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(particleShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform1f(glGetUniformLocation(particleShaderProgram, "time"), glfwGetTime());
        
        glBindVertexArray(particleVAO);
        glDrawArrays(GL_POINTS, 0, maxParticles);
        
        glfwSwapBuffers(window);
    }
    
    void run() {
        std::cout << "F1 Airflow Simulator Started" << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  WASD - Move camera" << std::endl;
        std::cout << "  Arrow keys - Control wind (Up/Down: speed, Left/Right: angle)" << std::endl;
        std::cout << "  Mouse - Look around" << std::endl;
        std::cout << "  ESC - Exit" << std::endl;
        std::cout << "Wind Speed: " << windSpeed << " m/s" << std::endl;
        std::cout << "Wind Angle: " << windAngle << " degrees" << std::endl;
        
        while (!glfwWindowShouldClose(window)) {
            float currentFrame = glfwGetTime();
            deltaTime = currentFrame - lastFrame;
            lastFrame = currentFrame;
            
            processInput();
            glfwPollEvents();
            
            updateParticles(deltaTime);
            render();
            
            // Print stats every 60 frames
            static int frameCount = 0;
            if (++frameCount % 60 == 0) {
                std::cout << "FPS: " << (1.0f / deltaTime) << 
                             " | Wind: " << windSpeed << " m/s @ " << windAngle << "°" << std::endl;
            }
        }
    }
    
    void cleanup() {
        glDeleteProgram(particleShaderProgram);
        glDeleteProgram(geometryShaderProgram);
        glDeleteVertexArrays(1, &particleVAO);
        glDeleteBuffers(1, &particleVBO);
        
        glfwTerminate();
    }
    
    // Static callback functions
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        F1AirflowSimulator* simulator = static_cast<F1AirflowSimulator*>(glfwGetWindowUserPointer(window));
        if (action == GLFW_PRESS) {
            simulator->keys[key] = true;
        } else if (action == GLFW_RELEASE) {
            simulator->keys[key] = false;
        }
    }
    
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
        F1AirflowSimulator* simulator = static_cast<F1AirflowSimulator*>(glfwGetWindowUserPointer(window));
        
        if (simulator->firstMouse) {
            simulator->lastX = xpos;
            simulator->lastY = ypos;
            simulator->firstMouse = false;
        }
        
        float xoffset = xpos - simulator->lastX;
        float yoffset = simulator->lastY - ypos;
        simulator->lastX = xpos;
        simulator->lastY = ypos;
        
        float sensitivity = 0.1f;
        xoffset *= sensitivity;
        yoffset *= sensitivity;
        
        simulator->cameraYaw += xoffset;
        simulator->cameraPitch += yoffset;
        
        if (simulator->cameraPitch > 89.0f)
            simulator->cameraPitch = 89.0f;
        if (simulator->cameraPitch < -89.0f)
            simulator->cameraPitch = -89.0f;
        
        glm::vec3 direction;
        direction.x = cos(glm::radians(simulator->cameraYaw)) * cos(glm::radians(simulator->cameraPitch));
        direction.y = sin(glm::radians(simulator->cameraPitch));
        direction.z = sin(glm::radians(simulator->cameraYaw)) * cos(glm::radians(simulator->cameraPitch));
        simulator->cameraFront = glm::normalize(direction);
    }
    
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        // Can be used for zoom functionality
    }
};

int main(int argc, char* argv[]) {
    try {
        F1AirflowSimulator simulator(1200, 800);
        simulator.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}