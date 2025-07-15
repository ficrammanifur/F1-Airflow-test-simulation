#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>

int main() {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile("f1_2026_v58.stl",
        aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);

    if(!scene) {
        std::cerr << "Error: " << importer.GetErrorString() << std::endl;
        return 1;
    }

    unsigned int totalVertices = 0;
    unsigned int totalFaces = 0;

    for(unsigned int i = 0; i < scene->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[i];
        totalVertices += mesh->mNumVertices;
        totalFaces += mesh->mNumFaces;
    }

    std::cout << "Vertices: " << totalVertices << std::endl;
    std::cout << "Faces: " << totalFaces << std::endl;

    return 0;
}
