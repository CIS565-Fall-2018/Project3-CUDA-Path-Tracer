#pragma once

#include <iostream>
#include <array>
#include <glm/detail/type_vec3.hpp>
#include <glm/detail/type_vec2.hpp>
#include <glm/common.hpp>

#include "tiny_obj_loader.h"

//using https://github.com/syoyo/tinyobjloader

struct Vertex
{
    tinyobj::index_t index{};
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texture_coordinates;
};

struct Triangle
{
    Vertex vertices[3];
};

struct MeshObject
{
    Triangle* triangles;
    std::size_t num_triangles;
    Triangle* dev_triangles;
    glm::vec3 bounding_min;
    glm::vec3 bounding_max;
};

struct MeshLoader
{
    explicit MeshLoader(std::string file_path_);
    MeshObject load_mesh_object();
    std::string file_path;
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
};
