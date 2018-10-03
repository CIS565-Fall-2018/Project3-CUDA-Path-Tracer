#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"
#include <iostream>
#include "objLoader.h"

int loadObj(std::string inputfile, Triangle* triangles)
{
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string err;
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, inputfile.c_str());

  if (!err.empty()) { // `err` may contain warning message.
    std::cerr << err << std::endl;
  }

  if (!ret) {
    std::cerr << "couldn't load obj" << std::endl;
    exit(1);
  }

  std::vector<Triangle> triangles_vec;

  // Loop over shapes
  for (size_t s = 0; s < shapes.size(); s++) {
    // Loop over faces(polygon)
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      int fv = shapes[s].mesh.num_face_vertices[f];

      if (fv != 3)
      {
        std::cerr << "Error: Mesh contains polygons that aren't triangles" << std::endl;
        exit(1);
      }

      std::vector<glm::vec3> vertices;
      std::vector<glm::vec3> normals;

      // Loop over vertices in the face.
      for (size_t v = 0; v < fv; v++) {
        // access to vertex
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
        tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
        tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
        tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
        tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
        tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
        tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];
       // tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
      //  tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];
        vertices.push_back(glm::vec3(vx, vy, vz));
        normals.push_back(glm::vec3(nx, ny, nz));

        // Optional: vertex colors
        // tinyobj::real_t red = attrib.colors[3*idx.vertex_index+0];
        // tinyobj::real_t green = attrib.colors[3*idx.vertex_index+1];
        // tinyobj::real_t blue = attrib.colors[3*idx.vertex_index+2];
      }
      index_offset += fv;

      Triangle newTriangle;
      newTriangle.v1 = vertices[0];
      newTriangle.v2 = vertices[1];
      newTriangle.v3 = vertices[2];
      newTriangle.n =( normals[0] + normals[1] + normals[2] ) / 3.0f;
      triangles_vec.push_back(newTriangle);
      // per-face material
      shapes[s].mesh.material_ids[f];
    }
  }

  triangles = new Triangle[triangles_vec.size()];
  for (int i = 0; i < triangles_vec.size(); ++i)
  {
    triangles[i] = triangles_vec[i];
  }

  return triangles_vec.size();
}