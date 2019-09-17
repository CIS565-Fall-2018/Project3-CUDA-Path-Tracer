#include "mesh_object.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

MeshLoader::MeshLoader(std::string file_path_) : file_path(std::move(file_path_))
{
    std::string error;
    const bool loaded_correctly = LoadObj(&attributes, &shapes, &materials, &error, file_path.c_str());
    if (loaded_correctly)
    {
        std::cout << "Loaded mesh object: " << file_path << "\n";
    }
    else
    {
        std::cerr << "Failed to load mesh - path: " << file_path << std::endl;
        exit(-1);
    }
}

//modified from https://github.com/syoyo/tinyobjloader
MeshObject MeshLoader::load_mesh_object()
{   
    MeshObject object{};
    std::vector<Triangle> vec_triangles;
    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++)
    {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
        {
            int fv = shapes[s].mesh.num_face_vertices[f];
            // Loop over vertices in the face.
            Triangle triangle;
            for (size_t v = 0; v < fv; v++)
            {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                Vertex vertex;
                vertex.index = idx;
                if (!attributes.vertices.empty())
                {
                    vertex.position[0] = attributes.vertices[3 * idx.vertex_index + 0];
                    vertex.position[1] = attributes.vertices[3 * idx.vertex_index + 1];
                    vertex.position[2] = attributes.vertices[3 * idx.vertex_index + 2];
                }
                if (!attributes.normals.empty())
                {
                    vertex.normal[0] = attributes.normals[3 * idx.normal_index + 0];
                    vertex.normal[1] = attributes.normals[3 * idx.normal_index + 1];
                    vertex.normal[2] = attributes.normals[3 * idx.normal_index + 2];
                }
                if (!attributes.texcoords.empty())
                {
                    vertex.texture_coordinates[0] = attributes.texcoords[2 * idx.texcoord_index + 0];
                    vertex.texture_coordinates[1] = attributes.texcoords[2 * idx.texcoord_index + 1];
                }

                triangle.vertices[v] = vertex;
            }
            index_offset += fv;
            vec_triangles.emplace_back(triangle);
            // per-face material
            shapes[s].mesh.material_ids[f];
        }
    }
    object.num_triangles = vec_triangles.size();
    object.triangles = new Triangle[object.num_triangles];
    std::copy(std::begin(vec_triangles), std::end(vec_triangles), object.triangles);

    //find the object that has the minimum/maximum
    for(const auto& triangle : vec_triangles)
    {
        object.bounding_min = min(object.bounding_min, triangle.vertices[0].position);
        object.bounding_min = glm::min(object.bounding_min, triangle.vertices[1].position);
        object.bounding_min = glm::min(object.bounding_min, triangle.vertices[2].position);

        object.bounding_max = glm::max(object.bounding_max, triangle.vertices[0].position);
        object.bounding_max = glm::max(object.bounding_max, triangle.vertices[1].position);
        object.bounding_max = glm::max(object.bounding_max, triangle.vertices[2].position);
    }

    std::cout << "cube num " << object.num_triangles;

    return object;
}
