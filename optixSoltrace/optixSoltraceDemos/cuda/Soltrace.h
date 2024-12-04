#pragma once

#include <vector_types.h>

#include <cuda/BufferView.h>

#include <cuda/GeometryDataST.h>
#include <cuda/MaterialDataST.h>

namespace soltrace
{
const unsigned int NUM_ATTRIBUTE_VALUES = 4u;
const unsigned int NUM_PAYLOAD_VALUES   = 4u;
const unsigned int MAX_TRACE_DEPTH      = 8u;
const unsigned int MAX_HITS             = 100u;     //  Maximum number of hit points to store in RayPath
    
struct HitGroupData
{
    GeometryData geometry_data;
    MaterialData material_data;
};

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_COUNT = 1         // not using occlusion/shadow rays atm
};

struct LaunchParams
{
    unsigned int                width;
    unsigned int                height;
    int                         max_depth;

    float3*                     output_buffer;      // TODO: Is this buffer only for image output? If so, can delete
    OptixTraversableHandle      handle;

    float3                      sun_center;
    float                       sun_radius;
    float                       max_sun_angle;
    int                         num_sun_points;
    float3                      scene_position;
};

struct RayPaths
{
    unsigned int num_hits;                       // Hit point count
    float3 hit_points[MAX_HITS];        // Hit points array
};

struct PerRayData
{
    unsigned int ray_path_index;  // Index of the ray in the ray path buffer
    unsigned int hit_count;       // Number of hits recorded along this ray
};

//struct RayPath
//{
//    float3 points[10];          // Array to store hit points along the ray path
//    int point_count;            // Tracks how many points have been recorded
//};
//
//struct PayloadRadiance
//{
//    RayPath path_data;          // Path data structure to store ray path
//};

// struct PayloadOcclusion         // not using shadow rays right now
// {
//     float3 result;
// };

} // end namespace soltrace