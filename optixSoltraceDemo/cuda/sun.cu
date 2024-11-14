#include <optix.h>
#include <curand_kernel.h>
#include <vector_types.h>

#include "helpers.h"
#include "random.h"
#include "Soltrace.h"

extern "C" {
    __constant__ soltrace::LaunchParams params;
}

__device__ float2 samplePointInDisk(float radius, unsigned int seed) {
    curandState rng_state;
    curand_init(seed, 0, 0, &rng_state);

    float r = radius * sqrtf(curand_uniform(&rng_state));
    float theta = 2.0f * M_PIf * curand_uniform(&rng_state);

    return make_float2(r * cosf(theta), r * sinf(theta));
}

__device__ float3 sampleRayDirection(float max_angle, unsigned int seed) {
    curandState rng_state;
    curand_init(seed, 0, 0, &rng_state);

    // Sampling a random direction within a cone 
    float angle = max_angle + curand_uniform(&rng_state);   // Random angle within max angular spread
    float phi = 2.0f * M_PIf * curand_uniform(&rng_state);  // Random azimuthal angle

    // Convert spherical coordinaes to Cartesian for direction
    float x = sinf(angle) * cosf(phi);
    float y = sinf(angle) * sinf(phi);
    float z = -cosf(angle);

    return normalize(make_float3(x, y, z));
}

// == Ray Generation Program
extern "C" __global__ void __raygen__sun_source()
{
    // Lookup location in launch grid here
    const uint3 launch_idx = optixGetLaunchIndex();
    const uint3 launch_dims = optixGetLaunchDimensions();
    const unsigned int ray_number = launch_idx.y * launch_dims.x + launch_idx.x;
    const unsigned int seed = launch_idx.x; // Use launch index to seed RNG for unique sampling

    float2 sun_sample_pos = samplePointInDisk(params.sun_radius, seed);

    // Sample emission angle here - capturing sun distribution
    // TODO need to update for sun's position angle
    const float3 ray_gen_pos = params.sun_center + make_float3(sun_sample_pos.x, sun_sample_pos.y, 0.0f);

    float3 initial_ray_dir = normalize(params.scene_position - ray_gen_pos);
    float3 ray_dir = initial_ray_dir + sampleRayDirection(params.max_sun_angle, seed);

    soltrace::PerRayData prd;
    prd.ray_path_index = ray_number;
    prd.hit_count = 0;

    // Cast and trace the ray through the scene
    optixTrace(
        params.handle, ray_gen_pos, ray_dir,
        0.001f,                // Minimum distance
        1e16f,                 // Maximum distance
        0.0f,                  // Time
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        soltrace::RAY_TYPE_RADIANCE,     // Ray type
        soltrace::RAY_TYPE_COUNT,         // Number of ray types
        soltrace::RAY_TYPE_RADIANCE,     // SBT offset
        reinterpret_cast<unsigned int&>(prd.ray_path_index),
        reinterpret_cast<unsigned int&>(prd.hit_count)
    );
}


