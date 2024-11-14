#include <optix.h>
#include <vector_types.h>

#include "helpers.h"
#include "Soltrace.h"

extern "C" {
    __constant__ soltrace::LaunchParams params;
}

extern "C" {
    __constant__ soltrace::RayPaths* rayPaths;
}

static __device__ __inline__ soltrace::PerRayData getPayload()
{
    soltrace::PerRayData prd;
    prd.ray_path_index = optixGetPayload_0();
    prd.hit_count = optixGetPayload_1();
    return prd;
}

static __device__ __inline__ void setPayload(const soltrace::PerRayData& prd)
{
    optixSetPayload_0(prd.ray_path_index);
    optixSetPayload_1(prd.hit_count);
}

extern "C" __global__ void __closesthit__mirror()
{
    // TODO what is reinterpret_cast actually doing
    const soltrace::HitGroupData* sbt_data = reinterpret_cast<soltrace::HitGroupData*>(optixGetSbtDataPointer());
    const MaterialData::Mirror& mirror = sbt_data->material_data.mirror;

    soltrace::PerRayData& prd = *reinterpret_cast<soltrace::PerRayData*>(optixGetPayload_0());

    float3 object_normal = make_float3( __uint_as_float( optixGetAttribute_0() ), __uint_as_float( optixGetAttribute_1() ),
                                        __uint_as_float( optixGetAttribute_2() ) );
    float3 world_normal  = normalize( optixTransformNormalFromObjectToWorldSpace( object_normal ) );
    float3 ffnormal      = faceforward( world_normal, -optixGetWorldRayDirection(), world_normal );

    // incident ray
    // ideal reflection/refraction right now
    // monte carlo absorption? (done in soltrace)
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_t    = optixGetRayTmax();

    float3 hit_point = ray_orig + ray_t * ray_dir;

    //soltrace::PerRayData prd = getPayload();
    const unsigned int hit_count = prd.hit_count;
    const unsigned int ray_path_index = prd.ray_path_index;

    // TODO
    //soltrace::RayPaths& ray_path = hit_points[ray_path_index];
    //if (hit_count < soltrace::MAX_HITS) {
    //    soltrace::RayPaths[ray_path_index].hit_points[hit_count] = hit_point;
    //    prd.hit_count++;
    //}

    // Calculate ideal reflection direction
    float3 reflected_dir = reflect(ray_dir, ffnormal);

    // TODO: Add some noise here

    soltrace::RayPaths& ray_path = rayPaths[ray_path_index];
    if (hit_count < soltrace::MAX_HITS) {
        ray_path.hit_points[hit_count] = hit_point;
        prd.hit_count++;
    }


    // Store hit point in ray path if there's space left
    //if (payload.path_data.point_count < 10)
    //{
    //  payload.path_data.points[payload.path_data.point_count++] = hit_point;
    //}
    //optixSetPayload_0(__float_as_uint(hit_point.x));
    //optixSetPayload_1(__float_as_uint(hit_point.y));
    //optixSetPayload_2(__float_as_uint(hit_point.z));

    // Trace reflected ray
    optixTrace(
        params.handle,
        hit_point,
        reflected_dir,
        0.01f,  // Offset to avoid self-intersection
        1e16f,  // Max distance
        0.0f,   // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        soltrace::RAY_TYPE_RADIANCE,  // Use radiance ray type
        soltrace::RAY_TYPE_COUNT,
        soltrace::RAY_TYPE_RADIANCE,
        reinterpret_cast<unsigned int&>(prd.ray_path_index),
        reinterpret_cast<unsigned int&>(prd.hit_count)
    );

    setPayload(prd);
}

extern "C" __global__ void __closesthit__receiver()
{
    const soltrace::HitGroupData* sbt_data = reinterpret_cast<soltrace::HitGroupData*>(optixGetSbtDataPointer());
    const MaterialData::Receiver& receiver = sbt_data->material_data.receiver;
    
    soltrace::PerRayData prd = getPayload();

    float3 object_normal = make_float3( __uint_as_float( optixGetAttribute_0() ), __uint_as_float( optixGetAttribute_1() ),
                                        __uint_as_float( optixGetAttribute_2() ) );
    float3 world_normal  = normalize( optixTransformNormalFromObjectToWorldSpace( object_normal ) );
    float3 ffnormal      = faceforward( world_normal, -optixGetWorldRayDirection(), world_normal );

    // Incident ray
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_t    = optixGetRayTmax();

    float3 hit_point = ray_orig + ray_t * ray_dir;

    const unsigned int hit_count = prd.hit_count;
    const unsigned int ray_path_index = prd.ray_path_index;

    soltrace::RayPaths& ray_path = rayPaths[ray_path_index];
    if (hit_count < soltrace::MAX_HITS) {
        ray_path.hit_points[hit_count] = hit_point;
        prd.hit_count++;
    }


    ///* Store payload data here, e.g. Hit Point and associated data for flux map */
    //if (prd.hit_count < soltrace::MAX_HITS) {
    //    soltrace::RayPaths[prd.ray_path_index].hit_points[prd.hit_count] = hit_point;
    //    prd.hit_count++;
    //}
}

extern "C" __global__ void __miss__ms()
{
    // No action is taken here.
    // This function simply acts as a terminator for rays that miss all geometry.
}