#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <cuda/Soltrace.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/Matrix.h>
#include <sutil/Record.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <GLFW/glfw3.h>
#include <iomanip>
#include <cstring>

// == Data Storage 

// == Configure OptiX Pipeline
// Initialize OptiX context
// - Initialize CUDA
// - Create OptiX context

// Create modules: each program needs to be compiled into a module, which are compiled from PTX (CUDA assembly code)
// - Ray generation
// - Miss
// - Hit

// Create program groups: groups organize the ray generation, miss, and hit programs into specific slots that pipeline uses during execution
// - Ray generation
// - Miss
// - Hit

// Create and configure pipeline

// Create shader binding table (SBT): links programs to specific ray types, serving as bridge between host and device
// - Allocate and fill records for ray generation
// - Allocate and fill records for miss
// - Allocate and fill records for hit

// Launch pipeline

// == Launch Optix 

// == Visualize

typedef sutil::Record<soltrace::HitGroupData> HitGroupRecord;

const uint32_t OBJ_COUNT = 2;
const int      max_trace = 12;

struct SoltraceState
{
    OptixDeviceContext          context                         = 0;
    OptixTraversableHandle      gas_handle                      = {};
    CUdeviceptr                 d_gas_output_buffer             = {};

    OptixModule                 geometry_module                 = 0;
    OptixModule                 shading_module                  = 0;
    OptixModule                 sun_module                      = 0;

    OptixProgramGroup           raygen_prog_group               = 0;
    OptixProgramGroup           radiance_miss_prog_group        = 0;
    OptixProgramGroup           radiance_mirror_prog_group      = 0;
    OptixProgramGroup           radiance_receiver_prog_group    = 0;

    OptixPipeline               pipeline                        = 0;
    OptixPipelineCompileOptions pipeline_compile_options        = {};

    CUstream                    stream                          = 0;
    soltrace::LaunchParams      params;
    soltrace::LaunchParams*     d_params                        = nullptr;
    soltrace::RayPaths          rayPaths;
    soltrace::RayPaths*         d_rayPaths                      = nullptr;

    OptixShaderBindingTable     sbt                             = {};
};

// == Scene Setup
const GeometryData::Parallelogram heliostat(
    make_float3( 10.0f, 0.0f, 0.0f ),   // v1
    make_float3( 0.0f, 10.0f, 0.0f ),   // v2
    make_float3( 10.0f, 10.0f, 0.0f )   // anchor
    );
const GeometryData::Parallelogram receiver(
    make_float3( 5.0f, 0.0f, 0.0f ),    // v1
    make_float3( 0.0f, 5.0f, 0.0f ),    // v2
    make_float3( 0.0f, 0.0f, 0.0f )     // anchor
    );

inline OptixAabb parallelogram_bound( float3 v1, float3 v2, float3 anchor )
{
    // v1 and v2 are scaled by 1./length^2.  Rescale back to normal for the bounds computation.
    const float3 tv1  = v1 / dot( v1, v1 );
    const float3 tv2  = v2 / dot( v2, v2 );
    const float3 p00  = anchor;
    const float3 p01  = anchor + tv1;
    const float3 p10  = anchor + tv2;
    const float3 p11  = anchor + tv1 + tv2;

    float3 m_min = fminf( fminf( p00, p01 ), fminf( p10, p11 ));
    float3 m_max = fmaxf( fmaxf( p00, p01 ), fmaxf( p10, p11 ));
    return {
        m_min.x, m_min.y, m_min.z,
        m_max.x, m_max.y, m_max.z
    };
}

static void buildGas(
    const SoltraceState &state,
    const OptixAccelBuildOptions &accel_options,
    const OptixBuildInput &build_input,
    OptixTraversableHandle &gas_handle,
    CUdeviceptr &d_gas_output_buffer
    )
{
    OptixAccelBufferSizes gas_buffer_sizes;
    CUdeviceptr d_temp_buffer_gas;

    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &build_input,
        1,
        &gas_buffer_sizes));

    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_temp_buffer_gas ),
        gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output and size of compacted GAS
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                compactedSizeOffset + 8
                ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK( optixAccelBuild(
        state.context,
        0,
        &accel_options,
        &build_input,
        1,
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle,
        &emitProperty,
        1) );

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

void createGeometry(SoltraceState& state)
{
    // Build custom primitives (parallelograms)

    // Load AABB into device memory
    OptixAabb aabb[OBJ_COUNT] = { parallelogram_bound(heliostat.v1, heliostat.v2, heliostat.anchor),
                                    parallelogram_bound(receiver.v1, receiver.v2, receiver.anchor) };

    CUdeviceptr d_aabb;

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb
        ), OBJ_COUNT * sizeof( OptixAabb ) ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_aabb ),
                &aabb,
                OBJ_COUNT * sizeof( OptixAabb ),
                cudaMemcpyHostToDevice
                ) );

    // Setup AABB build input
    uint32_t aabb_input_flags[] = {
        /* flags for heliostat */
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        /* flag for receiver */
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
    };

    const uint32_t sbt_index[] = { 0, 1 };
    CUdeviceptr    d_sbt_index;

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_sbt_index ), sizeof(sbt_index) ) );
    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast<void*>( d_sbt_index ),
        sbt_index,
        sizeof( sbt_index ),
        cudaMemcpyHostToDevice ) );

    OptixBuildInput aabb_input = {};
    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb;
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = OBJ_COUNT;
    aabb_input.customPrimitiveArray.numPrimitives = OBJ_COUNT;
    aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer         = d_sbt_index;
    aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes    = sizeof( uint32_t );
    aabb_input.customPrimitiveArray.primitiveIndexOffset         = 0;


    OptixAccelBuildOptions accel_options = {
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION,  // buildFlags
        OPTIX_BUILD_OPERATION_BUILD         // operation
    };


    buildGas(
        state,
        accel_options,
        aabb_input,
        state.gas_handle,
        state.d_gas_output_buffer);

    CUDA_CHECK( cudaFree( (void*)d_aabb) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>(d_sbt_index) ) );
}

void createModules( SoltraceState &state )
{
    OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( nullptr, nullptr, "parallelogram.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreate(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            input,
            inputSize,
            LOG, &LOG_SIZE,
            &state.geometry_module ) );
    }

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( nullptr, nullptr, "materials.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreate(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            input,
            inputSize,
            LOG, &LOG_SIZE,
            &state.shading_module ) );
    }

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( nullptr, nullptr, "sun.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreate(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            input,
            inputSize,
            LOG, &LOG_SIZE,
            &state.sun_module ) );
    }
}

static void createSunProgram( SoltraceState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           sun_prog_group;
    OptixProgramGroupOptions    sun_prog_group_options = {};
    OptixProgramGroupDesc       sun_prog_group_desc = {};
    sun_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    sun_prog_group_desc.raygen.module            = state.sun_module;
    sun_prog_group_desc.raygen.entryFunctionName = "__raygen__sun_source";

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &sun_prog_group_desc,
        1,
        &sun_prog_group_options,
        LOG, &LOG_SIZE,
        &sun_prog_group ) );

    program_groups.push_back(sun_prog_group);
    state.raygen_prog_group = sun_prog_group;
}

static void createMirrorProgram( SoltraceState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_mirror_prog_group;
    OptixProgramGroupOptions    radiance_mirror_prog_group_options = {};
    OptixProgramGroupDesc       radiance_mirror_prog_group_desc = {};
    radiance_mirror_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_mirror_prog_group_desc.hitgroup.moduleIS               = state.geometry_module;
    radiance_mirror_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__parallelogram";
    radiance_mirror_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    radiance_mirror_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__mirror";
    radiance_mirror_prog_group_desc.hitgroup.moduleAH               = nullptr;
    radiance_mirror_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &radiance_mirror_prog_group_desc,
        1,
        &radiance_mirror_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_mirror_prog_group ) );

    program_groups.push_back(radiance_mirror_prog_group);
    state.radiance_mirror_prog_group = radiance_mirror_prog_group;
}

static void createReceiverProgram( SoltraceState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_receiver_prog_group;
    OptixProgramGroupOptions    radiance_receiver_prog_group_options = {};
    OptixProgramGroupDesc       radiance_receiver_prog_group_desc = {};
    radiance_receiver_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_receiver_prog_group_desc.hitgroup.moduleIS               = state.geometry_module;
    radiance_receiver_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__parallelogram";
    radiance_receiver_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    radiance_receiver_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__receiver";
    radiance_receiver_prog_group_desc.hitgroup.moduleAH               = nullptr;
    radiance_receiver_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &radiance_receiver_prog_group_desc,
        1,
        &radiance_receiver_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_receiver_prog_group ) );

    program_groups.push_back(radiance_receiver_prog_group);
    state.radiance_receiver_prog_group = radiance_receiver_prog_group;
}

static void createMissProgram( SoltraceState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_miss_prog_group;
    OptixProgramGroupOptions    miss_prog_group_options = {};
    OptixProgramGroupDesc       miss_prog_group_desc = {};
    miss_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module             = state.shading_module;
    miss_prog_group_desc.miss.entryFunctionName  = "__miss__ms";

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1,
        &miss_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_miss_prog_group ) );

    program_groups.push_back(radiance_miss_prog_group);
    state.radiance_miss_prog_group = radiance_miss_prog_group;
}

void createPipeline( SoltraceState &state )
{
    std::vector<OptixProgramGroup> program_groups;

    state.pipeline_compile_options = {
        false,                                                  // usesMotionBlur
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,          // traversableGraphFlags
        2,    /* RadiancePRD uses 5 payloads */                 // numPayloadValues TODO
        5,    /* Parallelogram intersection uses 5 attrs */     // numAttributeValues TODO
        OPTIX_EXCEPTION_FLAG_NONE,                              // exceptionFlags
        "params"                                                // pipelineLaunchParamsVariableName
    };

    // Prepare program groups
    createModules( state );
    createSunProgram( state, program_groups );
    createMirrorProgram( state, program_groups );
    createReceiverProgram( state, program_groups );
    createMissProgram( state, program_groups );

    // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = max_trace;
    OPTIX_CHECK_LOG( optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups.data(),
        static_cast<unsigned int>( program_groups.size() ),
        LOG, &LOG_SIZE,
        &state.pipeline ) );

    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, state.pipeline ) );
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace,
                                             0,  // maxCCDepth
                                             0,  // maxDCDepth
                                             &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( state.pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            1  // maxTraversableDepth
                                            ) );
}

void createSBT( SoltraceState &state )
{
    // Raygen program record
    {
        CUdeviceptr d_raygen_record;
        size_t      sizeof_raygen_record = sizeof( sutil::EmptyRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_raygen_record ),
            sizeof_raygen_record ) );

        sutil::EmptyRecord rg_sbt;
        optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_raygen_record ),
            &rg_sbt,
            sizeof_raygen_record,
            cudaMemcpyHostToDevice
        ) );

        state.sbt.raygenRecord = d_raygen_record;
    }

    // Miss program record
    {
        CUdeviceptr d_miss_record;
        size_t sizeof_miss_record = sizeof( sutil::EmptyRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_miss_record ),
            sizeof_miss_record*soltrace::RAY_TYPE_COUNT ) );

        sutil::EmptyRecord ms_sbt[soltrace::RAY_TYPE_COUNT];
        optixSbtRecordPackHeader( state.radiance_miss_prog_group, &ms_sbt[0] );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_miss_record ),
            ms_sbt,
            sizeof_miss_record*soltrace::RAY_TYPE_COUNT,
            cudaMemcpyHostToDevice
        ) );

        state.sbt.missRecordBase          = d_miss_record;
        state.sbt.missRecordCount         = soltrace::RAY_TYPE_COUNT;
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof_miss_record );
    }

    // Hitgroup program record
    {
        const size_t count_records = soltrace::RAY_TYPE_COUNT * OBJ_COUNT;
        HitGroupRecord hitgroup_records[count_records];

        // Note: Fill SBT record array the same order like AS is built.
        int sbt_idx = 0;
        // TODO: Params - arbitrary right now
        // mirror
        OPTIX_CHECK( optixSbtRecordPackHeader(
            state.radiance_mirror_prog_group,
            &hitgroup_records[sbt_idx] ) );
        hitgroup_records[ sbt_idx ].data.geometry_data.setParallelogram( heliostat );
        hitgroup_records[ sbt_idx ].data.material_data.mirror = {
            0.95f,      // reflectivity
            0.0f,       // transmissivity
            0.0f,       // slope error
            0.0f        // specularity error
        };
        sbt_idx++;

        // receiver
        OPTIX_CHECK( optixSbtRecordPackHeader(
            state.radiance_receiver_prog_group,
            &hitgroup_records[sbt_idx] ) );
        hitgroup_records[ sbt_idx ].data.geometry_data.setParallelogram( receiver );
        hitgroup_records[ sbt_idx ].data.material_data.receiver = {
            0.05f,      // reflectivity
            0.0f,       // transmissivity
            0.0f,       // slope error
            0.0f        // specularity error
        };
        sbt_idx++;

        CUdeviceptr d_hitgroup_records;
        size_t      sizeof_hitgroup_record = sizeof( HitGroupRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_hitgroup_records ),
            sizeof_hitgroup_record*count_records
        ) );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_hitgroup_records ),
            hitgroup_records,
            sizeof_hitgroup_record*count_records,
            cudaMemcpyHostToDevice
        ) );

        state.sbt.hitgroupRecordBase            = d_hitgroup_records;
        state.sbt.hitgroupRecordCount           = count_records;
        state.sbt.hitgroupRecordStrideInBytes   = static_cast<uint32_t>( sizeof_hitgroup_record );
    }
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
        << message << "\n";
}

void createContext( SoltraceState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext          cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;     // TODO: setup some logs for monitoring sim
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    state.context = context;
}

void initLaunchParams( SoltraceState& state )
{
    state.params.max_depth = max_trace;

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( soltrace::LaunchParams ) ) );

    state.params.handle = state.gas_handle;
}


void cleanupState( SoltraceState& state )
{
    OPTIX_CHECK( optixPipelineDestroy     ( state.pipeline                ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.raygen_prog_group       ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_mirror_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_receiver_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_miss_prog_group         ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.shading_module          ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.geometry_module         ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context                 ) );


    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer    ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params               ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_rayPaths             ) ) );
}

int main(int argc, char* argv[])
{
    SoltraceState state;

    try
    {
        // Initialize sun
        state.params.sun_center = make_float3(10.0f, 50.0f, 10.0f);
        state.params.sun_radius = 10.0;
        state.params.num_sun_points = 1000;
        state.params.scene_position = make_float3(0.0f, 0.0f, 0.0f); 

        state.params.width  = 100;
        state.params.height = 10;

        // Initialize output buffer
        const int num_rays = state.params.width * state.params.height;
        //soltrace::RayPaths* rayPaths;
        //cudaMallocManaged(&rayPaths, num_rays * sizeof(soltrace::RayPaths));
        //cudaMemset(rayPaths, 0, num_rays * sizeof(soltrace::RayPaths));
        //state.context["rayPathsBuffer"]->setUserData(sizeof(soltrace::RayPaths*), &rayPaths);

        //CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_rayPaths),
        //    &state.rayPaths,
        //    sizeof(soltrace::RayPaths),
        //    cudaMemcpyHostToDevice,
        //    state.stream
        //));


        createContext(state);
        createGeometry(state);
        createPipeline(state);
        createSBT(state);

        initLaunchParams(state);

        OPTIX_CHECK( optixLaunch(
        state.pipeline,
        state.stream,
        reinterpret_cast<CUdeviceptr>( state.d_params ),
        sizeof( soltrace::LaunchParams ),
        &state.sbt,
        state.params.width,  // launch width, raygen point location
        state.params.height, // launch height, number of raygens per location
        1                    // launch depth - TODO: 1 is typical, but what does this actually mean?
        ) );

        //for (int i = 0; i < num_rays; i++) {
        //    const soltrace::RayPaths& ray = state.rayPaths[i];
        //    printf("Ray %d has %d hit points:\n", i, ray.num_hits);
        //    for (int j = 0; j < ray.num_hits; j++) {
        //        printf("Hit %d: (%f, %f, %f)\n", j, ray.hit_points[j].x, ray.hit_points[j].y, ray.hit_points[j].z);
        //    }
        //}

        cleanupState(state);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}