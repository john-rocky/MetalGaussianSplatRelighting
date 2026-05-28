#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

constant const int kMaxViewCount = 2;
constant static const half kBoundsRadius = 3;
constant static const half kBoundsRadiusSquared = kBoundsRadius*kBoundsRadius;

// Spherical Harmonics constants
// These are normalization factors for SH basis functions
constant const float SH_C0 = 0.28209479177387814f;  // 1/(2*sqrt(pi))
constant const float SH_C1 = 0.4886025119029199f;   // sqrt(3/(4*pi))

constant const float SH_C2_0 =  1.0925484305920792f;  //  0.5 * sqrt(15/pi)
constant const float SH_C2_1 = -1.0925484305920792f;  // -0.5 * sqrt(15/pi)
constant const float SH_C2_2 =  0.31539156525252005f; //  0.25 * sqrt(5/pi)
constant const float SH_C2_3 = -1.0925484305920792f;  // -0.5 * sqrt(15/pi)
constant const float SH_C2_4 =  0.5462742152960396f;  //  0.25 * sqrt(15/pi)

constant const float SH_C3_0 = -0.5900435899266435f;  // -0.25 * sqrt(35/(2*pi))
constant const float SH_C3_1 =  2.890611442640554f;   //  0.5 * sqrt(105/pi)
constant const float SH_C3_2 = -0.4570457994644658f;  // -0.25 * sqrt(21/(2*pi))
constant const float SH_C3_3 =  0.3731763325901154f;  //  0.25 * sqrt(7/pi)
constant const float SH_C3_4 = -0.4570457994644658f;  // -0.25 * sqrt(21/(2*pi))
constant const float SH_C3_5 =  1.445305721320277f;   //  0.25 * sqrt(105/pi)
constant const float SH_C3_6 = -0.5900435899266435f;  // -0.25 * sqrt(35/(2*pi))

// Spherical harmonics degree enum - must match Swift SHDegree
enum SHDegree: uint8_t
{
    SHDegree0 = 0,  // 1 coefficient (DC only)
    SHDegree1 = 1,  // 4 coefficients
    SHDegree2 = 2,  // 9 coefficients
    SHDegree3 = 3   // 16 coefficients
};

enum BufferIndex: int32_t
{
    BufferIndexUniforms    = 0,
    BufferIndexChunks      = 1,
    BufferIndexSplatIndex  = 2,
    BufferIndexRelight     = 3,
};

typedef struct
{
    matrix_float4x4 projectionMatrix;
    matrix_float4x4 viewMatrix;
    packed_float3 cameraPosition;  // World-space camera position for SH evaluation
    uint _padding0;                // Padding for alignment
    uint2 screenSize;

    // Precomputed values for covariance projection (derived from projectionMatrix and screenSize)
    float focalX;                  // screenSize.x * projectionMatrix[0][0] / 2
    float focalY;                  // screenSize.y * projectionMatrix[1][1] / 2
    float tanHalfFovX;             // 1 / projectionMatrix[0][0]
    float tanHalfFovY;             // 1 / projectionMatrix[1][1]

    uint chunkCount;

    /*
     The first N splats are represented as 2N primitives and 4N vertex indices. The remainder are represented
     as instances of these first N. This allows us to limit the size of the indexed array (and associated memory),
     but also avoid the performance penalty of a very large number of instances.
     */
    uint splatCount;
    uint indexedSplatCount;
} Uniforms;

typedef struct
{
    Uniforms uniforms[kMaxViewCount];
} UniformsArray;

// Relightable rendering parameters. Bound at BufferIndexRelight to the splat vertex shaders.
// Keep in sync with Swift: SplatRenderer.RelightUniforms
typedef struct
{
    matrix_float4x4 envRotation;          // rotates the world-space sample direction (environment orientation)
    matrix_float4x4 inverseViewProjection;// unprojects screen pixels to world-space rays (skybox background)
    uint enabled;                         // 0 = standard SH path, non-zero = split-sum IBL relight
    uint debugMode;                       // 0 shaded, 1 normal, 2 roughness, 3 reflectionStrength
    uint prefilteredMipCount;             // mip levels in the prefiltered environment cubemap
    float envIntensity;                   // scales sampled environment radiance
    float roughnessOverride;              // >= 0 overrides per-splat roughness (useful for synthetic materials)
    float reflectionStrengthOverride;     // >= 0 overrides per-splat reflection strength
    vector_float2 screenSize;             // render-target size in pixels (skybox ray reconstruction)
    uint arBackground;                    // 1 = composite over the AR camera image (texture 4) instead of the skybox
    uint _pad0;                           // padding
    vector_float2 _pad1;                  // padding (align the tint vector to 16 bytes)
    vector_float4 tint;                   // configurator repaint: xyz = paint color (linear), w = strength (0 = original)
} RelightUniforms;

// Keep in sync with EncodedSplatPoint
typedef struct
{
    packed_float3 position;
    packed_half4 color;
    packed_half3 covA;
    packed_half3 covB;
} Splat;

// Keep in sync with Swift: EncodedSplatMaterial (16 bytes)
typedef struct
{
    packed_half3 normal;             // world-space surface normal
    packed_half3 specularTint;       // specular tint / F0 color
    half roughness;
    half reflectionStrength;         // blends diffuse vs. specular
} SplatMaterial;

// Keep in sync with Swift: ChunkedSplatIndex
typedef struct
{
    uint16_t chunkIndex;
    uint16_t _padding;
    uint32_t splatIndex;
} ChunkedSplatIndex;

// Information about a single chunk, used in the chunk table
typedef struct
{
    device Splat* splats;
    device half* shCoefficients;   // Null for SH degree 0, otherwise higher-order SH coefficients
    device SplatMaterial* materials; // Null if this chunk has no relightable material
    uint32_t splatCount;
    SHDegree shDegree;             // Spherical harmonics degree for this chunk
    uint8_t enabled;               // Non-zero = enabled for rendering
    uint8_t _shPadding[2];         // Padding for alignment
} ChunkInfo;

typedef struct
{
    float4 position [[position]];
    half2 relativePosition; // Ranges from -kBoundsRadius to +kBoundsRadius
    half4 color;
    // Deferred relighting G-buffer inputs (multi-stage path). Per-splat, constant across the quad.
    half3 gNormal;   // world-space surface normal, faceforwarded toward the camera
    half3 gView;     // world-space view direction (camera - splat center)
    half2 gMaterial; // (roughness, reflectionStrength)
    half3 gOriColor; // ori_color (Ref-Gaussian albedo / specular F0 tint), in [0,1]
} FragmentIn;
