#import "ShaderCommon.h"

// MARK: - Spherical Harmonics

/// Evaluates spherical harmonics to compute view-dependent color.
half3 evaluateSH(float3 dir,
                 half3 sh0,
                 device const half* shCoeffs,
                 SHDegree shDegree,
                 uint splatIndex);

// MARK: - Covariance

float3 calcCovariance2D(float3 viewPos,
                        packed_half3 cov3Da,
                        packed_half3 cov3Db,
                        float4x4 viewMatrix,
                        float focalX,
                        float focalY,
                        float tanHalfFovX,
                        float tanHalfFovY);

void decomposeCovariance(float3 cov2D, thread float2 &v1, thread float2 &v2);

// MARK: - Vertex Processing

/// Vertex processing with spherical harmonics evaluation.
/// All splats use this path - splat.color contains raw SH0 coefficients.
/// For SH degree 0, shCoefficients can be nullptr.
FragmentIn splatVertex(Splat splat,
                       Uniforms uniforms,
                       uint relativeVertexIndex,
                       device const half* shCoefficients,
                       SHDegree shDegree,
                       uint splatIndex);

// MARK: - Fragment Processing

half splatFragmentAlpha(half2 relativePosition, half splatAlpha);

// MARK: - Relighting (split-sum IBL)

/// Computes a relit color for a splat using split-sum image-based lighting.
/// `baseColorLinear` is the splat's existing (SH-evaluated, linear) color, used as the diffuse term.
/// Returns the final linear color: (1 - reflectionStrength) * base + specular.
half3 shadePBR(half3 baseColorLinear,
               SplatMaterial material,
               float3 worldPosition,
               float3 cameraPosition,
               constant RelightUniforms &relight,
               texturecube<float> prefilteredEnv,
               texturecube<float> irradianceEnv,
               texture2d<float> brdfLUT,
               sampler iblSampler);

/// Deferred split-sum IBL shading. Takes explicit per-pixel inputs (already blended + normalized in
/// the multi-stage postprocess), so the noisy per-splat normals are averaged before shading.
/// `baseColorLinear` is the SH diffuse base color; `oriColorLinear` is the Ref-Gaussian albedo
/// (ori_color) that tints the specular F0. final = (1 - refl) * base + specular.
half3 shadeIBLDeferred(half3 baseColorLinear,
                       half3 oriColorLinear,
                       float3 N,
                       float3 V,
                       float roughness,
                       float reflectionStrength,
                       constant RelightUniforms &relight,
                       texturecube<float> prefilteredEnv,
                       texturecube<float> irradianceEnv,
                       texture2d<float> brdfLUT,
                       sampler iblSampler);
