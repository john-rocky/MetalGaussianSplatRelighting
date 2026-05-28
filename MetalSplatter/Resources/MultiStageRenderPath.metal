#include "SplatProcessing.h"

// Multi-stage G-buffer, blended per pixel in tile memory. Color uses standard over-compositing
// (every splat). Normal + material use a *camera-facing-only* weight so the far side of the surface
// (whose consistently-outward normals point away from the camera) cannot blend in and cancel the
// front normal into noise. The postprocess pass recovers per-pixel averages and shades once.
typedef struct
{
    half4 color [[raster_order_group(0)]];         // rgb: base_color (SH) * a (premultiplied), a: accumulated alpha (coverage)
    half4 normalRough [[raster_order_group(0)]];   // rgb: normal * w,  a: roughness * w
    half4 viewRefl [[raster_order_group(0)]];      // rgb: viewDir * w, a: reflectionStrength * w
    half4 oriColor [[raster_order_group(0)]];      // rgb: ori_color (albedo/F0 tint) * w, a: unused
    half materialWeight [[raster_order_group(0)]]; // accumulated alpha weight (normalizes normal/material); == coverage
    float depth [[raster_order_group(0)]];
} FragmentValues;

typedef struct
{
    FragmentValues values [[imageblock_data]];
} FragmentStore;

typedef struct
{
    half4 color [[color(0)]];
    float depth [[depth(any)]];
} FragmentOut;

kernel void initializeFragmentStore(imageblock<FragmentValues, imageblock_layout_explicit> blockData,
                                    ushort2 localThreadID [[thread_position_in_threadgroup]]) {
    threadgroup_imageblock FragmentValues *values = blockData.data(localThreadID);
    values->color = { 0, 0, 0, 0 };
    values->normalRough = { 0, 0, 0, 0 };
    values->viewRefl = { 0, 0, 0, 0 };
    values->oriColor = { 0, 0, 0, 0 };
    values->materialWeight = 0;
    values->depth = 0;
}

vertex FragmentIn multiStageSplatVertexShader(uint vertexID [[vertex_id]],
                                              uint instanceID [[instance_id]],
                                              ushort amplificationID [[amplification_id]],
                                              device const ChunkInfo* chunks [[ buffer(BufferIndexChunks) ]],
                                              constant ChunkedSplatIndex* splatIndexArray [[ buffer(BufferIndexSplatIndex) ]],
                                              constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]]) {
    Uniforms uniforms = uniformsArray.uniforms[min(int(amplificationID), kMaxViewCount)];

    uint splatID = instanceID * uniforms.indexedSplatCount + (vertexID / 4);
    if (splatID >= uniforms.splatCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    ChunkedSplatIndex idx = splatIndexArray[splatID];

    // Bounds check chunk index
    if (idx.chunkIndex >= uniforms.chunkCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    ChunkInfo chunk = chunks[idx.chunkIndex];

    // Bounds check local splat index; protects against transient stale indices.
    if (idx.splatIndex >= chunk.splatCount) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    if (!chunk.enabled) {
        FragmentIn out;
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    Splat splat = chunk.splats[idx.splatIndex];

    FragmentIn out = splatVertex(splat, uniforms, vertexID % 4,
                                 chunk.shCoefficients, chunk.shDegree,
                                 idx.splatIndex);

    // Deferred relighting: pass the per-splat material through to the G-buffer. Split-sum shading
    // runs per pixel in the postprocess pass, after normals are blended (camera-facing only).
    // Normals are oriented consistently outward at load time, so dot(N, V) > 0 means front-facing.
    if (chunk.materials != nullptr) {
        SplatMaterial material = chunk.materials[idx.splatIndex];
        float3 N = normalize(float3(material.normal));
        float3 V = normalize(float3(uniforms.cameraPosition) - float3(splat.position));
        out.gNormal = half3(N);
        out.gView = half3(V);
        out.gMaterial = half2(material.roughness, material.reflectionStrength);
        out.gOriColor = half3(material.specularTint);
    } else {
        out.gNormal = half3(0, 0, 1);
        out.gView = half3(0, 0, 1);
        out.gMaterial = half2(0.5h, 0.0h);
        out.gOriColor = half3(1, 1, 1);
    }

    return out;
}

fragment FragmentStore multiStageSplatFragmentShader(FragmentIn in [[stage_in]],
                                                     FragmentValues previousFragmentValues [[imageblock_data]]) {
    FragmentStore out;

    half alpha = splatFragmentAlpha(in.relativePosition, in.color.a);
    half oneMinusAlpha = 1 - alpha;

    // Color / coverage: every splat contributes (standard premultiplied over-compositing).
    out.values.color = previousFragmentValues.color * oneMinusAlpha + half4(in.color.rgb * alpha, alpha);

    // Normal + material: orient each splat's normal to FACE THE CAMERA (Ref-Gaussian flip_align_view:
    // flip when dot(N, V) < 0), then accumulate ALL splats with alpha. The previous code used a binary
    // camera-facing *gate* (exclude back-facing) which produced a mottled normal map: at 2DGS sign-
    // transition boundaries only a noisy subset passed the gate and opposite-signed neighbors cancelled
    // to near-zero. Flipping (instead of excluding) resolves the sign ambiguity per-view -> smooth normal.
    half facing = (dot(float3(in.gNormal), float3(in.gView)) < 0.0f) ? -1.0h : 1.0h;
    half3 viewNormal = in.gNormal * facing;
    half w = alpha;
    out.values.normalRough = previousFragmentValues.normalRough * oneMinusAlpha + half4(viewNormal * w, in.gMaterial.x * w);
    out.values.viewRefl = previousFragmentValues.viewRefl * oneMinusAlpha + half4(in.gView * w, in.gMaterial.y * w);
    out.values.oriColor = previousFragmentValues.oriColor * oneMinusAlpha + half4(in.gOriColor * w, 0.0h);
    out.values.materialWeight = previousFragmentValues.materialWeight * oneMinusAlpha + w;

    float previousDepth = previousFragmentValues.depth;
    float depth = in.position.z;
    out.values.depth = previousDepth * oneMinusAlpha + depth * alpha;

    return out;
}

constexpr sampler skySampler(coord::normalized, s_address::repeat, t_address::clamp_to_edge, filter::linear);
constexpr sampler arBgSampler(coord::normalized, address::clamp_to_edge, filter::linear);

// Sample the full-resolution equirect environment as a skybox along this pixel's world-space view
// ray (reconstructed by unprojecting near+far through the inverse view-projection). Uses the same
// direction->uv mapping as IBL.metal's dirToEquirectUV so the background matches the reflections.
static float3 skyboxColor(float4 fragCoord,
                          constant RelightUniforms & relight,
                          texture2d<float> environmentEquirect) {
    float2 uv = fragCoord.xy / max(relight.screenSize, float2(1.0f));
    float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    float4 nearH = relight.inverseViewProjection * float4(ndc, 0.0f, 1.0f);  // Metal NDC near z = 0
    float4 farH  = relight.inverseViewProjection * float4(ndc, 1.0f, 1.0f);  //            far  z = 1
    float3 dir = normalize(farH.xyz / farH.w - nearH.xyz / nearH.w);
    dir = (relight.envRotation * float4(dir, 0.0f)).xyz;                      // rotate with the env
    float2 euv = float2(atan2(dir.z, dir.x) / (2.0f * M_PI_F) + 0.5f,
                        acos(clamp(dir.y, -1.0f, 1.0f)) / M_PI_F);
    return environmentEquirect.sample(skySampler, euv).rgb * relight.envIntensity;
}

// Resolve the blended G-buffer to a final color. With relighting enabled (shaded mode) this runs
// split-sum IBL per pixel on the camera-facing-averaged normal/material and composites the result
// over the environment skybox; debug modes keep a transparent background (original behavior).
half4 resolveRelight(FragmentValues v,
                     float4 fragCoord,
                     constant RelightUniforms & relight,
                     texturecube<float> prefilteredEnv,
                     texturecube<float> irradianceEnv,
                     texture2d<float> brdfLUT,
                     texture2d<float> environmentEquirect,
                     texture2d<float> arCameraImage,
                     sampler iblSampler) {
    half coverage = v.color.a;
    half mw = v.materialWeight;

    // Background: only in shaded mode with relighting on; debug views keep a transparent bg. In AR
    // mode the live camera image (already display-transformed + linearized) replaces the skybox, so
    // the splat composites over the real room; otherwise the equirect skybox is reconstructed by ray.
    bool showBg = (relight.enabled != 0 && relight.debugMode == 0);
    half3 bg = half3(0.0h);
    if (showBg) {
        if (relight.arBackground != 0) {
            float2 uv = fragCoord.xy / max(relight.screenSize, float2(1.0f));
            bg = half3(arCameraImage.sample(arBgSampler, uv).rgb);
        } else {
            bg = half3(skyboxColor(fragCoord, relight, environmentEquirect));
        }
    }

    if (relight.enabled == 0 || coverage <= 0.0001h) {
        return showBg ? half4(bg, 1.0h) : v.color;
    }
    // Pixels with no camera-facing material (mw ~ 0) have no usable surface normal -> show the
    // blended SH color (premultiplied) instead of normalizing a near-zero vector into noise.
    if (mw <= 0.0001h) {
        return showBg ? half4(v.color.rgb + bg * (1.0h - coverage), 1.0h) : v.color;
    }
    float invCov = 1.0f / float(coverage);
    float invMW = 1.0f / float(mw);
    half3 baseColor = half3(float3(v.color.rgb) * invCov);   // SH-evaluated diffuse base color
    half3 oriColor = half3(float3(v.oriColor.rgb) * invMW);  // Ref-Gaussian albedo / specular F0 tint
    float3 N = normalize(float3(v.normalRough.rgb));
    float3 V = normalize(float3(v.viewRefl.rgb));
    float roughness = float(v.normalRough.a) * invMW;
    float reflStrength = float(v.viewRefl.a) * invMW;
    half3 shaded = shadeIBLDeferred(baseColor, oriColor, N, V, roughness, reflStrength,
                                    relight, prefilteredEnv, irradianceEnv, brdfLUT, iblSampler);
    if (relight.debugMode != 0) {
        return half4(shaded * coverage, coverage);   // debug channels: premultiplied, transparent bg
    }
    // Shaded splats composited over the background (skybox or AR camera; opaque result).
    return half4(shaded * coverage + bg * (1.0h - coverage), 1.0h);
}

/// Generate a single triangle covering the entire screen
vertex FragmentIn postprocessVertexShader(uint vertexID [[vertex_id]]) {
    FragmentIn out;

    float4 position;
    position.x = (vertexID == 2) ? 3.0 : -1.0;
    position.y = (vertexID == 0) ? -3.0 : 1.0;
    position.zw = 1.0;

    out.position = position;
    return out;
}

fragment FragmentOut postprocessFragmentShader(FragmentValues fragmentValues [[imageblock_data]],
                                               float4 fragCoord [[position]],
                                               constant RelightUniforms & relight [[ buffer(0) ]],
                                               texturecube<float> prefilteredEnv [[ texture(0) ]],
                                               texture2d<float> brdfLUT [[ texture(1) ]],
                                               texturecube<float> irradianceEnv [[ texture(2) ]],
                                               texture2d<float> environmentEquirect [[ texture(3) ]],
                                               texture2d<float> arCameraImage [[ texture(4) ]],
                                               sampler iblSampler [[ sampler(0) ]]) {
    FragmentOut out;
    out.depth = (fragmentValues.color.a == 0) ? 0 : fragmentValues.depth / fragmentValues.color.a;
    out.color = resolveRelight(fragmentValues, fragCoord, relight, prefilteredEnv, irradianceEnv, brdfLUT, environmentEquirect, arCameraImage, iblSampler);
    return out;
}

fragment half4 postprocessFragmentShaderNoDepth(FragmentValues fragmentValues [[imageblock_data]],
                                                float4 fragCoord [[position]],
                                                constant RelightUniforms & relight [[ buffer(0) ]],
                                                texturecube<float> prefilteredEnv [[ texture(0) ]],
                                                texture2d<float> brdfLUT [[ texture(1) ]],
                                                texturecube<float> irradianceEnv [[ texture(2) ]],
                                                texture2d<float> environmentEquirect [[ texture(3) ]],
                                                texture2d<float> arCameraImage [[ texture(4) ]],
                                                sampler iblSampler [[ sampler(0) ]]) {
    return resolveRelight(fragmentValues, fragCoord, relight, prefilteredEnv, irradianceEnv, brdfLUT, environmentEquirect, arCameraImage, iblSampler);
}
