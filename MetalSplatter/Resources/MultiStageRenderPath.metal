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
    half3 indirect [[raster_order_group(0)]];      // Ref-Gaussian ASG indirect radiance * w (zero when ASG disabled / absent)
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
    values->indirect = { 0, 0, 0 };
    values->materialWeight = 0;
    values->depth = 0;
}

// Transpose of Ref-Gaussian's `rotation_between_z(N)` (utils/graphics_utils.py): applied to `v`,
// it expresses `v` in the local frame where the +Z axis aligns with `N`. The closed-form formula
// avoids any explicit basis construction (and the discontinuity that picking an arbitrary tangent
// would introduce); the only singular case is N pointing straight down, where the matrix becomes
// a 180° flip.
static float3 worldToLocalAlignedToZ(float3 N, float3 v) {
    if (N.z < -0.999999f) return -v;
    float inv = 1.0f / (N.z + 1.0f);
    float nxx = N.x * N.x * inv;
    float nyy = N.y * N.y * inv;
    float nxy = N.x * N.y * inv;
    float vx = (1.0f - nxx) * v.x + (-nxy) * v.y + (-N.x) * v.z;
    float vy = (-nxy) * v.x + (1.0f - nyy) * v.y + (-N.y) * v.z;
    float vz = N.x * v.x + N.y * v.y + N.z * v.z;
    return float3(vx, vy, vz);
}

// Evaluate Ref-Gaussian's 32-lobe ASG indirect term in the normal-aligned local frame. Each lobe
// k = i*8 + j has a principal direction ω, anisotropy axes ω_λ, ω_μ, and trained per-channel
// params (ep_rgb, λ, μ). See utils/graphics_utils.py `init_predefined_omega(4, 8)` for the lobe
// layout and gaussian_renderer/__init__.py:284-295 for the math.
//
//   ω      = (cos φ sin θ, sin φ sin θ, cos θ)
//   ω_λ    = (cos φ cos θ, sin φ cos θ, -sin θ)        (= ω rotated +π/2 in θ)
//   ω_μ    = (-sin φ, cos φ, 0)                        (= ω × ω_λ)
//   ep_c   = exp(asg[ch] - 3)                         (per-channel RGB intensity)
//   λ      = softplus(asg[3] - 1), μ = softplus(asg[4] - 1)
//   lobe_k = ep * relu(ω · R) * exp(-λ (ω_λ · R)² - μ (ω_μ · R)²)
//   I      = max(Σ lobe_k, 0)
static half3 evaluateASG(device const half* asg, float3 reflLocal) {
    float3 indirect = float3(0.0f);
    for (uint i = 0; i < 4; i++) {
        float theta = (float(i) + 0.5f) * (M_PI_F / 8.0f);
        float sinT, cosT;
        sinT = sincos(theta, cosT);
        for (uint j = 0; j < 8; j++) {
            float phi = (float(j) + 0.5f) * (M_PI_F / 4.0f);
            float sinP, cosP;
            sinP = sincos(phi, cosP);

            float3 omega   = float3(cosP * sinT, sinP * sinT, cosT);
            float3 omegaLa = float3(cosP * cosT, sinP * cosT, -sinT);
            float3 omegaMu = float3(-sinP, cosP, 0.0f);

            uint k = i * 8u + j;
            device const half* base = asg + 5u * k;
            float3 ep = float3(exp(float(base[0]) - 3.0f),
                               exp(float(base[1]) - 3.0f),
                               exp(float(base[2]) - 3.0f));
            float la = log(1.0f + exp(float(base[3]) - 1.0f));
            float mu = log(1.0f + exp(float(base[4]) - 1.0f));

            float smoothK = max(dot(omega, reflLocal), 0.0f);
            float dotLa = dot(omegaLa, reflLocal);
            float dotMu = dot(omegaMu, reflLocal);
            float gain = smoothK * exp(-la * dotLa * dotLa - mu * dotMu * dotMu);
            indirect += ep * gain;
        }
    }
    return half3(max(indirect, 0.0f));
}

vertex FragmentIn multiStageSplatVertexShader(uint vertexID [[vertex_id]],
                                              uint instanceID [[instance_id]],
                                              ushort amplificationID [[amplification_id]],
                                              device const ChunkInfo* chunks [[ buffer(BufferIndexChunks) ]],
                                              constant ChunkedSplatIndex* splatIndexArray [[ buffer(BufferIndexSplatIndex) ]],
                                              constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                                              constant RelightUniforms & relight [[ buffer(BufferIndexRelight) ]]) {
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
    half3 gIndirect = half3(0);
    if (chunk.materials != nullptr) {
        SplatMaterial material = chunk.materials[idx.splatIndex];
        float3 N = normalize(float3(material.normal));
        float3 V = normalize(float3(uniforms.cameraPosition) - float3(splat.position));
        out.gNormal = half3(N);
        out.gView = half3(V);
        out.gMaterial = half2(material.roughness, material.reflectionStrength);
        out.gOriColor = half3(material.specularTint);

        // Ref-Gaussian ASG indirect: rotate the world-space reflection vector into the splat's
        // normal-aligned local frame (same convention as `rotation_between_z(N)^T` in the paper)
        // and evaluate the 32-lobe ASG sum. Skip when the toggle is off OR the chunk wasn't
        // trained with the ASG tail — N flipping below uses the same faceforward as the normal
        // composite to stay consistent with the material weight downstream.
        if (relight.asgEnabled != 0u && chunk.asg != nullptr) {
            float facing = (dot(N, V) < 0.0f) ? -1.0f : 1.0f;
            float3 Nf = N * facing;
            float3 reflW = reflect(-V, Nf);
            float3 reflLocal = worldToLocalAlignedToZ(Nf, reflW);
            gIndirect = evaluateASG(chunk.asg + 160u * idx.splatIndex, reflLocal);
        }
    } else {
        out.gNormal = half3(0, 0, 1);
        out.gView = half3(0, 0, 1);
        out.gMaterial = half2(0.5h, 0.0h);
        out.gOriColor = half3(1, 1, 1);
    }
    out.gIndirect = gIndirect;

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
    out.values.indirect = previousFragmentValues.indirect * oneMinusAlpha + in.gIndirect * w;
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

// Configurator repaint of one color. For a CHROMATIC target paint we change only the hue (keeping
// each pixel's brightness and chroma magnitude) and gate by relative saturation, so achromatic detail
// — white/gray/black markings, glass, tyres, chrome — is left untouched; only the colored body is
// repainted. For a NEUTRAL target (black/white/silver) we fall back to a value repaint that affects
// everything (painting the whole car that shade). `tint` is the paint (linear RGB), `strength` 0...1.
static float3 recolorPreservingValue(float3 c, float3 tint, float3 lw, float strength) {
    float lum = dot(c, lw);
    float3 chroma = c - lum;
    float sat = length(chroma);
    float relSat = sat / max(lum, 1e-3f);                 // ~ HSV saturation (brightness-independent)

    float tLum = dot(tint, lw);
    float3 tChroma = tint - tLum;
    float tSat = length(tChroma);
    float3 tDir = tSat > 1e-4f ? tChroma / tSat : float3(0.0f);

    float3 neutralRecolor = lum * tint;                    // value repaint (black/white/silver)
    float3 chromaticRecolor = max(lum + tDir * sat, 0.0f); // hue repaint; preserves achromatic detail
    float targetChromaticity = smoothstep(0.02f, 0.08f, tSat);
    float3 recolored = mix(neutralRecolor, chromaticRecolor, targetChromaticity);

    float bodyGate = smoothstep(0.10f, 0.35f, relSat);     // chromatic repaint: only the colored body
    float gate = mix(1.0f, bodyGate, targetChromaticity);
    return mix(c, recolored, strength * gate);
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
    float sceneDepthM = -1.0f;   // AR scene depth (meters) at this pixel, packed in the camera alpha
    if (showBg) {
        if (relight.arBackground != 0) {
            float2 uv = fragCoord.xy / max(relight.screenSize, float2(1.0f));
            float4 cam = arCameraImage.sample(arBgSampler, uv);
            bg = half3(cam.rgb);
            sceneDepthM = cam.a;
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
    // Configurator repaint: change the body's hue to the chosen paint while preserving the shading and
    // leaving achromatic detail (white/gray/black markings) untouched. See recolorPreservingValue.
    if (relight.tint.w > 0.0f) {
        const float3 lw = float3(0.2126f, 0.7152f, 0.0722f);
        baseColor = half3(recolorPreservingValue(float3(baseColor), relight.tint.xyz, lw, relight.tint.w));
        oriColor  = half3(recolorPreservingValue(float3(oriColor),  relight.tint.xyz, lw, relight.tint.w));
    }
    float3 N = normalize(float3(v.normalRough.rgb));
    float3 V = normalize(float3(v.viewRefl.rgb));
    float roughness = float(v.normalRough.a) * invMW;
    float reflStrength = float(v.viewRefl.a) * invMW;
    // ASG indirect is weighted-summed in the imageblock the same way as the other material
    // channels; recover the per-pixel mean by dividing by materialWeight. Zero when the chunk
    // carries no ASG tail (vertex shader emits gIndirect = 0 in that case).
    half3 indirectBlended = half3(float3(v.indirect) * invMW);
    half3 shaded = shadeIBLDeferred(baseColor, oriColor, indirectBlended, N, V, roughness, reflStrength,
                                    relight, prefilteredEnv, irradianceEnv, brdfLUT, iblSampler);
    if (relight.debugMode != 0) {
        return half4(shaded * coverage, coverage);   // debug channels: premultiplied, transparent bg
    }
    // Depth occlusion: where the real world (AR scene depth) is in front of this splat pixel, suppress
    // the splat so the camera shows through. Splat NDC depth -> meters via depthLinearize, then compare.
    half effectiveCoverage = coverage;
    if (relight.occlusionEnabled != 0u && sceneDepthM > 0.01f) {
        float splatNDC = float(v.depth) / float(coverage);
        float splatDepthM = relight.depthLinearize.y / max(relight.depthLinearize.x - splatNDC, 1e-4f);
        float occlusion = smoothstep(0.0f, 0.05f, splatDepthM - sceneDepthM);   // 1 = real world in front
        effectiveCoverage = coverage * half(1.0f - occlusion);
    }
    // Shaded splats composited over the background (skybox or AR camera; opaque result).
    return half4(shaded * effectiveCoverage + bg * (1.0h - effectiveCoverage), 1.0h);
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
