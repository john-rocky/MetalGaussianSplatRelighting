#include <metal_stdlib>
using namespace metal;

// MARK: - Presentation post-process
//
// The scene (splats + deferred relighting + skybox) is rendered into an offscreen *linear HDR*
// texture. This pass is the SINGLE, FINAL presentation step that maps that linear-HDR image to the
// drawable. Relighting stays untouched in linear HDR upstream; everything here is display-only.
//
// Design rules (must hold):
//  - `bypass` performs an exact copy. It is set for the "reference / linear" mode (numerically
//    comparable to the official Ref-Gaussian renderer) and for every debug view (normal / roughness
//    / reflection / albedo are DATA, not radiance, so they must never be tonemapped).
//  - Exposure -> bloom add -> tonemap is applied once, here, on the composited linear image.
//
// Keep CompositeUniforms in sync with Swift: PostProcessor.CompositeUniforms.

typedef struct {
    float exposure;          // linear multiplier (2^EV); 1.0 = neutral
    uint  tonemapOperator;   // 0 none/linear, 1 ACES, 2 Khronos PBR Neutral
    uint  bypass;            // 1 = copy source unchanged (reference mode or debug view)
    float bloomIntensity;    // 0 = no bloom add
    float maxValue;          // output ceiling: 1.0 SDR, > 1.0 for EDR headroom
    float bloomThreshold;    // linear-HDR luminance above which pixels contribute to bloom
    float _pad1;
    float _pad2;
} CompositeUniforms;

// Per-direction parameters for the separable bloom blur. Keep in sync with PostProcessor.BlurUniforms.
typedef struct {
    float2 texelStep;        // uv-space offset of one source texel along the blur axis
    float2 _pad;
} BlurUniforms;

constexpr sampler ppLinearSampler(coord::normalized, address::clamp_to_edge, filter::linear);

struct PPVertexOut {
    float4 position [[position]];
    float2 uv;
};

// Fullscreen triangle. uv is provided for sampler-based passes (bloom); the composite reads texels
// directly via [[position]] for an exact 1:1 mapping.
vertex PPVertexOut ppVertexShader(uint vertexID [[vertex_id]]) {
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    PPVertexOut out;
    out.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = float2(uv.x, 1.0 - uv.y);
    return out;
}

// ACES filmic tonemap (Narkowicz 2015 fit). Maps linear HDR -> linear display [0,1]; the drawable's
// sRGB format applies the OETF on write, so we return linear here (same as the bypass path).
static float3 tonemapACES(float3 x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// Khronos PBR Neutral tonemap (KhronosGroup, 2024). Hue-preserving, gentle highlight roll-off —
// keeps material albedo faithful, which suits a PBR/relighting demo. Returns linear display [0,1].
static float3 tonemapPBRNeutral(float3 color) {
    const float startCompression = 0.8f - 0.04f;
    const float desaturation = 0.15f;

    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08f ? x - 6.25f * x * x : 0.04f;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) {
        return color;
    }

    const float d = 1.0f - startCompression;
    float newPeak = 1.0f - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1.0f - 1.0f / (desaturation * (peak - newPeak) + 1.0f);
    return mix(color, float3(newPeak), g);
}

// MARK: Bloom (HDR bright-pass + separable Gaussian blur, added in linear HDR before tonemap)

// Bright-pass + downsample: keep the (exposed) energy above the luminance threshold, preserving hue.
fragment half4 ppBloomBright(PPVertexOut in [[stage_in]],
                             texture2d<float> source [[texture(0)]],
                             constant CompositeUniforms &u [[buffer(0)]]) {
    float3 c = max(source.sample(ppLinearSampler, in.uv).rgb * u.exposure, 0.0f);
    float luma = dot(c, float3(0.2126f, 0.7152f, 0.0722f));
    float contribution = max(luma - u.bloomThreshold, 0.0f) / max(luma, 1e-4f);
    return half4(half3(c * contribution), 1.0h);
}

// One separable Gaussian pass (9 taps). texelStep selects the horizontal or vertical direction.
fragment half4 ppBloomBlur(PPVertexOut in [[stage_in]],
                           texture2d<float> source [[texture(0)]],
                           constant BlurUniforms &b [[buffer(0)]]) {
    const float w0 = 0.227027f;
    const float w1 = 0.1945946f;
    const float w2 = 0.1216216f;
    const float w3 = 0.054054f;
    const float w4 = 0.016216f;
    float2 t = b.texelStep;
    float3 acc = source.sample(ppLinearSampler, in.uv).rgb * w0;
    acc += source.sample(ppLinearSampler, in.uv + t).rgb * w1;
    acc += source.sample(ppLinearSampler, in.uv - t).rgb * w1;
    acc += source.sample(ppLinearSampler, in.uv + 2.0f * t).rgb * w2;
    acc += source.sample(ppLinearSampler, in.uv - 2.0f * t).rgb * w2;
    acc += source.sample(ppLinearSampler, in.uv + 3.0f * t).rgb * w3;
    acc += source.sample(ppLinearSampler, in.uv - 3.0f * t).rgb * w3;
    acc += source.sample(ppLinearSampler, in.uv + 4.0f * t).rgb * w4;
    acc += source.sample(ppLinearSampler, in.uv - 4.0f * t).rgb * w4;
    return half4(half3(acc), 1.0h);
}

fragment half4 ppComposite(PPVertexOut in [[stage_in]],
                           texture2d<float> source [[texture(0)]],
                           texture2d<float> bloomTex [[texture(1)]],
                           constant CompositeUniforms &u [[buffer(0)]]) {
    float4 src = source.read(uint2(in.position.xy));

    // Reference / linear mode and all debug views: exact passthrough (no presentation).
    if (u.bypass != 0) {
        return half4(src);
    }

    // Presentation chain (linear HDR -> display), applied once on the composited image:
    //   exposure  ->  bloom add  ->  tonemap.
    float3 hdr = max(src.rgb * u.exposure, 0.0f);
    if (u.bloomIntensity > 0.0f) {
        // bloomTex is the blurred, exposure-scaled bright-pass; sampled (upscaled) and added linearly.
        hdr += bloomTex.sample(ppLinearSampler, in.uv).rgb * u.bloomIntensity;
    }

    // Tonemap with a retargetable white point: the brightest highlights map to u.maxValue (1.0 for
    // SDR; the EDR headroom when EDR output is on, so highlights extend past 1.0 instead of clamping).
    float w = max(u.maxValue, 1.0f);
    float3 mapped;
    if (u.tonemapOperator == 1u) {
        mapped = tonemapACES(hdr / w) * w;
    } else if (u.tonemapOperator == 2u) {
        mapped = tonemapPBRNeutral(hdr / w) * w;
    } else {
        mapped = hdr;   // linear: exposure only, no curve (already extends past 1.0)
    }
    return half4(half3(mapped), half(src.a));
}
