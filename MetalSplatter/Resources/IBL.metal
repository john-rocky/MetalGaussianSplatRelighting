#include <metal_stdlib>
using namespace metal;

// MARK: - Image-Based Lighting precompute kernels
//
// These build the split-sum IBL inputs from an equirectangular HDR environment:
//   1. iblEquirectToCubemap  - lat/long HDR -> base radiance cubemap
//   2. iblPrefilterEnvmap    - GGX-prefiltered specular cubemap (one dispatch per roughness mip)
//   3. iblComputeIrradiance  - cosine-convolved diffuse irradiance cubemap
//   4. iblIntegrateBRDF      - environment BRDF (DFG) 2D LUT, indexed by (NdotV, roughness)
//
// Direction convention is shared with the runtime sampling in SplatProcessing.metal so the
// generated maps are self-consistent (env orientation can be adjusted with the rotation uniform).

namespace ibl {

constant float PI = 3.14159265358979323846f;

// Map a cubemap face + face-local uv in [0,1] to a world-space direction.
inline float3 cubeDirection(uint face, float2 uv) {
    float2 st = uv * 2.0f - 1.0f; // [-1, 1]
    float3 dir;
    switch (face) {
        case 0: dir = float3( 1.0f, -st.y, -st.x); break; // +X
        case 1: dir = float3(-1.0f, -st.y,  st.x); break; // -X
        case 2: dir = float3( st.x,  1.0f,  st.y); break; // +Y
        case 3: dir = float3( st.x, -1.0f, -st.y); break; // -Y
        case 4: dir = float3( st.x, -st.y,  1.0f); break; // +Z
        default: dir = float3(-st.x, -st.y, -1.0f); break; // -Z
    }
    return normalize(dir);
}

// Direction -> equirectangular (lat/long) uv. Y is up.
inline float2 dirToEquirectUV(float3 d) {
    float u = atan2(d.z, d.x) / (2.0f * PI) + 0.5f;
    float v = acos(clamp(d.y, -1.0f, 1.0f)) / PI;
    return float2(u, v);
}

// Van der Corput radical inverse + Hammersley low-discrepancy sequence.
inline float radicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;
}
inline float2 hammersley(uint i, uint n) {
    return float2(float(i) / float(n), radicalInverseVdC(i));
}

// GGX importance sample: returns a half-vector around N for the given roughness.
inline float3 importanceSampleGGX(float2 xi, float3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0f * PI * xi.x;
    float cosTheta = sqrt((1.0f - xi.y) / (1.0f + (a * a - 1.0f) * xi.y));
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float3 H = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    float3 up = abs(N.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 tangent = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);
    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

inline float geometrySchlickGGX(float NdotV, float roughness) {
    float a = roughness;
    float k = (a * a) / 2.0f;
    return NdotV / (NdotV * (1.0f - k) + k);
}
inline float geometrySmith(float NdotV, float NdotL, float roughness) {
    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

constexpr sampler equirectSampler(coord::normalized, address::repeat, filter::linear);
constexpr sampler cubeSampler(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::linear);

} // namespace ibl

// MARK: - Kernels

kernel void iblEquirectToCubemap(texture2d<float, access::sample> equirect [[texture(0)]],
                                 texturecube<float, access::write> cube [[texture(1)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    uint size = cube.get_width();
    if (gid.x >= size || gid.y >= size) { return; }
    float2 uv = (float2(gid.xy) + 0.5f) / float(size);
    float3 dir = ibl::cubeDirection(gid.z, uv);
    float4 color = equirect.sample(ibl::equirectSampler, ibl::dirToEquirectUV(dir));
    cube.write(color, gid.xy, gid.z);
}

// Bind one mip slice of the output cube per dispatch (as its own texture view), and pass
// that mip's roughness. envCube should have a full mip chain for variance reduction.
kernel void iblPrefilterEnvmap(texturecube<float, access::sample> envCube [[texture(0)]],
                               texturecube<float, access::write> outMip [[texture(1)]],
                               constant float &roughness [[buffer(0)]],
                               uint3 gid [[thread_position_in_grid]]) {
    uint size = outMip.get_width();
    if (gid.x >= size || gid.y >= size) { return; }
    // Split-sum approximation: assume the view and reflection directions equal the normal.
    float3 N = ibl::cubeDirection(gid.z, (float2(gid.xy) + 0.5f) / float(size));
    float3 V = N;

    constexpr uint kSampleCount = 256u;
    float3 prefiltered = float3(0.0f);
    float totalWeight = 0.0f;
    for (uint i = 0u; i < kSampleCount; ++i) {
        float2 xi = ibl::hammersley(i, kSampleCount);
        float3 H = ibl::importanceSampleGGX(xi, N, roughness);
        float3 L = normalize(2.0f * dot(V, H) * H - V);
        float NdotL = max(dot(N, L), 0.0f);
        if (NdotL > 0.0f) {
            prefiltered += envCube.sample(ibl::cubeSampler, L).rgb * NdotL;
            totalWeight += NdotL;
        }
    }
    prefiltered /= max(totalWeight, 1e-4f);
    outMip.write(float4(prefiltered, 1.0f), gid.xy, gid.z);
}

kernel void iblComputeIrradiance(texturecube<float, access::sample> envCube [[texture(0)]],
                                 texturecube<float, access::write> irradiance [[texture(1)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    uint size = irradiance.get_width();
    if (gid.x >= size || gid.y >= size) { return; }
    float3 N = ibl::cubeDirection(gid.z, (float2(gid.xy) + 0.5f) / float(size));

    float3 up = abs(N.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(0.0f, 0.0f, 1.0f);
    float3 right = normalize(cross(up, N));
    up = cross(N, right);

    float3 irr = float3(0.0f);
    float nrSamples = 0.0f;
    const float delta = 0.025f;
    for (float phi = 0.0f; phi < 2.0f * ibl::PI; phi += delta) {
        for (float theta = 0.0f; theta < 0.5f * ibl::PI; theta += delta) {
            float3 tangentSample = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            float3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * N;
            irr += envCube.sample(ibl::cubeSampler, sampleVec).rgb * cos(theta) * sin(theta);
            nrSamples += 1.0f;
        }
    }
    irr = ibl::PI * irr / max(nrSamples, 1.0f);
    irradiance.write(float4(irr, 1.0f), gid.xy, gid.z);
}

kernel void iblIntegrateBRDF(texture2d<float, access::write> lut [[texture(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint w = lut.get_width();
    uint h = lut.get_height();
    if (gid.x >= w || gid.y >= h) { return; }

    float NdotV = (float(gid.x) + 0.5f) / float(w);
    float roughness = (float(gid.y) + 0.5f) / float(h);
    float3 V = float3(sqrt(max(0.0f, 1.0f - NdotV * NdotV)), 0.0f, NdotV);
    float3 N = float3(0.0f, 0.0f, 1.0f);

    float A = 0.0f;
    float B = 0.0f;
    constexpr uint kSampleCount = 1024u;
    for (uint i = 0u; i < kSampleCount; ++i) {
        float2 xi = ibl::hammersley(i, kSampleCount);
        float3 H = ibl::importanceSampleGGX(xi, N, roughness);
        float3 L = normalize(2.0f * dot(V, H) * H - V);
        float NdotL = max(L.z, 0.0f);
        float NdotH = max(H.z, 0.0f);
        float VdotH = max(dot(V, H), 0.0f);
        if (NdotL > 0.0f) {
            float G = ibl::geometrySmith(NdotV, NdotL, roughness);
            float GVis = (G * VdotH) / max(NdotH * NdotV, 1e-4f);
            float Fc = pow(1.0f - VdotH, 5.0f);
            A += (1.0f - Fc) * GVis;
            B += Fc * GVis;
        }
    }
    lut.write(float4(A / float(kSampleCount), B / float(kSampleCount), 0.0f, 1.0f), gid);
}
