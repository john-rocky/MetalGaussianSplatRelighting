#include "SplatProcessing.h"

vertex FragmentIn singleStageSplatVertexShader(uint vertexID [[vertex_id]],
                                               uint instanceID [[instance_id]],
                                               ushort amplificationID [[amplification_id]],
                                               device const ChunkInfo* chunks [[ buffer(BufferIndexChunks) ]],
                                               constant ChunkedSplatIndex* splatIndexArray [[ buffer(BufferIndexSplatIndex) ]],
                                               constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                                               constant RelightUniforms & relight [[ buffer(BufferIndexRelight) ]],
                                               texturecube<float> prefilteredEnv [[ texture(0) ]],
                                               texture2d<float> brdfLUT [[ texture(1) ]],
                                               texturecube<float> irradianceEnv [[ texture(2) ]],
                                               sampler iblSampler [[ sampler(0) ]]) {
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

    if (relight.enabled != 0 && chunk.materials != nullptr) {
        SplatMaterial material = chunk.materials[idx.splatIndex];
        out.color.rgb = shadePBR(out.color.rgb, material,
                                 float3(splat.position), float3(uniforms.cameraPosition),
                                 relight, prefilteredEnv, irradianceEnv, brdfLUT, iblSampler);
    }

    return out;
}

fragment half4 singleStageSplatFragmentShader(FragmentIn in [[stage_in]]) {
    half alpha = splatFragmentAlpha(in.relativePosition, in.color.a);
    return half4(alpha * in.color.rgb, alpha);
}
