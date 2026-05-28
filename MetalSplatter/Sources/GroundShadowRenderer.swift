import Foundation
import simd
@preconcurrency import Metal

/// Draws a soft elliptical contact shadow on the floor under an AR-placed model, so it reads as
/// resting on the ground rather than floating. The shadow is a unit quad on the y=0 plane,
/// transformed by an MVP and alpha-blended (darkening) onto the camera-background texture before the
/// splats composite over it. See the `groundShadow*` shaders in PostProcess.metal.
public final class GroundShadowRenderer: @unchecked Sendable {
    public enum Error: Swift.Error {
        case functionNotFound(String)
    }

    private let pipelineState: MTLRenderPipelineState

    public init(device: MTLDevice, colorFormat: MTLPixelFormat = .rgba16Float) throws {
        let library = try device.makeDefaultLibrary(bundle: Bundle.module)
        guard let vertexFunction = library.makeFunction(name: "groundShadowVertex") else {
            throw Error.functionNotFound("groundShadowVertex")
        }
        guard let fragmentFunction = library.makeFunction(name: "groundShadowFragment") else {
            throw Error.functionNotFound("groundShadowFragment")
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        let attachment = descriptor.colorAttachments[0]!
        attachment.pixelFormat = colorFormat
        // Standard "over" blend with black source -> dst * (1 - srcAlpha): darkens the floor.
        attachment.isBlendingEnabled = true
        attachment.rgbBlendOperation = .add
        attachment.alphaBlendOperation = .add
        attachment.sourceRGBBlendFactor = .sourceAlpha
        attachment.sourceAlphaBlendFactor = .sourceAlpha
        attachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
        attachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        self.pipelineState = try device.makeRenderPipelineState(descriptor: descriptor)
    }

    /// Encodes the shadow into `destination` (the camera background). `mvp` maps the floor quad
    /// (a unit XZ quad) to clip space; `strength` is the peak darkness and `softness` the falloff.
    public func encode(into commandBuffer: MTLCommandBuffer,
                       mvp: simd_float4x4,
                       strength: Float,
                       softness: Float,
                       destination: MTLTexture) {
        let renderPass = MTLRenderPassDescriptor()
        renderPass.colorAttachments[0].texture = destination
        renderPass.colorAttachments[0].loadAction = .load    // preserve the camera background
        renderPass.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPass) else { return }
        encoder.label = "AR Ground Shadow"
        encoder.setRenderPipelineState(pipelineState)
        var uniforms = GroundShadowUniforms(mvp: mvp, strength: strength, softness: softness)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<GroundShadowUniforms>.stride, index: 0)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<GroundShadowUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
    }
}

/// Matches `GroundShadowUniforms` in PostProcess.metal.
private struct GroundShadowUniforms {
    var mvp: simd_float4x4
    var strength: Float
    var softness: Float
    var pad: SIMD2<Float> = .zero
}
