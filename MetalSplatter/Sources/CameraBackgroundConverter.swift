import Foundation
import simd
@preconcurrency import Metal

/// Converts an ARKit captured frame (biplanar YCbCr: luma + chroma) into a linear-RGB image that the
/// splat renderer composites over in AR mode (`SplatRenderer.arBackgroundTexture`). The conversion
/// runs the `arCameraVertexShader` / `arCameraBackground` shaders in PostProcess.metal, applying the
/// frame's display transform so the camera fills the drawable at the correct orientation and crop.
public final class CameraBackgroundConverter: @unchecked Sendable {
    public enum Error: Swift.Error {
        case functionNotFound(String)
    }

    private let pipelineState: MTLRenderPipelineState
    /// 1×1 zero-depth texture bound when no LiDAR scene depth is available (its value is ignored
    /// downstream because occlusion is disabled in that case).
    private let dummyDepth: MTLTexture

    public init(device: MTLDevice, colorFormat: MTLPixelFormat = .rgba16Float) throws {
        let library = try device.makeDefaultLibrary(bundle: Bundle.module)
        guard let vertexFunction = library.makeFunction(name: "arCameraVertexShader") else {
            throw Error.functionNotFound("arCameraVertexShader")
        }
        guard let fragmentFunction = library.makeFunction(name: "arCameraBackground") else {
            throw Error.functionNotFound("arCameraBackground")
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.colorAttachments[0].pixelFormat = colorFormat
        self.pipelineState = try device.makeRenderPipelineState(descriptor: descriptor)

        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float,
                                                                       width: 1, height: 1, mipmapped: false)
        depthDescriptor.usage = .shaderRead
        depthDescriptor.storageMode = .shared
        guard let dummy = device.makeTexture(descriptor: depthDescriptor) else { throw Error.functionNotFound("dummyDepth") }
        var zero: Float = 0
        dummy.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &zero, bytesPerRow: 4)
        self.dummyDepth = dummy
    }

    /// Encodes the conversion into `destination`. `luma`/`chroma` are the captured frame's two planes
    /// (R8 / RG8); `displayToCamera` maps a top-left-origin viewport uv to the captured-image uv
    /// (i.e. `frame.displayTransform(...).inverted()` as a homogeneous 3x3). `sceneDepth` (optional
    /// LiDAR depth in meters, camera-image space) is sampled and packed into the destination alpha.
    public func encode(into commandBuffer: MTLCommandBuffer,
                       luma: MTLTexture,
                       chroma: MTLTexture,
                       sceneDepth: MTLTexture?,
                       displayToCamera: simd_float3x3,
                       destination: MTLTexture) {
        let renderPass = MTLRenderPassDescriptor()
        renderPass.colorAttachments[0].texture = destination
        renderPass.colorAttachments[0].loadAction = .dontCare
        renderPass.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPass) else { return }
        encoder.label = "AR Camera Background"
        encoder.setRenderPipelineState(pipelineState)
        encoder.setFragmentTexture(luma, index: 0)
        encoder.setFragmentTexture(chroma, index: 1)
        encoder.setFragmentTexture(sceneDepth ?? dummyDepth, index: 2)
        var uniforms = ARCameraUniforms(displayToCamera: displayToCamera)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<ARCameraUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }
}

/// Matches `ARCameraUniforms` in PostProcess.metal (a single column-major 3x3).
private struct ARCameraUniforms {
    var displayToCamera: simd_float3x3
}
