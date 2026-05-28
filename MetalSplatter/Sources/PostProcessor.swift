import Foundation
@preconcurrency import Metal

/// Final presentation pass for the relightable renderer.
///
/// The scene is rendered into an offscreen *linear HDR* texture (relighting completes there,
/// untouched). `PostProcessor` is the single, separate step that maps that linear-HDR image to the
/// drawable: exposure, bloom, and filmic tonemapping — all display-only. A "reference" mode (and any
/// debug view) bypasses everything for an exact copy, so the output stays numerically comparable to
/// the official Ref-Gaussian renderer.
///
/// Pipelines are keyed by destination pixel format so the same instance serves both the SDR drawable
/// (`bgra8Unorm_srgb`) and the EDR drawable (`rgba16Float`).
public final class PostProcessor {
    /// Filmic tonemap operator applied to the composited linear-HDR image (shaded mode only).
    public enum Tonemap: Int, Sendable, CaseIterable, Identifiable {
        case none = 0               // linear (reference)
        case aces = 1               // ACES filmic (fitted)
        case khronosPBRNeutral = 2  // Khronos PBR Neutral (hue-preserving)
        public var id: Int { rawValue }
        public var displayName: String {
            switch self {
            case .none: return "Linear"
            case .aces: return "ACES"
            case .khronosPBRNeutral: return "Neutral"
            }
        }
    }

    /// Display-only presentation settings. Mutate between frames.
    public struct Settings: Sendable {
        /// When true, bypass all presentation so the output matches the official renderer numerically.
        public var referenceMode: Bool = true
        /// Linear exposure multiplier (e.g. 2^EV).
        public var exposure: Float = 1.0
        public var tonemap: Tonemap = .none
        public var bloomEnabled: Bool = false
        /// Linear-HDR luminance above which pixels contribute to bloom.
        public var bloomThreshold: Float = 1.0
        /// How strongly the blurred bloom is added back to the linear image.
        public var bloomIntensity: Float = 0.0
        /// Output ceiling: 1.0 for SDR; the display's EDR headroom (> 1.0) when EDR is enabled.
        public var maxValue: Float = 1.0
        public init() {}
    }

    // Keep in sync with PostProcess.metal : CompositeUniforms.
    private struct CompositeUniforms {
        var exposure: Float
        var tonemapOperator: UInt32
        var bypass: UInt32
        var bloomIntensity: Float
        var maxValue: Float
        var bloomThreshold: Float = 0
        var _pad1: Float = 0
        var _pad2: Float = 0
    }

    // Keep in sync with PostProcess.metal : BlurUniforms.
    private struct BlurUniforms {
        var texelStep: SIMD2<Float>
        var _pad: SIMD2<Float> = .zero
    }

    private let device: MTLDevice
    private let library: MTLLibrary
    private var compositePipelines: [MTLPixelFormat: MTLRenderPipelineState] = [:]

    // Bloom: bright-pass + separable Gaussian blur, kept at half resolution in linear HDR.
    private static let bloomFormat: MTLPixelFormat = .rgba16Float
    private var bloomBrightPipeline: MTLRenderPipelineState?
    private var bloomBlurPipeline: MTLRenderPipelineState?
    private var bloomTextureA: MTLTexture?   // bright-pass target, then the final blurred bloom
    private var bloomTextureB: MTLTexture?   // separable-blur ping-pong
    private var bloomWidth = 0
    private var bloomHeight = 0
    /// Bound to the composite's bloom slot whenever bloom is off, so the binding is always valid.
    private let dummyTexture: MTLTexture

    public init(device: MTLDevice) throws {
        self.device = device
        self.library = try device.makeDefaultLibrary(bundle: Bundle.module)

        let dummyDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float,
                                                                       width: 1, height: 1, mipmapped: false)
        dummyDescriptor.usage = .shaderRead
        dummyDescriptor.storageMode = .private
        guard let dummy = device.makeTexture(descriptor: dummyDescriptor) else {
            throw PostProcessorError.textureCreationFailed
        }
        self.dummyTexture = dummy
    }

    private func compositePipeline(for format: MTLPixelFormat) throws -> MTLRenderPipelineState {
        if let pipeline = compositePipelines[format] { return pipeline }
        guard let vertexFunction = library.makeFunction(name: "ppVertexShader"),
              let fragmentFunction = library.makeFunction(name: "ppComposite") else {
            throw PostProcessorError.missingFunction
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "PostProcess Composite"
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.colorAttachments[0].pixelFormat = format
        let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        compositePipelines[format] = pipeline
        return pipeline
    }

    private func bloomPipelines() throws -> (bright: MTLRenderPipelineState, blur: MTLRenderPipelineState) {
        if let bright = bloomBrightPipeline, let blur = bloomBlurPipeline { return (bright, blur) }
        guard let vertexFunction = library.makeFunction(name: "ppVertexShader"),
              let brightFunction = library.makeFunction(name: "ppBloomBright"),
              let blurFunction = library.makeFunction(name: "ppBloomBlur") else {
            throw PostProcessorError.missingFunction
        }
        let brightDescriptor = MTLRenderPipelineDescriptor()
        brightDescriptor.label = "PostProcess Bloom Bright"
        brightDescriptor.vertexFunction = vertexFunction
        brightDescriptor.fragmentFunction = brightFunction
        brightDescriptor.colorAttachments[0].pixelFormat = Self.bloomFormat
        let bright = try device.makeRenderPipelineState(descriptor: brightDescriptor)

        let blurDescriptor = MTLRenderPipelineDescriptor()
        blurDescriptor.label = "PostProcess Bloom Blur"
        blurDescriptor.vertexFunction = vertexFunction
        blurDescriptor.fragmentFunction = blurFunction
        blurDescriptor.colorAttachments[0].pixelFormat = Self.bloomFormat
        let blur = try device.makeRenderPipelineState(descriptor: blurDescriptor)

        bloomBrightPipeline = bright
        bloomBlurPipeline = blur
        return (bright, blur)
    }

    private func ensureBloomTextures(width: Int, height: Int) {
        if bloomTextureA != nil, bloomWidth == width, bloomHeight == height { return }
        guard width > 0, height > 0 else { return }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: Self.bloomFormat,
                                                                  width: width, height: height, mipmapped: false)
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .private
        bloomTextureA = device.makeTexture(descriptor: descriptor)
        bloomTextureA?.label = "Bloom A"
        bloomTextureB = device.makeTexture(descriptor: descriptor)
        bloomTextureB?.label = "Bloom B"
        bloomWidth = width
        bloomHeight = height
    }

    private func encodeFullscreen(_ commandBuffer: MTLCommandBuffer,
                                  pipeline: MTLRenderPipelineState,
                                  destination: MTLTexture,
                                  _ configure: (MTLRenderCommandEncoder) -> Void) {
        let descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = destination
        descriptor.colorAttachments[0].loadAction = .dontCare
        descriptor.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }
        encoder.setRenderPipelineState(pipeline)
        configure(encoder)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    /// Runs the bloom chain (bright-pass + separable blur) at half resolution and returns the blurred
    /// result (exposure-scaled), or nil if bloom is unavailable.
    private func encodeBloom(into commandBuffer: MTLCommandBuffer,
                             source: MTLTexture,
                             uniforms: CompositeUniforms) -> MTLTexture? {
        let halfWidth = max(source.width / 2, 1)
        let halfHeight = max(source.height / 2, 1)
        ensureBloomTextures(width: halfWidth, height: halfHeight)
        guard let textureA = bloomTextureA, let textureB = bloomTextureB,
              let pipelines = try? bloomPipelines() else {
            return nil
        }

        // Bright-pass + downsample: source -> A.
        var brightUniforms = uniforms
        encodeFullscreen(commandBuffer, pipeline: pipelines.bright, destination: textureA) { encoder in
            encoder.setFragmentTexture(source, index: 0)
            encoder.setFragmentBytes(&brightUniforms, length: MemoryLayout<CompositeUniforms>.stride, index: 0)
        }
        // Horizontal blur: A -> B.
        var horizontal = BlurUniforms(texelStep: SIMD2<Float>(1.0 / Float(halfWidth), 0))
        encodeFullscreen(commandBuffer, pipeline: pipelines.blur, destination: textureB) { encoder in
            encoder.setFragmentTexture(textureA, index: 0)
            encoder.setFragmentBytes(&horizontal, length: MemoryLayout<BlurUniforms>.stride, index: 0)
        }
        // Vertical blur: B -> A.
        var vertical = BlurUniforms(texelStep: SIMD2<Float>(0, 1.0 / Float(halfHeight)))
        encodeFullscreen(commandBuffer, pipeline: pipelines.blur, destination: textureA) { encoder in
            encoder.setFragmentTexture(textureB, index: 0)
            encoder.setFragmentBytes(&vertical, length: MemoryLayout<BlurUniforms>.stride, index: 0)
        }
        return textureA
    }

    /// Encodes the presentation pass: `source` (linear HDR) -> `destination` (drawable).
    /// - Parameter debugMode: the renderer's debug mode; any non-zero value forces a passthrough
    ///   because debug channels are data, not radiance.
    public func encode(into commandBuffer: MTLCommandBuffer,
                       source: MTLTexture,
                       destination: MTLTexture,
                       debugMode: UInt32,
                       settings: Settings) {
        let bypass = settings.referenceMode || debugMode != 0
        let bloomActive = settings.bloomEnabled && settings.bloomIntensity > 0 && !bypass
        var uniforms = CompositeUniforms(exposure: settings.exposure,
                                         tonemapOperator: UInt32(settings.tonemap.rawValue),
                                         bypass: bypass ? 1 : 0,
                                         bloomIntensity: bloomActive ? settings.bloomIntensity : 0,
                                         maxValue: settings.maxValue,
                                         bloomThreshold: settings.bloomThreshold)

        // Bloom runs first (its own passes) so its result is ready for the composite. Skipped under
        // bypass so reference / debug output stays an exact copy.
        let bloomResult = bloomActive ? encodeBloom(into: commandBuffer, source: source, uniforms: uniforms) : nil

        let descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = destination
        descriptor.colorAttachments[0].loadAction = .dontCare
        descriptor.colorAttachments[0].storeAction = .store

        guard let pipeline = try? compositePipeline(for: destination.pixelFormat),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else {
            return
        }
        encoder.label = "PostProcess Composite"
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(source, index: 0)
        encoder.setFragmentTexture(bloomResult ?? dummyTexture, index: 1)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<CompositeUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }
}

enum PostProcessorError: Error {
    case missingFunction
    case textureCreationFailed
}
