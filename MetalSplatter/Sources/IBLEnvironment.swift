import Foundation
@preconcurrency import Metal

/// Builds and owns the split-sum image-based-lighting (IBL) inputs used for relightable rendering:
/// a GGX-prefiltered specular cubemap (roughness increasing with mip level), a diffuse irradiance
/// cubemap, and the environment BRDF (DFG) integration LUT.
///
/// Construct from an equirectangular (lat/long) environment texture — e.g. an HDR panorama loaded
/// via `MTKTextureLoader`. The maps are built once, on the GPU, at init.
public final class IBLEnvironment: @unchecked Sendable {
    public enum Constants {
        public static let baseCubeSize = 512
        public static let prefilteredCubeSize = 256
        /// Number of roughness levels stored across the prefiltered cubemap's mips (roughness 0...1).
        public static let prefilteredMipCount = 6
        public static let irradianceCubeSize = 32
        public static let brdfLUTSize = 256
    }

    public enum Error: Swift.Error {
        case functionNotFound(String)
        case textureCreationFailed
        case commandCreationFailed
    }

    public let device: MTLDevice
    /// GGX-prefiltered specular radiance; sample at `mipLevel = roughness * (prefilteredMipCount - 1)`.
    public let prefilteredCubemap: MTLTexture
    /// Diffuse irradiance (cosine-convolved environment).
    public let irradianceCubemap: MTLTexture
    /// Environment BRDF LUT: `.rg` = (scale, bias), indexed by `(NdotV, roughness)`.
    public let brdfLUT: MTLTexture
    /// Number of roughness mip levels in `prefilteredCubemap`.
    public let prefilteredMipCount: Int
    /// The source equirectangular panorama (full resolution), retained for sharp skybox rendering.
    public let equirectangular: MTLTexture

    public init(device: MTLDevice, equirectangular: MTLTexture) throws {
        self.device = device
        self.equirectangular = equirectangular
        self.prefilteredMipCount = Constants.prefilteredMipCount

        let library = try device.makeDefaultLibrary(bundle: Bundle.module)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw Error.functionNotFound(name) }
            return try device.makeComputePipelineState(function: function)
        }
        let equirectToCubePSO = try pipeline("iblEquirectToCubemap")
        let prefilterPSO = try pipeline("iblPrefilterEnvmap")
        let irradiancePSO = try pipeline("iblComputeIrradiance")
        let brdfPSO = try pipeline("iblIntegrateBRDF")

        // Intermediate base radiance cube; mipmapped so the prefilter pass can sample lower mips
        // (variance reduction). Not retained after the build.
        let baseCube = try Self.makeCubemap(device: device, size: Constants.baseCubeSize, mipmapped: true)
        let prefiltered = try Self.makeCubemap(device: device, size: Constants.prefilteredCubeSize,
                                               mipmapped: true, mipCountLimit: Constants.prefilteredMipCount)
        let irradiance = try Self.makeCubemap(device: device, size: Constants.irradianceCubeSize, mipmapped: false)

        let lutDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg16Float,
                                                                     width: Constants.brdfLUTSize,
                                                                     height: Constants.brdfLUTSize,
                                                                     mipmapped: false)
        lutDescriptor.usage = [.shaderRead, .shaderWrite]
        lutDescriptor.storageMode = .private
        guard let lut = device.makeTexture(descriptor: lutDescriptor) else { throw Error.textureCreationFailed }

        guard let queue = device.makeCommandQueue(),
              let commandBuffer = queue.makeCommandBuffer() else { throw Error.commandCreationFailed }

        // 1. Equirectangular -> base cube (mip 0).
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(equirectToCubePSO)
            encoder.setTexture(equirectangular, index: 0)
            encoder.setTexture(baseCube, index: 1)
            Self.dispatchCube(encoder, size: Constants.baseCubeSize)
            encoder.endEncoding()
        }

        // 2. Generate the base cube's mip chain for prefilter sampling.
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.generateMipmaps(for: baseCube)
            blit.endEncoding()
        }

        // 3. GGX-prefilter specular: one dispatch per roughness mip, into a per-mip cube view.
        for mip in 0..<Constants.prefilteredMipCount {
            let mipSize = max(1, Constants.prefilteredCubeSize >> mip)
            var roughness = Float(mip) / Float(max(1, Constants.prefilteredMipCount - 1))
            guard let mipView = prefiltered.makeTextureView(pixelFormat: prefiltered.pixelFormat,
                                                            textureType: .typeCube,
                                                            levels: mip..<(mip + 1),
                                                            slices: 0..<6) else { throw Error.textureCreationFailed }
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(prefilterPSO)
                encoder.setTexture(baseCube, index: 0)
                encoder.setTexture(mipView, index: 1)
                encoder.setBytes(&roughness, length: MemoryLayout<Float>.size, index: 0)
                Self.dispatchCube(encoder, size: mipSize)
                encoder.endEncoding()
            }
        }

        // 4. Diffuse irradiance.
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(irradiancePSO)
            encoder.setTexture(baseCube, index: 0)
            encoder.setTexture(irradiance, index: 1)
            Self.dispatchCube(encoder, size: Constants.irradianceCubeSize)
            encoder.endEncoding()
        }

        // 5. Environment BRDF LUT.
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(brdfPSO)
            encoder.setTexture(lut, index: 0)
            Self.dispatch2D(encoder, width: Constants.brdfLUTSize, height: Constants.brdfLUTSize)
            encoder.endEncoding()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        self.prefilteredCubemap = prefiltered
        self.irradianceCubemap = irradiance
        self.brdfLUT = lut
    }

    // MARK: - Helpers

    private static func makeCubemap(device: MTLDevice,
                                    size: Int,
                                    mipmapped: Bool,
                                    mipCountLimit: Int? = nil) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: .rgba16Float,
                                                                    size: size,
                                                                    mipmapped: mipmapped)
        if let mipCountLimit {
            descriptor.mipmapLevelCount = min(descriptor.mipmapLevelCount, mipCountLimit)
        }
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private
        guard let texture = device.makeTexture(descriptor: descriptor) else { throw Error.textureCreationFailed }
        return texture
    }

    private static func dispatchCube(_ encoder: MTLComputeCommandEncoder, size: Int) {
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        encoder.dispatchThreads(MTLSize(width: size, height: size, depth: 6),
                                threadsPerThreadgroup: threadsPerThreadgroup)
    }

    private static func dispatch2D(_ encoder: MTLComputeCommandEncoder, width: Int, height: Int) {
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                threadsPerThreadgroup: threadsPerThreadgroup)
    }
}
