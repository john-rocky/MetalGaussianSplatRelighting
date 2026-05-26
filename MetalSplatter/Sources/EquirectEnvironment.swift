import Foundation
import Metal

/// Loads a preprocessed equirectangular HDR environment that is bundled with the package, into an
/// `.rgba16Float` texture suitable for `IBLEnvironment(device:equirectangular:)`.
///
/// File format (little-endian): magic `"HEQR"`, `UInt32 width`, `UInt32 height`, then
/// `width * height * 4` IEEE half floats (RGBA, row 0 = top, +Y up) — the same lat/long layout the
/// IBL precompute expects. HDR panoramas are preprocessed offline (decoded from Radiance .hdr and
/// normalized to a sane linear range) so the device needs no HDR decoder.
public enum EquirectEnvironment {
    /// Loads `<resource>.bin` from the package bundle, or nil if missing/malformed.
    public static func loadBundled(device: MTLDevice, resource: String) -> MTLTexture? {
        guard let url = Bundle.module.url(forResource: resource, withExtension: "bin"),
              let data = try? Data(contentsOf: url) else { return nil }
        return makeTexture(device: device, data: data)
    }

    static func makeTexture(device: MTLDevice, data: Data) -> MTLTexture? {
        return data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> MTLTexture? in
            guard raw.count > 12, let base = raw.baseAddress else { return nil }
            // Magic "HEQR"
            guard base.load(fromByteOffset: 0, as: UInt8.self) == 0x48,  // H
                  base.load(fromByteOffset: 1, as: UInt8.self) == 0x45,  // E
                  base.load(fromByteOffset: 2, as: UInt8.self) == 0x51,  // Q
                  base.load(fromByteOffset: 3, as: UInt8.self) == 0x52   // R
            else { return nil }
            let width = Int(UInt32(littleEndian: base.loadUnaligned(fromByteOffset: 4, as: UInt32.self)))
            let height = Int(UInt32(littleEndian: base.loadUnaligned(fromByteOffset: 8, as: UInt32.self)))
            let bytesPerRow = width * 4 * MemoryLayout<UInt16>.size
            guard width > 0, height > 0, raw.count >= 12 + bytesPerRow * height else { return nil }

            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float,
                                                                      width: width, height: height,
                                                                      mipmapped: false)
            descriptor.usage = .shaderRead
            descriptor.storageMode = .shared
            guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }

            texture.replace(region: MTLRegionMake2D(0, 0, width, height),
                            mipmapLevel: 0,
                            withBytes: base.advanced(by: 12),
                            bytesPerRow: bytesPerRow)
            return texture
        }
    }
}
