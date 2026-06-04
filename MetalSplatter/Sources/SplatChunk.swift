import Foundation
import Metal
import simd
import SplatIO

/// Stable, opaque handle identifying a chunk within a SplatRenderer.
/// ChunkIDs are assigned monotonically and never reused; they are *not* the same as the contiguous
/// chunk indices used internally by the sorter and shaders (see ``ChunkedSplatIndex``).
public struct ChunkID: Hashable, Sendable, Equatable {
    public let rawValue: UInt

    public init(rawValue: UInt) {
        self.rawValue = rawValue
    }
}

/// A chunk of gaussian splats.
/// Created externally and added to a SplatRenderer via `addChunk(_:)`.
public struct SplatChunk: @unchecked Sendable {
    /// The splat data buffer
    public let splats: MetalBuffer<EncodedSplatPoint>

    /// Optional buffer containing higher-order spherical harmonics coefficients.
    /// This is nil for SH degree 0 (view-independent color).
    /// For SH1: 9 Float16 values per splat (3 coefficients × 3 RGB)
    /// For SH2: 24 Float16 values per splat (8 coefficients × 3 RGB, cumulative)
    /// For SH3: 45 Float16 values per splat (15 coefficients × 3 RGB, cumulative)
    public let shCoefficients: MetalBuffer<Float16>?

    /// Optional buffer of per-splat PBR material attributes for relightable rendering.
    /// `nil` if this chunk carries no material.
    public let materials: MetalBuffer<EncodedSplatMaterial>?

    /// Optional buffer of per-splat Ref-Gaussian ASG (Anisotropic Spherical Gaussian) indirect
    /// coefficients: 160 Float16 per splat (32 lobes × 5 channels, lobe-major). `nil` for any
    /// scene that wasn't trained with the ASG-indirect head. ~320 bytes/splat → ~64 MB for a
    /// 200k-splat scene.
    public let asgCoefficients: MetalBuffer<Float16>?

    /// The spherical harmonics degree for this chunk.
    /// All splats in a chunk share the same SH degree.
    public let shDegree: SHDegree

    /// Number of splats in this chunk
    public var splatCount: Int { splats.count }

    /// Creates a new splat chunk with spherical harmonics support.
    /// - Parameters:
    ///   - splats: The Metal buffer containing splat data (with raw SH0 in color field)
    ///   - shCoefficients: Optional buffer with higher-order SH coefficients (nil for SH0)
    ///   - shDegree: The spherical harmonics degree for this chunk
    public init(splats: MetalBuffer<EncodedSplatPoint>,
                shCoefficients: MetalBuffer<Float16>? = nil,
                materials: MetalBuffer<EncodedSplatMaterial>? = nil,
                asgCoefficients: MetalBuffer<Float16>? = nil,
                shDegree: SHDegree = .sh0) {
        self.splats = splats
        self.shCoefficients = shCoefficients
        self.materials = materials
        self.asgCoefficients = asgCoefficients
        self.shDegree = shDegree
    }

    /// Creates a chunk from scene points, extracting and preserving spherical harmonics data.
    /// - Parameters:
    ///   - device: Metal device for buffer allocation
    ///   - points: Array of scene points with SH data
    public init(device: MTLDevice,
                from points: [SplatPoint]) throws {
        // Determine the SH degree from the data
        let shDegree = points.first?.color.shDegree ?? .sh0

        // Create splat buffer - always stores raw SH0 coefficients
        let splatBuffer = try MetalBuffer<EncodedSplatPoint>(device: device, capacity: points.count)
        splatBuffer.count = points.count

        for (i, point) in points.enumerated() {
            splatBuffer.values[i] = EncodedSplatPoint(point)
        }

        // Always build a per-splat material buffer so any scene can be relit. For standard 3DGS
        // (no material), EncodedSplatMaterial synthesizes a normal from the splat geometry.
        //
        // 2DGS surfels (e.g. Ref-Gaussian) have a sign-ambiguous normal — the disk is two-sided, so
        // ~half the stored normals point inward. Left as-is they blend into noise (the directions are
        // smooth, but the signs are random). Orient them consistently outward from the cloud centroid,
        // which resolves the sign globally for a mostly-convex object and yields a smooth normal field.
        let materialBuffer = try MetalBuffer<EncodedSplatMaterial>(device: device, capacity: points.count)
        materialBuffer.count = points.count
        var positionSum = SIMD3<Float>.zero
        for point in points { positionSum += point.position }
        let centroid = points.isEmpty ? .zero : positionSum / Float(points.count)
        for (i, point) in points.enumerated() {
            var material = EncodedSplatMaterial(point)
            let normal = SIMD3<Float>(Float(material.normal.x), Float(material.normal.y), Float(material.normal.z))
            if simd_dot(normal, point.position - centroid) < 0 {
                material.normal = PackedHalf3(-normal)
            }
            materialBuffer.values[i] = material
        }
        self.materials = materialBuffer

        // Build the ASG buffer iff any point carries an ASG tail. One-shot: we either have it for
        // every splat (Ref-Gaussian relightable scene) or for none (standard 3DGS / older
        // checkpoints). 160 fp16 per splat, lobe-major (5 floats per lobe).
        let asgCount = 160
        if points.first?.material?.indirectASG?.count == asgCount {
            let asgBuffer = try MetalBuffer<Float16>(device: device, capacity: points.count * asgCount)
            asgBuffer.count = points.count * asgCount
            for (i, point) in points.enumerated() {
                guard let asg = point.material?.indirectASG, asg.count == asgCount else {
                    // Defensive: if a point inside the chunk is missing its ASG tail, fill with
                    // zeros so the shader's exp(ep - 3) -> exp(-3) ~= 0.05 stays bounded but
                    // doesn't blow up reflections.
                    let base = i * asgCount
                    for j in 0..<asgCount { asgBuffer.values[base + j] = 0 }
                    continue
                }
                let base = i * asgCount
                for j in 0..<asgCount {
                    asgBuffer.values[base + j] = Float16(asg[j])
                }
            }
            self.asgCoefficients = asgBuffer
        } else {
            self.asgCoefficients = nil
        }

        // Create SH coefficient buffer if we have higher-order SH
        if shDegree > .sh0 {
            let coeffsPerSplat = shDegree.extraCoefficientCount * 3  // RGB per coefficient
            let totalCoeffs = points.count * coeffsPerSplat
            let shBuffer = try MetalBuffer<Float16>(device: device, capacity: totalCoeffs)
            shBuffer.count = totalCoeffs

            // Copy SH coefficients to buffer
            for (i, point) in points.enumerated() {
                let higherOrderCoeffs = point.color.higherOrderSHCoefficients
                let offset = i * coeffsPerSplat
                for (j, coeff) in higherOrderCoeffs.enumerated() {
                    shBuffer.values[offset + j] = Float16(coeff)
                }
            }

            self.shCoefficients = shBuffer
            self.shDegree = shDegree
        } else {
            self.shCoefficients = nil
            self.shDegree = .sh0
        }

        self.splats = splatBuffer
    }
}

/// Index into the sorted splat list, identifying both the chunk and the local splat index.
/// Uses 8 bytes for alignment (6 bytes of meaningful data).
/// Keep in sync with ShaderCommon.h : ChunkedSplatIndex
public struct ChunkedSplatIndex: Sendable {
    /// Contiguous index into the chunks array (0..<chunkCount), *not* the same as ``ChunkID``.
    public var chunkIndex: UInt16

    /// Padding for alignment
    public var _padding: UInt16

    /// Index of the splat within its chunk
    public var splatIndex: UInt32

    public init(chunkIndex: UInt16, splatIndex: UInt32) {
        self.chunkIndex = chunkIndex
        self._padding = 0
        self.splatIndex = splatIndex
    }
}
