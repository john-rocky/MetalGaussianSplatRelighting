import Foundation
import PLYIO
import simd

public class SplatPLYSceneReader: SplatSceneReader {
    enum Error: LocalizedError {
        case unsupportedFileContents(String?)
        case unexpectedPointCountDiscrepancy
        case internalConsistency(String?)

        public var errorDescription: String? {
            switch self {
            case .unsupportedFileContents(let description):
                if let description {
                    "Unexpected file contents for a splat PLY: \(description)"
                } else {
                    "Unexpected file contents for a splat PLY"
                }
            case .unexpectedPointCountDiscrepancy:
                "Unexpected point count discrepancy"
            case .internalConsistency(let description):
                "Internal error in SplatPLYSceneReader: \(description ?? "(unknown)")"
            }
        }
    }

    private let source: ReaderSource

    public init(_ url: URL) throws {
        guard url.isFileURL else {
            throw ReaderSource.Error.cannotOpen(url: url)
        }
        self.source = .url(url)
    }

    public init(_ data: Data) throws {
        self.source = .memory(data)
    }

    public func read() async throws -> AsyncThrowingStream<[SplatPoint], Swift.Error> {
        let (header, plyStream) = try await PLYReader(source).read()

        let elementMapping = try ElementInputMapping.elementMapping(for: header)

        // TODO SCIER: report expected point count
        return AsyncThrowingStream { continuation in
            Task {
                var points = [SplatPoint]()

                for try await plyStreamElementSeries in plyStream {
                    var pointCount = 0
                    // Skip non-vertex element types (e.g. face, extrinsic, intrinsic metadata)
                    guard plyStreamElementSeries.typeIndex == elementMapping.elementTypeIndex else {
                        continue
                    }

                    do {
                        for element in plyStreamElementSeries.elements {
                            if points.count == pointCount {
                                points.append(SplatPoint(position: .zero,
                                                              color: .sRGBUInt8(.zero),
                                                              opacity: .linearFloat(.zero),
                                                              scale: .exponent(.zero),
                                                              rotation: .init(vector: .zero)))
                            }

                            try points[pointCount].apply(elementMapping, from: element)
                            pointCount += 1
                        }
                    } catch {
                        continuation.finish(throwing: error)
                        return
                    }

                    continuation.yield(Array(points.prefix(pointCount)))
                }
                
                // TODO SCIER: validate expected point count
                continuation.finish()
            }
        }
    }
}

private struct ElementInputMapping {
    public enum Color {
        case sphericalHarmonicFloat([SIMD3<Int>])
        case sRGBFloat256(SIMD3<Int>)  // Legacy NeRF Studio format, normalized on read
        case sRGBUInt8(SIMD3<Int>)
    }

    static let sphericalHarmonicsCount = 45

    /// Property indices for Ref-Gaussian (2DGS) PBR material attributes.
    struct MaterialMapping {
        let reflectionStrengthIndex: Int
        let roughnessIndex: Int
        let specularTintIndices: SIMD3<Int>     // ori_color_0..2
        let normalDeltaIndices: SIMD3<Int>?     // nx/ny/nz residual (nil -> use geometric normal only)
        /// Property indices for the 160 `ind_asg_*` floats per splat, ordered 0..159 so
        /// `asgIndices[i]` reads the i-th flat ASG coefficient. `nil` if the scene wasn't trained
        /// with the ASG-indirect head.
        let asgIndices: [Int]?
    }

    let elementTypeIndex: Int

    let positionXPropertyIndex: Int
    let positionYPropertyIndex: Int
    let positionZPropertyIndex: Int
    let colorPropertyIndices: Color
    let scaleXPropertyIndex: Int
    let scaleYPropertyIndex: Int
    let scaleZPropertyIndex: Int?           // nil for 2DGS surfels (Ref-Gaussian): a thin 3rd axis is synthesized on read
    let opacityPropertyIndex: Int
    let rotation0PropertyIndex: Int
    let rotation1PropertyIndex: Int
    let rotation2PropertyIndex: Int
    let rotation3PropertyIndex: Int
    let material: MaterialMapping?          // non-nil for Ref-Gaussian relightable scenes

    static func elementMapping(for header: PLYHeader) throws -> ElementInputMapping {
        guard let elementTypeIndex = header.index(forElementNamed: SplatPLYConstants.ElementName.point.rawValue) else {
            throw SplatPLYSceneReader.Error.unsupportedFileContents("No element type \"\(SplatPLYConstants.ElementName.point.rawValue)\" found")
        }
        let headerElement = header.elements[elementTypeIndex]

        let positionXPropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.positionX)
        let positionYPropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.positionY)
        let positionZPropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.positionZ)

        let color: Color
        if let sh0_rPropertyIndex = try headerElement.index(forOptionalFloat32PropertyNamed: SplatPLYConstants.PropertyName.sh0_r),
           let sh0_gPropertyIndex = try headerElement.index(forOptionalFloat32PropertyNamed: SplatPLYConstants.PropertyName.sh0_g),
           let sh0_bPropertyIndex = try headerElement.index(forOptionalFloat32PropertyNamed: SplatPLYConstants.PropertyName.sh0_b) {
            let primaryColorPropertyIndices = SIMD3<Int>(x: sh0_rPropertyIndex, y: sh0_gPropertyIndex, z: sh0_bPropertyIndex)
            if headerElement.hasProperty(forName: "\(SplatPLYConstants.PropertyName.sphericalHarmonicsPrefix)0") {
                let individualSphericalHarmonicsPropertyIndices: [Int] = try (0..<sphericalHarmonicsCount).map {
                    try headerElement.index(forFloat32PropertyNamed: [ "\(SplatPLYConstants.PropertyName.sphericalHarmonicsPrefix)\($0)" ])
                }
                // PLY files store SH coefficients channel-by-channel (all R, then all G, then all B),
                // but we need them RGB-interleaved. Reorganize the indices accordingly.
                // For 45 f_rest properties: R=0-14, G=15-29, B=30-44
                // Coefficient i maps to: (R[i], G[i], B[i]) = (i, i+15, i+30)
                let coeffsPerChannel = individualSphericalHarmonicsPropertyIndices.count / 3
                let sphericalHarmonicsPropertyIndices: [SIMD3<Int>] = (0..<coeffsPerChannel).map { i in
                    SIMD3<Int>(individualSphericalHarmonicsPropertyIndices[i],
                               individualSphericalHarmonicsPropertyIndices[i + coeffsPerChannel],
                               individualSphericalHarmonicsPropertyIndices[i + 2 * coeffsPerChannel])
                }
                color = .sphericalHarmonicFloat([primaryColorPropertyIndices] + sphericalHarmonicsPropertyIndices)
            } else {
                color = .sphericalHarmonicFloat([primaryColorPropertyIndices])
            }
        } else if headerElement.hasProperty(forName: SplatPLYConstants.PropertyName.colorR, type: .float32) &&
                    headerElement.hasProperty(forName: SplatPLYConstants.PropertyName.colorG, type: .float32) &&
                    headerElement.hasProperty(forName: SplatPLYConstants.PropertyName.colorB, type: .float32) {
            // Special case for legacy NeRF Studio SH=0 files
            let colorRPropertyIndex = try headerElement.index(forPropertyNamed: SplatPLYConstants.PropertyName.colorR, type: .float32)
            let colorGPropertyIndex = try headerElement.index(forPropertyNamed: SplatPLYConstants.PropertyName.colorG, type: .float32)
            let colorBPropertyIndex = try headerElement.index(forPropertyNamed: SplatPLYConstants.PropertyName.colorB, type: .float32)
            color = .sRGBFloat256(SIMD3(colorRPropertyIndex, colorGPropertyIndex, colorBPropertyIndex))
        } else if headerElement.hasProperty(forName: SplatPLYConstants.PropertyName.colorR, type: .uint8) &&
                    headerElement.hasProperty(forName: SplatPLYConstants.PropertyName.colorG, type: .uint8) &&
                    headerElement.hasProperty(forName: SplatPLYConstants.PropertyName.colorB, type: .uint8) {
            let colorRPropertyIndex = try headerElement.index(forPropertyNamed: SplatPLYConstants.PropertyName.colorR, type: .uint8)
            let colorGPropertyIndex = try headerElement.index(forPropertyNamed: SplatPLYConstants.PropertyName.colorG, type: .uint8)
            let colorBPropertyIndex = try headerElement.index(forPropertyNamed: SplatPLYConstants.PropertyName.colorB, type: .uint8)
            color = .sRGBUInt8(SIMD3(colorRPropertyIndex, colorGPropertyIndex, colorBPropertyIndex))
        } else {
            throw SplatPLYSceneReader.Error.unsupportedFileContents("No color property elements found with the expected types")
        }

        let scaleXPropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.scaleX)
        let scaleYPropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.scaleY)
        // Ref-Gaussian / 2DGS surfels have no scale_2; treat it as optional and synthesize a thin axis on read.
        let scaleZPropertyIndex = try headerElement.index(forOptionalFloat32PropertyNamed: SplatPLYConstants.PropertyName.scaleZ)
        let opacityPropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.opacity)

        let rotation0PropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.rotation0)
        let rotation1PropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.rotation1)
        let rotation2PropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.rotation2)
        let rotation3PropertyIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.rotation3)

        // Detect Ref-Gaussian relightable material: presence of refl_strength implies the PBR channels.
        let material: MaterialMapping?
        if let reflectionStrengthIndex = try headerElement.index(forOptionalFloat32PropertyNamed: SplatPLYConstants.PropertyName.reflectionStrength) {
            let roughnessIndex = try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.roughness)
            let specularTintIndices = SIMD3<Int>(
                try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.specularTintR),
                try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.specularTintG),
                try headerElement.index(forFloat32PropertyNamed: SplatPLYConstants.PropertyName.specularTintB))
            var normalDeltaIndices: SIMD3<Int>? = nil
            if let nx = try headerElement.index(forOptionalFloat32PropertyNamed: SplatPLYConstants.PropertyName.normalX),
               let ny = try headerElement.index(forOptionalFloat32PropertyNamed: SplatPLYConstants.PropertyName.normalY),
               let nz = try headerElement.index(forOptionalFloat32PropertyNamed: SplatPLYConstants.PropertyName.normalZ) {
                normalDeltaIndices = SIMD3<Int>(nx, ny, nz)
            }
            // Detect the ASG indirect tail. Require ALL 160 properties to be present — a partial
            // tail would silently produce wrong indirect colors, so we'd rather fall back to
            // direct-only than half-populate.
            var asgIndices: [Int]? = nil
            if headerElement.index(forPropertyNamed: "\(SplatPLYConstants.PropertyName.indASGPrefix)0") != nil {
                var collected: [Int] = []
                collected.reserveCapacity(SplatPLYConstants.PropertyName.indASGCount)
                for i in 0..<SplatPLYConstants.PropertyName.indASGCount {
                    guard let idx = try headerElement.index(forOptionalFloat32PropertyNamed: [
                        "\(SplatPLYConstants.PropertyName.indASGPrefix)\(i)"
                    ]) else {
                        collected.removeAll()
                        break
                    }
                    collected.append(idx)
                }
                if collected.count == SplatPLYConstants.PropertyName.indASGCount {
                    asgIndices = collected
                }
            }
            material = MaterialMapping(reflectionStrengthIndex: reflectionStrengthIndex,
                                       roughnessIndex: roughnessIndex,
                                       specularTintIndices: specularTintIndices,
                                       normalDeltaIndices: normalDeltaIndices,
                                       asgIndices: asgIndices)
        } else {
            material = nil
        }

        return ElementInputMapping(elementTypeIndex: elementTypeIndex,
                                   positionXPropertyIndex: positionXPropertyIndex,
                                   positionYPropertyIndex: positionYPropertyIndex,
                                   positionZPropertyIndex: positionZPropertyIndex,
                                   colorPropertyIndices: color,
                                   scaleXPropertyIndex: scaleXPropertyIndex,
                                   scaleYPropertyIndex: scaleYPropertyIndex,
                                   scaleZPropertyIndex: scaleZPropertyIndex,
                                   opacityPropertyIndex: opacityPropertyIndex,
                                   rotation0PropertyIndex: rotation0PropertyIndex,
                                   rotation1PropertyIndex: rotation1PropertyIndex,
                                   rotation2PropertyIndex: rotation2PropertyIndex,
                                   rotation3PropertyIndex: rotation3PropertyIndex,
                                   material: material)
    }
}

private extension SplatPoint {
    mutating func apply(_ mapping: ElementInputMapping, from element: PLYElement) throws {
        position = SIMD3(x: try element.float32Value(forPropertyIndex: mapping.positionXPropertyIndex),
                         y: try element.float32Value(forPropertyIndex: mapping.positionYPropertyIndex),
                         z: try element.float32Value(forPropertyIndex: mapping.positionZPropertyIndex))

        switch mapping.colorPropertyIndices {
        case .sphericalHarmonicFloat(let sphericalHarmonicsPropertyIndices):
            color = .sphericalHarmonicFloat(try sphericalHarmonicsPropertyIndices.map {
                try SIMD3<Float>(x: element.float32Value(forPropertyIndex: $0.x),
                                 y: element.float32Value(forPropertyIndex: $0.y),
                                 z: element.float32Value(forPropertyIndex: $0.z))
            })
        case .sRGBFloat256(let propertyIndices):
            // Legacy NeRF Studio format: normalize 0-256 float range to 0-255 uint8
            let floatValues = SIMD3<Float>(try element.float32Value(forPropertyIndex: propertyIndices.x),
                                           try element.float32Value(forPropertyIndex: propertyIndices.y),
                                           try element.float32Value(forPropertyIndex: propertyIndices.z))
            let scaled = floatValues / 256.0 * 255.0
            color = .sRGBUInt8(SIMD3<UInt8>(UInt8(scaled.x.clamped(to: 0...255)),
                                            UInt8(scaled.y.clamped(to: 0...255)),
                                            UInt8(scaled.z.clamped(to: 0...255))))
        case .sRGBUInt8(let propertyIndices):
            color = .sRGBUInt8(SIMD3(try element.uint8Value(forPropertyIndex: propertyIndices.x),
                                     try element.uint8Value(forPropertyIndex: propertyIndices.y),
                                     try element.uint8Value(forPropertyIndex: propertyIndices.z)))
        }

        let scaleXRaw = try element.float32Value(forPropertyIndex: mapping.scaleXPropertyIndex)
        let scaleYRaw = try element.float32Value(forPropertyIndex: mapping.scaleYPropertyIndex)
        if let scaleZPropertyIndex = mapping.scaleZPropertyIndex {
            scale = .exponent(SIMD3(scaleXRaw, scaleYRaw,
                                    try element.float32Value(forPropertyIndex: scaleZPropertyIndex)))
        } else {
            // 2DGS surfel: no 3rd scale. Synthesize a thin axis in log-space so the existing
            // 3D-covariance path renders the surfel as a flat disk along its normal.
            // exp(min(sx,sy) + ln(0.1)) == 0.1 * min(scaleX, scaleY) in linear space.
            let thinZRaw = min(scaleXRaw, scaleYRaw) + Float(log(0.1))
            scale = .exponent(SIMD3(scaleXRaw, scaleYRaw, thinZRaw))
        }
        opacity = .logitFloat(try element.float32Value(forPropertyIndex: mapping.opacityPropertyIndex))
        rotation.real   = try element.float32Value(forPropertyIndex: mapping.rotation0PropertyIndex)
        rotation.imag.x = try element.float32Value(forPropertyIndex: mapping.rotation1PropertyIndex)
        rotation.imag.y = try element.float32Value(forPropertyIndex: mapping.rotation2PropertyIndex)
        rotation.imag.z = try element.float32Value(forPropertyIndex: mapping.rotation3PropertyIndex)

        if let materialMapping = mapping.material {
            func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }
            let roughness = sigmoid(try element.float32Value(forPropertyIndex: materialMapping.roughnessIndex))
            let reflectionStrength = sigmoid(try element.float32Value(forPropertyIndex: materialMapping.reflectionStrengthIndex))
            let specularTint = SIMD3<Float>(
                sigmoid(try element.float32Value(forPropertyIndex: materialMapping.specularTintIndices.x)),
                sigmoid(try element.float32Value(forPropertyIndex: materialMapping.specularTintIndices.y)),
                sigmoid(try element.float32Value(forPropertyIndex: materialMapping.specularTintIndices.z)))

            // Reconstruct the shading normal: 2DGS geometric normal (surfel local z-axis, i.e. the
            // rotation applied to +Z) plus the learned residual (nx,ny,nz), renormalized.
            // The view-dependent flip + second residual (nx2..) are deferred to the shader (faceforward).
            let geometricNormal = rotation.normalized.act(SIMD3<Float>(0, 0, 1))
            var normal = geometricNormal
            if let normalDeltaIndices = materialMapping.normalDeltaIndices {
                let delta = SIMD3<Float>(
                    try element.float32Value(forPropertyIndex: normalDeltaIndices.x),
                    try element.float32Value(forPropertyIndex: normalDeltaIndices.y),
                    try element.float32Value(forPropertyIndex: normalDeltaIndices.z))
                normal = geometricNormal + delta
            }
            let normalLength = simd_length(normal)
            let unitNormal = normalLength > 1e-6 ? normal / normalLength : geometricNormal

            // ASG indirect coefficients, remapped from PLY channel-major (all 32 ep_r, then ep_g,
            // ep_b, λ, μ) to lobe-major (5 consecutive floats per lobe). The shader pulls one
            // lobe's worth at a time, so co-locating them avoids 5 strided loads per iteration.
            var asg: [Float]? = nil
            if let asgIndices = materialMapping.asgIndices {
                let count = SplatPLYConstants.PropertyName.indASGCount
                let lobeCount = SplatPLYConstants.PropertyName.indASGLobeCount  // 32
                var lobeMajor = [Float](repeating: 0, count: count)
                for ch in 0..<5 {
                    let base = ch * lobeCount
                    for k in 0..<lobeCount {
                        lobeMajor[5 * k + ch] = try element.float32Value(forPropertyIndex: asgIndices[base + k])
                    }
                }
                asg = lobeMajor
            }

            material = SplatPoint.Material(normal: unitNormal,
                                           roughness: roughness,
                                           reflectionStrength: reflectionStrength,
                                           specularTint: specularTint,
                                           indirectASG: asg)
        }
    }
}

private extension PLYHeader.Element {
    func hasProperty(forName name: String, type: PLYHeader.PrimitivePropertyType? = nil) -> Bool {
        guard let index = index(forPropertyNamed: name) else {
            return false
        }

        if let type {
            guard case .primitive(type) = properties[index].type else {
                return false
            }
        }

        return true
    }

    func hasProperty(forName names: [String], type: PLYHeader.PrimitivePropertyType? = nil) -> Bool {
        for name in names {
            if hasProperty(forName: name, type: type) {
                return true
            }
        }
        return false
    }

    func index(forOptionalPropertyNamed names: [String], type: PLYHeader.PrimitivePropertyType) throws -> Int? {
        for name in names {
            if let index = index(forPropertyNamed: name) {
                guard case .primitive(type) = properties[index].type else { throw SplatPLYSceneReader.Error.unsupportedFileContents("Unexpected type for property \"\(name)\"") }
                return index
            }
        }
        return nil
    }

    func index(forPropertyNamed names: [String], type: PLYHeader.PrimitivePropertyType) throws -> Int {
        guard let result = try index(forOptionalPropertyNamed: names, type: type) else {
            throw SplatPLYSceneReader.Error.unsupportedFileContents("No property named \"\(names.first ?? "(none)")\" found")
        }
        return result
    }

    func index(forOptionalFloat32PropertyNamed names: [String]) throws -> Int? {
        try index(forOptionalPropertyNamed: names, type: .float32)
    }

    func index(forFloat32PropertyNamed names: [String]) throws -> Int {
        try index(forPropertyNamed: names, type: .float32)
    }
}

private extension PLYElement {
    func float32Value(forPropertyIndex propertyIndex: Int) throws -> Float {
        guard case .float32(let typedValue) = properties[propertyIndex] else { throw SplatPLYSceneReader.Error.internalConsistency("Unexpected type for property at index \(propertyIndex)") }
        return typedValue
    }

    func uint8Value(forPropertyIndex propertyIndex: Int) throws -> UInt8 {
        guard case .uint8(let typedValue) = properties[propertyIndex] else { throw SplatPLYSceneReader.Error.internalConsistency("Unexpected type for property at index \(propertyIndex)") }
        return typedValue
    }
}
