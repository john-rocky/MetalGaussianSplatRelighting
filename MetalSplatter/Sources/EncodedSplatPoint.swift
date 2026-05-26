import Foundation
import Metal
import simd
import SplatIO

#if arch(x86_64)
#warning("x86_64 targets are unsupported by MetalSplatter and will fail at runtime. MetalSplatter builds on x86_64 only because Xcode builds Swift Packages as universal binaries and provides no way to override this. When Swift supports Float16 on x86_64, this may be revisited.")
public typealias Float16 = Float
#endif

struct PackedHalf3 {
    var x: Float16
    var y: Float16
    var z: Float16

    init(x: Float16, y: Float16, z: Float16) {
        self.x = x
        self.y = y
        self.z = z
    }

    init(_ v: SIMD3<Float>) {
        self.init(x: Float16(v.x), y: Float16(v.y), z: Float16(v.z))
    }
}

struct PackedRGBHalf4 {
    var r: Float16
    var g: Float16
    var b: Float16
    var a: Float16
}

// Keep in sync with Shaders.metal : Splat
public struct EncodedSplatPoint {
    var position: MTLPackedFloat3
    var colorSH0: PackedRGBHalf4 // sh0 color + opacity
    var covA: PackedHalf3
    var covB: PackedHalf3
}

// MARK: - Splat Conversion

extension EncodedSplatPoint {
    public init(_ splat: SplatPoint) {
        self.init(position: splat.position,
                  colorSH0: splat.color.sh0,
                  opacity: splat.opacity.asLinearFloat,
                  scale: splat.scale.asLinearFloat,
                  rotation: splat.rotation.normalized)
    }

    /// Creates a splat with explicit raw SH0 coefficients.
    /// - Parameters:
    ///   - position: World-space position
    ///   - colorSH0: SH degree-0 coefficients (NOT sRGB color; use `SplatPoint.Color.sh0` to convert)
    ///   - opacity: Linear opacity (0-1)
    ///   - scale: Linear scale
    ///   - rotation: Rotation quaternion
    public init(position: SIMD3<Float>,
                colorSH0: SIMD3<Float>,
                opacity: Float,
                scale: SIMD3<Float>,
                rotation: simd_quatf) {
        let transform = simd_float3x3(rotation) * simd_float3x3(diagonal: scale)
        let cov3D = transform * transform.transpose
        self.init(position: MTLPackedFloat3Make(position.x, position.y, position.z),
                  colorSH0: PackedRGBHalf4(r: Float16(colorSH0.x), g: Float16(colorSH0.y), b: Float16(colorSH0.z), a: Float16(opacity)),
                  covA: PackedHalf3(x: Float16(cov3D[0, 0]), y: Float16(cov3D[0, 1]), z: Float16(cov3D[0, 2])),
                  covB: PackedHalf3(x: Float16(cov3D[1, 1]), y: Float16(cov3D[1, 2]), z: Float16(cov3D[2, 2])))
    }
}

// MARK: - Material

// Keep in sync with ShaderCommon.h : SplatMaterial (16 bytes)
public struct EncodedSplatMaterial {
    var normal: PackedHalf3            // object-space surface normal
    var specularTint: PackedHalf3      // specular tint / F0 color
    var roughness: Float16
    var reflectionStrength: Float16    // blends diffuse vs. specular
}

extension EncodedSplatMaterial {
    /// Encodes a splat's PBR material. For standard 3DGS splats (no material) a material is
    /// synthesized so the scene can still be relit: the normal is the splat's flattest principal
    /// axis, with neutral defaults that the shader can override via uniforms.
    public init(_ splat: SplatPoint) {
        if let material = splat.material {
            self.init(normal: PackedHalf3(material.normal),
                      specularTint: PackedHalf3(material.specularTint),
                      roughness: Float16(material.roughness),
                      reflectionStrength: Float16(material.reflectionStrength))
        } else {
            let scale = splat.scale.asLinearFloat
            let basis = simd_float3x3(splat.rotation.normalized)
            let axis: SIMD3<Float>
            if scale.x <= scale.y && scale.x <= scale.z {
                axis = basis.columns.0
            } else if scale.y <= scale.z {
                axis = basis.columns.1
            } else {
                axis = basis.columns.2
            }
            let length = simd_length(axis)
            let normal = length > 1e-6 ? axis / length : SIMD3<Float>(0, 0, 1)
            self.init(normal: PackedHalf3(normal),
                      specularTint: PackedHalf3(SIMD3<Float>(repeating: 1)),
                      roughness: Float16(0.4),
                      reflectionStrength: Float16(0.4))
        }
    }
}

// MARK: - Deprecated

@available(*, deprecated, renamed: "EncodedSplatPoint")
public typealias EncodedSplat = EncodedSplatPoint
