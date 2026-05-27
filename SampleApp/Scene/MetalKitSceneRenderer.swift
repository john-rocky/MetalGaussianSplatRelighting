#if os(iOS) || os(macOS)

import Metal
import MetalKit
import MetalSplatter
import Observation
import os
import SampleBoxRenderer
import simd
import SplatIO
import SwiftUI

@MainActor
class MetalKitSceneRenderer: NSObject, MTKViewDelegate {
    private static let log =
        Logger(subsystem: Bundle.main.bundleIdentifier!,
               category: "MetalKitSceneRenderer")

    let metalKitView: MTKView
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    var model: ModelIdentifier?
    var modelRenderer: (any ModelRenderer)?
    var proceduralSplatController: ProceduralSplatController?
    var relightControls: RelightControls?

    /// The IBL environment currently applied to the splat renderer, and a cache so repeated swaps
    /// don't re-run the (GPU) prefilter/irradiance precompute.
    private var appliedEnvironmentChoice: RelightControls.EnvironmentChoice?
    private var environmentCache: [RelightControls.EnvironmentChoice: IBLEnvironment] = [:]

    let inFlightSemaphore = DispatchSemaphore(value: Constants.maxSimultaneousRenders)

    var lastRotationUpdateTimestamp: Date? = nil
    var rotation: Angle = .zero

    // Orbit camera state. Until the user first interacts, the model gently auto-yaws (idle demo);
    // any drag/pinch takes over. Yaw spins about the (corrected) up axis, pitch tilts up/down,
    // distance is the camera's distance from the model for pinch-zoom.
    var cameraYaw: Angle = .zero
    var cameraPitch: Angle = Constants.cameraInitialPitch
    var cameraDistance: Float = Constants.cameraInitialDistance
    var userHasInteracted = false

    var drawableSize: CGSize = .zero

    /// Apply an orbit drag (screen-point deltas). Disables auto-rotation on first use.
    func orbitBy(dx: Float, dy: Float) {
        userHasInteracted = true
        cameraYaw += Angle(radians: Double(dx * Constants.cameraOrbitRadiansPerPoint))
        let newPitch = cameraPitch.radians + Double(dy * Constants.cameraOrbitRadiansPerPoint)
        cameraPitch = Angle(radians: min(max(newPitch, -Constants.cameraPitchLimit.radians), Constants.cameraPitchLimit.radians))
    }

    /// Apply a pinch-zoom. `scale` is the gesture's cumulative scale relative to `referenceDistance`.
    func zoomBy(scale: Float, referenceDistance: Float) {
        userHasInteracted = true
        cameraDistance = min(max(referenceDistance / max(scale, 0.0001), Constants.cameraMinDistance), Constants.cameraMaxDistance)
    }

    init?(_ metalKitView: MTKView) {
        self.device = metalKitView.device!
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        self.metalKitView = metalKitView
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float
        metalKitView.sampleCount = 1
        metalKitView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    }

    func load(_ model: ModelIdentifier?) async throws {
        guard model != self.model else { return }
        self.model = model

        modelRenderer = nil
        proceduralSplatController = nil
        switch model {
        case .gaussianSplat(let url):
            let splat = try SplatRenderer(device: device,
                                          colorFormat: metalKitView.colorPixelFormat,
                                          depthFormat: metalKitView.depthStencilPixelFormat,
                                          sampleCount: metalKitView.sampleCount,
                                          maxViewCount: 1,
                                          maxSimultaneousRenders: Constants.maxSimultaneousRenders)
            let reader = try AutodetectSceneReader(url)
            let points = try await reader.readAll()
            let chunk = try SplatChunk(device: device, from: points)
            await splat.addChunk(chunk)
            // The IBL environment (default: bundled studio HDR) is built lazily on the first render
            // frame and rebuilt whenever the user changes the environment choice — see applyEnvironment.
            modelRenderer = splat
        case .proceduralSplat:
            let controller = try await ProceduralSplatController(
                device: device,
                colorFormat: metalKitView.colorPixelFormat,
                depthFormat: metalKitView.depthStencilPixelFormat,
                sampleCount: metalKitView.sampleCount,
                maxViewCount: 1,
                maxSimultaneousRenders: Constants.maxSimultaneousRenders)
            proceduralSplatController = controller
            modelRenderer = controller.splatRenderer
        case .sampleBox:
            modelRenderer = try! SampleBoxRenderer(device: device,
                                                   colorFormat: metalKitView.colorPixelFormat,
                                                   depthFormat: metalKitView.depthStencilPixelFormat,
                                                   sampleCount: metalKitView.sampleCount,
                                                   maxViewCount: 1,
                                                   maxSimultaneousRenders: Constants.maxSimultaneousRenders)
        case .none:
            break
        }
    }

    private var viewport: ModelRendererViewportDescriptor {
        let projectionMatrix = matrix_perspective_right_hand(fovyRadians: Float(Constants.fovy.radians),
                                                             aspectRatio: Float(drawableSize.width / drawableSize.height),
                                                             nearZ: 0.1,
                                                             farZ: 100.0)

        // Orbit camera: pull back by the (pinch-controlled) distance, then pitch and yaw around the
        // model. Before the first user interaction we add a gentle auto-yaw for an idle demo.
        let autoYaw = userHasInteracted ? 0 : Float(rotation.radians)
        let yawMatrix = matrix4x4_rotation(radians: Float(cameraYaw.radians) + autoYaw, axis: SIMD3<Float>(0, 1, 0))
        let pitchMatrix = matrix4x4_rotation(radians: Float(cameraPitch.radians), axis: SIMD3<Float>(1, 0, 0))
        let translationMatrix = matrix4x4_translation(0.0, 0.0, -cameraDistance)

        // Ref-NeRF / Blender datasets (helmet, car, ...) are Z-up, but this viewer's camera is Y-up,
        // which otherwise renders them rolled 90° onto their side. Map data +Z -> viewer +Y.
        let upCalibration = matrix4x4_rotation(radians: -.pi / 2, axis: SIMD3<Float>(1, 0, 0))

        let viewport = MTLViewport(originX: 0, originY: 0, width: drawableSize.width, height: drawableSize.height, znear: 0, zfar: 1)

        return ModelRendererViewportDescriptor(viewport: viewport,
                                               projectionMatrix: projectionMatrix,
                                               viewMatrix: translationMatrix * pitchMatrix * yawMatrix * upCalibration,
                                               screenSize: SIMD2(x: Int(drawableSize.width), y: Int(drawableSize.height)))
    }

    private func updateRotation() {
        let now = Date()
        defer {
            lastRotationUpdateTimestamp = now
        }

        guard let lastRotationUpdateTimestamp else { return }
        rotation += Constants.rotationPerSecond * now.timeIntervalSince(lastRotationUpdateTimestamp)
    }

    func draw(in view: MTKView) {
        guard let modelRenderer, modelRenderer.isReadyToRender else { return }
        guard let drawable = view.currentDrawable else { return }

        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            inFlightSemaphore.signal()
            return
        }

        let semaphore = inFlightSemaphore
        commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
            semaphore.signal()
        }

        updateRotation()
        proceduralSplatController?.update()

        // Apply relighting controls to the splat renderer (if any).
        if let splat = modelRenderer as? SplatRenderer, let controls = relightControls {
            // Build / swap the IBL environment when the selected choice changes (cached after first use).
            if appliedEnvironmentChoice != controls.environmentChoice {
                applyEnvironment(controls.environmentChoice, to: splat)
                appliedEnvironmentChoice = controls.environmentChoice
            }
            var settings = SplatRenderer.RelightSettings()
            settings.isEnabled = controls.isEnabled && splat.environment != nil
            // The IBL environment is authored Y-up, but the scene (Ref-NeRF/Blender) is Z-up. The
            // shader samples the env (skybox rays + reflections) with scene-space directions, so map
            // them into the env's Y-up frame (Z-up -> Y-up); the rotation slider spins about the
            // scene up axis (Z) first. Without this the background/reflections are rolled 90°.
            let envYaw = matrix4x4_rotation(radians: Float(controls.rotationDegrees * .pi / 180.0),
                                            axis: SIMD3<Float>(0, 0, 1))
            let zUpToYUp = matrix4x4_rotation(radians: -.pi / 2, axis: SIMD3<Float>(1, 0, 0))
            settings.environmentRotation = zUpToYUp * envYaw
            settings.environmentIntensity = Float(controls.environmentIntensity)
            settings.debugMode = UInt32(controls.debugMode)
            settings.roughnessOverride = controls.useTrainedMaterial ? -1 : Float(controls.roughness)
            settings.reflectionStrengthOverride = controls.useTrainedMaterial ? -1 : Float(controls.reflectionStrength)
            splat.relightSettings = settings
        }

        let didRender: Bool
        do {
            didRender = try modelRenderer.render(viewports: [viewport],
                                                 colorTexture: view.multisampleColorTexture ?? drawable.texture,
                                                 colorStoreAction: view.multisampleColorTexture == nil ? .store : .multisampleResolve,
                                                 depthTexture: view.depthStencilTexture,
                                                 rasterizationRateMap: nil,
                                                 renderTargetArrayLength: 0,
                                                 to: commandBuffer)
        } catch {
            Self.log.error("Unable to render scene: \(error.localizedDescription)")
            didRender = false
        }

        if didRender {
            commandBuffer.present(drawable)
        }

        commandBuffer.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
    }

    /// Builds (or reuses) the IBL environment for `choice` and assigns it to the splat renderer.
    /// Falls back to the procedural sky if the bundled HDR can't be loaded or precomputed.
    private func applyEnvironment(_ choice: RelightControls.EnvironmentChoice, to splat: SplatRenderer) {
        if let cached = environmentCache[choice] {
            splat.environment = cached
            return
        }
        let equirect: MTLTexture?
        if let resource = choice.resourceName {
            equirect = EquirectEnvironment.loadBundled(device: device, resource: resource)
        } else {
            equirect = ProceduralSky.makeEquirectangular(device: device)
        }
        if let equirect, let environment = try? IBLEnvironment(device: device, equirectangular: equirect) {
            environmentCache[choice] = environment
            splat.environment = environment
        } else if choice != .procedural {
            applyEnvironment(.procedural, to: splat)   // studio asset unavailable -> graceful fallback
        }
    }
}

// MARK: - Relighting controls, environment, and UI

/// Observable relighting controls, shared between the SwiftUI control panel and the renderer.
@Observable
@MainActor
final class RelightControls {
    /// Selectable image-based-lighting environment. `studio` loads a bundled HDR panorama; reflections
    /// (and the Env-rotation slider) respond to whichever is chosen.
    enum EnvironmentChoice: Int, CaseIterable, Identifiable {
        case autoshop, studio, procedural
        var id: Int { rawValue }
        var label: String {
            switch self {
            case .autoshop: return "Auto Shop"
            case .studio: return "Studio"
            case .procedural: return "Procedural"
            }
        }
        /// Bundled equirect resource name, or nil for the procedural sky.
        var resourceName: String? {
            switch self {
            case .autoshop: return "autoshop_env"
            case .studio: return "studio_env"
            case .procedural: return nil
            }
        }
    }
    var environmentChoice: EnvironmentChoice = .autoshop
    var isEnabled: Bool = true
    /// When true, use each splat's trained per-splat roughness / reflection; when false the sliders
    /// override every splat globally (useful for standard 3DGS scenes with synthesized materials).
    var useTrainedMaterial: Bool = true
    /// Environment rotation about the up axis, in degrees.
    var rotationDegrees: Double = 0
    /// Global roughness override (0...1), applied to synthetic-material scenes.
    var roughness: Double = 0.4
    /// Global reflection-strength override (0...1).
    var reflectionStrength: Double = 0.5
    /// Scales the sampled environment radiance.
    var environmentIntensity: Double = 1.0
    /// 0 shaded, 1 normal, 2 roughness, 3 reflectionStrength, 4 prefiltered environment.
    var debugMode: Int = 0
}

/// Generates a simple procedural HDR-ish sky as an equirectangular texture (gradient + sun),
/// for driving IBL relighting without an external environment map. The bright sun makes specular
/// reflections easy to see as the environment is rotated.
enum ProceduralSky {
    static func makeEquirectangular(device: MTLDevice, width: Int = 512, height: Int = 256) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float,
                                                                  width: width,
                                                                  height: height,
                                                                  mipmapped: false)
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else { return nil }

        let sunDirection = simd_normalize(SIMD3<Float>(0.4, 0.7, 0.3))
        let zenith = SIMD3<Float>(0.10, 0.22, 0.45)
        let horizon = SIMD3<Float>(0.65, 0.72, 0.80)
        let ground = SIMD3<Float>(0.20, 0.18, 0.16)
        let sunColor = SIMD3<Float>(1.0, 0.95, 0.85)

        var pixels = [Float16](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            let v = (Float(y) + 0.5) / Float(height)
            let theta = v * Float.pi
            let sinTheta = sin(theta)
            let cosTheta = cos(theta)
            for x in 0..<width {
                let u = (Float(x) + 0.5) / Float(width)
                let phi = (u * 2 - 1) * Float.pi
                let dir = SIMD3<Float>(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi))

                var color: SIMD3<Float>
                if dir.y >= 0 {
                    color = simd_mix(horizon, zenith, SIMD3<Float>(repeating: pow(dir.y, 0.5)))
                } else {
                    color = simd_mix(horizon, ground, SIMD3<Float>(repeating: pow(-dir.y, 0.5)))
                }

                let s = max(0, simd_dot(dir, sunDirection))
                let sun = pow(s, 600.0) * 12.0 + pow(s, 30.0) * 0.5
                color += sunColor * sun

                let base = (y * width + x) * 4
                pixels[base + 0] = Float16(color.x)
                pixels[base + 1] = Float16(color.y)
                pixels[base + 2] = Float16(color.z)
                pixels[base + 3] = 1
            }
        }

        pixels.withUnsafeBytes { raw in
            texture.replace(region: MTLRegionMake2D(0, 0, width, height),
                            mipmapLevel: 0,
                            withBytes: raw.baseAddress!,
                            bytesPerRow: width * 4 * MemoryLayout<Float16>.size)
        }
        return texture
    }
}

/// A compact control panel for the relightable rendering settings.
struct RelightControlsView: View {
    @Bindable var controls: RelightControls

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle("Relight (split-sum IBL)", isOn: $controls.isEnabled)
                .font(.subheadline.bold())

            if controls.isEnabled {
                Picker("Environment", selection: $controls.environmentChoice) {
                    ForEach(RelightControls.EnvironmentChoice.allCases) { choice in
                        Text(choice.label).tag(choice)
                    }
                }
                .pickerStyle(.segmented)

                labeledSlider("Env rotation", value: $controls.rotationDegrees, range: 0...360)
                labeledSlider("Intensity", value: $controls.environmentIntensity, range: 0...3)

                Toggle("Use trained material", isOn: $controls.useTrainedMaterial)
                    .font(.caption)
                if !controls.useTrainedMaterial {
                    labeledSlider("Roughness", value: $controls.roughness, range: 0...1)
                    labeledSlider("Reflection", value: $controls.reflectionStrength, range: 0...1)
                }

                Picker("Debug", selection: $controls.debugMode) {
                    Text("Shaded").tag(0)
                    Text("Normal").tag(1)
                    Text("Rough").tag(2)
                    Text("Refl").tag(3)
                    Text("Env").tag(4)
                    Text("Irr").tag(5)
                    Text("Albedo").tag(6)
                }
                .pickerStyle(.segmented)
            }
        }
        .padding(12)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .frame(maxWidth: 380)
    }

    @ViewBuilder
    private func labeledSlider(_ title: String, value: Binding<Double>, range: ClosedRange<Double>) -> some View {
        HStack {
            Text(title)
                .font(.caption)
                .frame(width: 90, alignment: .leading)
            Slider(value: value, in: range)
        }
    }
}

#endif // os(iOS) || os(macOS)
