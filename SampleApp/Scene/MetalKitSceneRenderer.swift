#if os(iOS) || os(macOS)

import Metal
import MetalKit
import MetalSplatter
import Observation
import QuartzCore
import os
import SampleBoxRenderer
import simd
import SplatIO
import SwiftUI

#if os(iOS)
import ARKit
import CoreVideo
import UIKit
#endif

@MainActor
class MetalKitSceneRenderer: NSObject, MTKViewDelegate {
    private static let log =
        Logger(subsystem: Bundle.main.bundleIdentifier!,
               category: "MetalKitSceneRenderer")

    let metalKitView: MTKView
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    /// The scene renders into this linear-HDR format; presentation (tonemap/bloom/EDR) is applied
    /// afterward by `postProcessor`. Keeping the scene in float HDR means relighting highlights are
    /// not clamped before tonemapping.
    let sceneColorFormat: MTLPixelFormat = .rgba16Float
    /// Offscreen linear-HDR target the scene is rendered into, before the presentation pass.
    private var hdrTexture: MTLTexture?
    /// Tracks the drawable's currently-applied EDR configuration so it's only reconfigured on change.
    private var appliedEDR: Bool?
    /// Final, separate presentation pass (exposure / tonemap / bloom / EDR, or reference passthrough).
    let postProcessor: PostProcessor

    var model: ModelIdentifier?
    var modelRenderer: (any ModelRenderer)?
    var proceduralSplatController: ProceduralSplatController?
    var relightControls: RelightControls?

    /// The IBL environment currently applied to the splat renderer, and a cache so repeated swaps
    /// don't re-run the (GPU) prefilter/irradiance precompute.
    private var appliedEnvironmentChoice: RelightControls.EnvironmentChoice?
    private var environmentCache: [RelightControls.EnvironmentChoice: IBLEnvironment] = [:]

#if os(iOS)
    /// Drives the "AR / Room" environment: an ARKit session that surfaces the room's environment
    /// probe cubemap (for relighting) and per-frame camera image + pose (for full AR). Created lazily
    /// when AR/Room is first selected; paused when leaving it.
    private var arProbeProvider: ARRoomProbeProvider?
    /// Last time the AR-room IBL was rebuilt, used to throttle the GPU-heavy precompute.
    private var lastARProbeBuild: Date?
    private static let arProbeRebuildInterval: TimeInterval = 1.0

    /// Converts the ARKit captured frame (YCbCr) into the linear-RGB background the splat composites
    /// over; the converted image, a cache for wrapping the camera planes, and the world placement.
    private lazy var cameraBackgroundConverter: CameraBackgroundConverter? =
        try? CameraBackgroundConverter(device: device, colorFormat: sceneColorFormat)
    private var cameraTextureCache: CVMetalTextureCache?
    private var arBackgroundTexture: MTLTexture?
    /// World transform where the model is anchored in AR, set once when tracking is first available.
    private var arModelAnchor: matrix_float4x4?
    /// Placement distance (meters) ahead of the camera when the model is first anchored in AR.
    private static let arModelDistance: Float = 1.2

    /// User manipulation of the AR-anchored model, analogous to the orbit camera: pinch scales it,
    /// drag yaws/pitches it. `arActive` mirrors draw()'s AR state so the gesture handlers target the
    /// model instead of the orbit camera. Reset when AR is left (the model is re-placed on re-entry).
    private var arActive = false
    private var arModelYaw: Angle = .zero
    private var arModelPitch: Angle = .zero
    private var arModelScale: Float = 0.4
    private var pinchReferenceScale: Float = 0.4
    private static let arModelScaleRange: ClosedRange<Float> = 0.05...3.0
#endif

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

    /// Zoom target snapshotted at pinch start (orbit distance), so the cumulative gesture scale maps
    /// relative to it. The AR model-scale reference is `pinchReferenceScale`.
    private var pinchReferenceDistance: Float = Constants.cameraInitialDistance

    var drawableSize: CGSize = .zero

    /// Apply a drag (screen-point deltas): in AR it yaws/pitches the anchored model, otherwise it
    /// orbits the camera (and disables auto-rotation on first use).
    func orbitBy(dx: Float, dy: Float) {
#if os(iOS)
        if arActive {
            arModelYaw += Angle(radians: Double(dx * Constants.cameraOrbitRadiansPerPoint))
            let pitch = arModelPitch.radians + Double(dy * Constants.cameraOrbitRadiansPerPoint)
            arModelPitch = Angle(radians: min(max(pitch, -Constants.cameraPitchLimit.radians), Constants.cameraPitchLimit.radians))
            return
        }
#endif
        userHasInteracted = true
        cameraYaw += Angle(radians: Double(dx * Constants.cameraOrbitRadiansPerPoint))
        let newPitch = cameraPitch.radians + Double(dy * Constants.cameraOrbitRadiansPerPoint)
        cameraPitch = Angle(radians: min(max(newPitch, -Constants.cameraPitchLimit.radians), Constants.cameraPitchLimit.radians))
    }

    /// Snapshots the pinch target (orbit distance + AR model scale) at gesture start, so the
    /// cumulative gesture scale in `zoomBy` maps relative to it.
    func pinchBegan() {
        pinchReferenceDistance = cameraDistance
#if os(iOS)
        pinchReferenceScale = arModelScale
#endif
    }

    /// Apply a pinch: in AR it scales the anchored model, otherwise it dollies the orbit camera.
    /// `scale` is the gesture's cumulative scale relative to the `pinchBegan` snapshot.
    func zoomBy(scale: Float) {
#if os(iOS)
        if arActive {
            arModelScale = min(max(pinchReferenceScale * scale, Self.arModelScaleRange.lowerBound),
                               Self.arModelScaleRange.upperBound)
            return
        }
#endif
        userHasInteracted = true
        cameraDistance = min(max(pinchReferenceDistance / max(scale, 0.0001), Constants.cameraMinDistance), Constants.cameraMaxDistance)
    }

    init?(_ metalKitView: MTKView) {
        self.device = metalKitView.device!
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        guard let postProcessor = try? PostProcessor(device: device) else { return nil }
        self.postProcessor = postProcessor
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
                                          colorFormat: sceneColorFormat,
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
                colorFormat: sceneColorFormat,
                depthFormat: metalKitView.depthStencilPixelFormat,
                sampleCount: metalKitView.sampleCount,
                maxViewCount: 1,
                maxSimultaneousRenders: Constants.maxSimultaneousRenders)
            proceduralSplatController = controller
            modelRenderer = controller.splatRenderer
        case .sampleBox:
            modelRenderer = try! SampleBoxRenderer(device: device,
                                                   colorFormat: sceneColorFormat,
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

    /// (Re)allocates the offscreen linear-HDR target to match the current drawable size.
    private func ensureHDRTexture(width: Int, height: Int) {
        if let texture = hdrTexture, texture.width == width, texture.height == height { return }
        guard width > 0, height > 0 else { return }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: sceneColorFormat,
                                                                  width: width,
                                                                  height: height,
                                                                  mipmapped: false)
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .private
        let texture = device.makeTexture(descriptor: descriptor)
        texture?.label = "Scene HDR"
        hdrTexture = texture
    }

    /// Configures the drawable + layer for SDR (8-bit sRGB) or EDR (float, extended-range linear).
    /// EDR is what keeps highlights from clamping at 1.0 on capable displays; the post-process
    /// composite pipeline adapts automatically because it is keyed by the drawable's pixel format.
    private func configureDrawable(edr: Bool) {
        let layer = metalKitView.layer as? CAMetalLayer
        if edr {
            metalKitView.colorPixelFormat = .rgba16Float
            layer?.wantsExtendedDynamicRangeContent = true
            layer?.colorspace = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)
        } else {
            metalKitView.colorPixelFormat = .bgra8Unorm_srgb
            layer?.wantsExtendedDynamicRangeContent = false
            layer?.colorspace = CGColorSpace(name: CGColorSpace.sRGB)
        }
    }

    /// Maps the UI controls to display-only presentation settings for the post-process pass.
    /// (Stage A① only wires reference mode; exposure / tonemap / bloom / EDR are added in A②–A④.)
    private var presentationSettings: PostProcessor.Settings {
        var settings = PostProcessor.Settings()
        guard let controls = relightControls else { return settings }
        settings.referenceMode = controls.referenceMode
        settings.exposure = Float(pow(2.0, controls.exposureEV))
        settings.tonemap = controls.tonemap
        settings.bloomEnabled = controls.bloomEnabled
        settings.bloomThreshold = Float(controls.bloomThreshold)
        settings.bloomIntensity = Float(controls.bloomIntensity)
        settings.maxValue = controls.edrEnabled ? Float(controls.edrPeak) : 1.0
        return settings
    }

    func draw(in view: MTKView) {
        guard let modelRenderer, modelRenderer.isReadyToRender else { return }

        // Reconfigure the drawable for SDR/EDR before acquiring it, when the toggle changes.
        let wantEDR = relightControls?.edrEnabled ?? false
        if appliedEDR != wantEDR {
            configureDrawable(edr: wantEDR)
            appliedEDR = wantEDR
        }

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

        let drawableTexture = drawable.texture
        ensureHDRTexture(width: drawableTexture.width, height: drawableTexture.height)

        let splat = modelRenderer as? SplatRenderer

        // Full AR: when "AR / Room" is selected and a tracked camera frame is available, drive the
        // camera from ARKit (view/projection from ARCamera, model anchored in the world) and convert
        // the captured frame into the background the splat composites over. Otherwise the orbit camera
        // + skybox path is used (including while the AR session is still warming up).
        var activeViewport = viewport
        var arActive = false
#if os(iOS)
        var arFrame: ARFrame?
        if relightControls?.environmentChoice == .arRoom {
            arFrame = arProbeProvider?.currentFrame
        }
        if let arFrame,
           let arVP = arViewport(for: arFrame, drawableSize: CGSize(width: drawableTexture.width,
                                                                    height: drawableTexture.height)),
           let splat,
           let background = prepareARBackground(frame: arFrame, drawableTexture: drawableTexture,
                                                commandBuffer: commandBuffer) {
            activeViewport = arVP
            splat.arBackgroundTexture = background
            arActive = true
        } else {
            splat?.arBackgroundTexture = nil
        }
        self.arActive = arActive   // let the gesture handlers target the model while AR is rendering
#endif

        // Apply relighting controls to the splat renderer (if any).
        if let splat, let controls = relightControls {
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
            settings.arBackground = arActive   // composite over the live camera instead of the skybox
            splat.relightSettings = settings
        }

        // Pass 1: render the scene into the offscreen linear-HDR target. Relighting completes here in
        // linear HDR, untouched by any presentation.
        let didRender: Bool
        if let hdr = hdrTexture {
            do {
                didRender = try modelRenderer.render(viewports: [activeViewport],
                                                     colorTexture: hdr,
                                                     colorStoreAction: .store,
                                                     depthTexture: view.depthStencilTexture,
                                                     rasterizationRateMap: nil,
                                                     renderTargetArrayLength: 0,
                                                     to: commandBuffer)
            } catch {
                Self.log.error("Unable to render scene: \(error.localizedDescription)")
                didRender = false
            }
        } else {
            didRender = false
        }

        // Pass 2: single, separate presentation pass (linear HDR -> drawable). Reference mode and
        // every debug view bypass it for an exact copy.
        if didRender, let hdr = hdrTexture {
            postProcessor.encode(into: commandBuffer,
                                 source: hdr,
                                 destination: drawableTexture,
                                 debugMode: UInt32(relightControls?.debugMode ?? 0),
                                 settings: presentationSettings)
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
#if os(iOS)
        if choice == .arRoom {
            startARRoom(for: splat)
            return
        }
        // Switching away from AR/Room: pause the session so the camera isn't left running, and drop
        // the world anchor + user manipulation so the model is re-placed fresh if AR is re-entered.
        arProbeProvider?.stop()
        arModelAnchor = nil
        arModelYaw = .zero
        arModelPitch = .zero
        arModelScale = 0.4
        arActive = false
#endif
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

#if os(iOS)
    /// Starts (or resumes) the ARKit room-probe session and rebuilds the IBL environment from the
    /// room's environment cubemap whenever ARKit updates it. Until the first probe resolves the
    /// previously-applied environment stays in place; on devices without world tracking (e.g. the
    /// Simulator) it falls back to the procedural sky.
    private func startARRoom(for splat: SplatRenderer) {
        guard ARRoomProbeProvider.isSupported else {
            applyEnvironment(.procedural, to: splat)
            return
        }
        let provider = arProbeProvider ?? ARRoomProbeProvider()
        arProbeProvider = provider
        provider.onUpdate = { [weak self, weak splat] cubemap in
            guard let self, let splat else { return }
            // Throttle: the IBL precompute is GPU-heavy and ARKit can update the probe frequently.
            let now = Date()
            if let last = self.lastARProbeBuild,
               now.timeIntervalSince(last) < Self.arProbeRebuildInterval { return }
            self.lastARProbeBuild = now
            if let environment = try? IBLEnvironment(device: self.device, cubemap: cubemap) {
                splat.environment = environment
            }
        }
        provider.start()
    }

    /// Builds the AR viewport from the current ARCamera: world->camera view + ARKit projection (both
    /// oriented for the interface), with the model placed at a fixed world anchor (scaled, Z-up->Y-up).
    /// Returns nil until tracking is usable, so the model isn't anchored from a bad pose.
    private func arViewport(for frame: ARFrame, drawableSize: CGSize) -> ModelRendererViewportDescriptor? {
        guard drawableSize.width > 0, drawableSize.height > 0 else { return nil }
        if case .notAvailable = frame.camera.trackingState { return nil }
        let camera = frame.camera
        let orientation = currentInterfaceOrientation()

        // Place the model once: a fixed distance ahead of the initial camera, at eye height and
        // world-axis-aligned (upright regardless of how the device was tilted when AR started).
        if arModelAnchor == nil {
            let transform = camera.transform
            var forward = -SIMD3<Float>(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
            forward.y = 0
            let length = simd_length(forward)
            forward = length > 1e-4 ? forward / length : SIMD3<Float>(0, 0, -1)
            let origin = SIMD3<Float>(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
            let position = origin + forward * Self.arModelDistance
            var anchor = matrix_identity_float4x4
            anchor.columns.3 = SIMD4<Float>(position, 1)
            arModelAnchor = anchor
        }
        guard let anchor = arModelAnchor else { return nil }

        let viewWorld = camera.viewMatrix(for: orientation)   // world -> camera
        let projection = camera.projectionMatrix(for: orientation, viewportSize: drawableSize,
                                                  zNear: 0.01, zFar: 100)
        let upCalibration = matrix4x4_rotation(radians: -.pi / 2, axis: SIMD3<Float>(1, 0, 0))
        // User manipulation at the anchor: drag = yaw (about world up) + pitch, pinch = scale.
        let yaw = matrix4x4_rotation(radians: Float(arModelYaw.radians), axis: SIMD3<Float>(0, 1, 0))
        let pitch = matrix4x4_rotation(radians: Float(arModelPitch.radians), axis: SIMD3<Float>(1, 0, 0))
        let modelWorld = anchor * yaw * pitch * matrix4x4_scale(arModelScale) * upCalibration
        let viewMatrix = viewWorld * modelWorld

        let mtlViewport = MTLViewport(originX: 0, originY: 0,
                                      width: drawableSize.width, height: drawableSize.height,
                                      znear: 0, zfar: 1)
        return ModelRendererViewportDescriptor(viewport: mtlViewport,
                                               projectionMatrix: projection,
                                               viewMatrix: viewMatrix,
                                               screenSize: SIMD2(x: Int(drawableSize.width), y: Int(drawableSize.height)))
    }

    private func currentInterfaceOrientation() -> UIInterfaceOrientation {
        metalKitView.window?.windowScene?.interfaceOrientation ?? .portrait
    }

    /// Converts the captured frame's YCbCr planes into the linear-RGB background texture (sized to the
    /// drawable) and encodes that pass into `commandBuffer`, returning the texture (nil on failure).
    private func prepareARBackground(frame: ARFrame, drawableTexture: MTLTexture,
                                     commandBuffer: MTLCommandBuffer) -> MTLTexture? {
        guard let converter = cameraBackgroundConverter else { return nil }
        let pixelBuffer = frame.capturedImage
        guard CVPixelBufferGetPlaneCount(pixelBuffer) >= 2,
              let luma = cameraPlaneTexture(pixelBuffer, plane: 0, format: .r8Unorm),
              let chroma = cameraPlaneTexture(pixelBuffer, plane: 1, format: .rg8Unorm) else { return nil }
        ensureARBackgroundTexture(width: drawableTexture.width, height: drawableTexture.height)
        guard let destination = arBackgroundTexture else { return nil }

        // displayTransform maps image uv -> viewport uv; invert it for the shader (viewport -> image).
        let viewportSize = CGSize(width: drawableTexture.width, height: drawableTexture.height)
        let inverse = frame.displayTransform(for: currentInterfaceOrientation(),
                                             viewportSize: viewportSize).inverted()
        let displayToCamera = simd_float3x3(SIMD3<Float>(Float(inverse.a), Float(inverse.b), 0),
                                            SIMD3<Float>(Float(inverse.c), Float(inverse.d), 0),
                                            SIMD3<Float>(Float(inverse.tx), Float(inverse.ty), 1))
        converter.encode(into: commandBuffer, luma: luma, chroma: chroma,
                         displayToCamera: displayToCamera, destination: destination)
        return destination
    }

    private func ensureARBackgroundTexture(width: Int, height: Int) {
        if let texture = arBackgroundTexture, texture.width == width, texture.height == height { return }
        guard width > 0, height > 0 else { return }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: sceneColorFormat,
                                                                  width: width, height: height,
                                                                  mipmapped: false)
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .private
        let texture = device.makeTexture(descriptor: descriptor)
        texture?.label = "AR Camera Background"
        arBackgroundTexture = texture
    }

    /// Wraps one plane of an ARKit captured-image pixel buffer as an MTLTexture (no copy).
    private func cameraPlaneTexture(_ pixelBuffer: CVPixelBuffer, plane: Int,
                                    format: MTLPixelFormat) -> MTLTexture? {
        if cameraTextureCache == nil {
            CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cameraTextureCache)
        }
        guard let cache = cameraTextureCache else { return nil }
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, plane)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, plane)
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, pixelBuffer,
                                                               nil, format, width, height, plane, &cvTexture)
        guard status == kCVReturnSuccess, let cvTexture,
              let texture = CVMetalTextureGetTexture(cvTexture) else { return nil }
        return texture
    }
#endif
}

// MARK: - Relighting controls, environment, and UI

/// Observable relighting controls, shared between the SwiftUI control panel and the renderer.
@Observable
@MainActor
final class RelightControls {
    /// Selectable image-based-lighting environment. `studio` loads a bundled HDR panorama; reflections
    /// (and the Env-rotation slider) respond to whichever is chosen.
    enum EnvironmentChoice: Int, CaseIterable, Identifiable {
        case autoshop, studio, procedural, arRoom
        var id: Int { rawValue }
        var label: String {
            switch self {
            case .autoshop: return "Auto Shop"
            case .studio: return "Studio"
            case .procedural: return "Procedural"
            case .arRoom: return "AR / Room"
            }
        }
        /// Bundled equirect resource name, or nil for the procedural sky / runtime AR room probe.
        var resourceName: String? {
            switch self {
            case .autoshop: return "autoshop_env"
            case .studio: return "studio_env"
            case .procedural, .arRoom: return nil
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

    // MARK: Presentation (display-only post-process; never affects linear-HDR relighting)
    /// When true, bypass all presentation so the render matches the official Ref-Gaussian renderer
    /// numerically. Turn off (default) to enable exposure / tonemap / bloom / EDR.
    var referenceMode: Bool = false
    /// Exposure in stops (EV); applied as a linear 2^EV multiplier before tonemapping.
    var exposureEV: Double = 0
    /// Filmic tonemap operator applied to the composited linear-HDR image.
    var tonemap: PostProcessor.Tonemap = .aces
    /// HDR bloom (bright-pass + separable blur), added in linear HDR before tonemapping.
    var bloomEnabled: Bool = false
    /// Linear-HDR luminance above which pixels contribute to bloom.
    var bloomThreshold: Double = 1.0
    /// How strongly the blurred bloom is added back to the image.
    var bloomIntensity: Double = 0.4
    /// Output to an extended-dynamic-range drawable (rgba16Float + extended-linear colorspace) so
    /// highlights above 1.0 display brighter-than-white instead of clamping. Needs an EDR display.
    var edrEnabled: Bool = false
    /// EDR white point: the tonemap maps the brightest highlights to this multiple of SDR white.
    var edrPeak: Double = 2.0
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
    /// Collapsed by default so the panel doesn't cover the 3D view; tap the header to reveal controls.
    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header: tap to expand/collapse. When collapsed only this bar shows.
            Button {
                withAnimation(.easeInOut(duration: 0.2)) { isExpanded.toggle() }
            } label: {
                HStack {
                    Image(systemName: "slider.horizontal.3")
                    Text("Render Controls").font(.subheadline.bold())
                    Spacer()
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.up")
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            // Controls live in a height-capped scroll view so even when expanded the panel can't
            // cover the whole screen.
            if isExpanded {
                ScrollView {
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

                        Divider()

                        // Presentation (display-only): the reference toggle keeps an exact, ground-
                        // truth-comparable linear output; turning it off enables tonemap + exposure.
                        Toggle("Reference (linear)", isOn: $controls.referenceMode)
                            .font(.caption)
                        if !controls.referenceMode {
                            Picker("Tonemap", selection: $controls.tonemap) {
                                ForEach(PostProcessor.Tonemap.allCases) { op in
                                    Text(op.displayName).tag(op)
                                }
                            }
                            .pickerStyle(.segmented)
                            labeledSlider("Exposure", value: $controls.exposureEV, range: -4...4)

                            Toggle("Bloom", isOn: $controls.bloomEnabled)
                                .font(.caption)
                            if controls.bloomEnabled {
                                labeledSlider("Threshold", value: $controls.bloomThreshold, range: 0...3)
                                labeledSlider("Intensity", value: $controls.bloomIntensity, range: 0...1)
                            }

                            Toggle("EDR (extended range)", isOn: $controls.edrEnabled)
                                .font(.caption)
                            if controls.edrEnabled {
                                labeledSlider("EDR peak", value: $controls.edrPeak, range: 1...4)
                            }
                        }
                    }
                    .padding(.top, 2)
                }
                .frame(maxHeight: 280)
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

#if os(iOS)

// MARK: - AR room environment probe

/// Runs a minimal ARKit world-tracking session purely to obtain the room's lighting. With
/// `environmentTexturing = .automatic`, ARKit produces `AREnvironmentProbeAnchor`s whose
/// `environmentTexture` is an MTLTexture **cubemap** of the surrounding room; that cube is handed to
/// `IBLEnvironment` so the splat is lit by — and reflects — the real room. (Camera passthrough and
/// world anchoring are the next AR stage; this stage only borrows the room's light.)
@MainActor
final class ARRoomProbeProvider: NSObject, ARSessionDelegate {
    /// Whether this device supports the world tracking required for environment probes.
    static var isSupported: Bool { ARWorldTrackingConfiguration.isSupported }

    private let session = ARSession()
    private var isRunning = false

    /// Invoked on the main actor with the latest room environment cubemap each time ARKit updates it.
    var onUpdate: (@MainActor (MTLTexture) -> Void)?

    /// The latest ARKit frame (camera pose + captured image), read by the renderer each draw for full AR.
    var currentFrame: ARFrame? { session.currentFrame }

    func start() {
        guard Self.isSupported, !isRunning else { return }
        let configuration = ARWorldTrackingConfiguration()
        configuration.environmentTexturing = .automatic
        session.delegate = self
        session.run(configuration)
        isRunning = true
    }

    func stop() {
        guard isRunning else { return }
        session.pause()
        isRunning = false
    }

    // ARKit may call these off the main thread; hop to the main actor before delivering the texture.
    nonisolated func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        deliverLatestProbe(from: anchors)
    }
    nonisolated func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        deliverLatestProbe(from: anchors)
    }

    nonisolated private func deliverLatestProbe(from anchors: [ARAnchor]) {
        guard let texture = anchors
            .compactMap({ ($0 as? AREnvironmentProbeAnchor)?.environmentTexture })
            .last else { return }
        let boxed = SendableTexture(texture)
        Task { @MainActor in self.onUpdate?(boxed.texture) }
    }
}

/// Carries a non-Sendable Metal texture across the ARKit-callback → main-actor hop. Safe because the
/// texture is only ever read on the main actor and ARKit hands us a fresh, immutable cube each time.
private struct SendableTexture: @unchecked Sendable {
    let texture: MTLTexture
    init(_ texture: MTLTexture) { self.texture = texture }
}

#endif // os(iOS)

#endif // os(iOS) || os(macOS)
