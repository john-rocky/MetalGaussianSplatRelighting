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
    /// Where the model sits in the world: a point on a detected horizontal plane (floor/table) found
    /// by raycast. nil until placed — the model is hidden (camera only) until then.
    private var arFloorPosition: SIMD3<Float>?
    /// Model-space -> normalized transform (Z-up→Y-up, base on y=0, centered in XZ, longest footprint
    /// = 1 unit), built once from the loaded model's bounds so the real-meter scale is life-size.
    private var arModelNormalize: matrix_float4x4 = matrix_identity_float4x4
    /// Normalized footprint extents (world XZ, longest = 1), used to size the ground contact shadow.
    private var arModelFootprint: SIMD2<Float> = SIMD2(1, 1)

    /// User manipulation of the placed model: drag yaws it (turntable), pinch sets its real-world size.
    /// `arActive` mirrors draw()'s AR state so the gesture handlers target the model, not the orbit camera.
    private var arActive = false
    private var arModelYaw: Angle = .zero
    private var arModelPitch: Angle = .zero   // vertical-drag tilt, to stand up a sideways-loaded splat
    // Normalized model AABB (base at y=0), used to re-ground the model on the floor after the user
    // rotates it, so tilting it upright never sinks it through the floor.
    private var arModelAABBMin = SIMD3<Float>(-0.5, 0, -0.5)
    private var arModelAABBMax = SIMD3<Float>(0.5, 1, 0.5)
    /// Real-world size of the model in meters (its longest footprint dimension). Default ≈ a car length.
    private var arModelLengthMeters: Float = 4.5
    private var pinchReferenceLength: Float = 4.5
    private static let arModelLengthRange: ClosedRange<Float> = 0.3...15.0
    /// Turntable showcase: timestamp for the time-based auto-rotation, and its speed (rad/s).
    private var lastTurntableTimestamp: Date?
    private static let turntableSpeed: Float = 0.35

    /// Depth occlusion (LiDAR): whether scene depth was available this frame, and the (A, B) that map
    /// a splat's NDC depth to meters for comparison against the scene depth. Set in prepareARBackground.
    private var arOcclusionEnabled = false
    private var arDepthLinearize: SIMD2<Float> = .zero

    /// Soft elliptical contact shadow on the floor under the model, so it reads as grounded.
    private lazy var groundShadowRenderer: GroundShadowRenderer? =
        try? GroundShadowRenderer(device: device, colorFormat: sceneColorFormat)
    private static let shadowMargin: Float = 1.15      // ellipse slightly larger than the footprint
    private static let shadowStrength: Float = 0.55    // peak darkness under the model
    private static let shadowSoftness: Float = 1.6     // penumbra falloff
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

    /// User-adjustable tilt added to the fixed up-calibration (about the model's X axis). Lets a
    /// two-finger rotation gesture stand up a model whose up-axis convention differs from this
    /// viewer's (e.g. TripoSplat splats load on their side until twisted upright).
    var modelTiltAdjust: Angle = .zero

    /// Model-to-viewer up-axis calibration, chosen per file at load time. Blender / Ref-NeRF
    /// style .ply datasets are Z-up and need the -90° X rotation; Scaniverse .spz scenes are
    /// already Y-up and load rolled onto their side if the rotation is applied anyway.
    var upAxisCalibration = matrix4x4_rotation(radians: -.pi / 2, axis: SIMD3<Float>(1, 0, 0))

    /// Zoom target snapshotted at pinch start (orbit distance), so the cumulative gesture scale maps
    /// relative to it. The AR model-scale reference is `pinchReferenceScale`.
    private var pinchReferenceDistance: Float = Constants.cameraInitialDistance

    var drawableSize: CGSize = .zero

    /// Apply a drag (screen-point deltas): in AR it yaws/pitches the anchored model, otherwise it
    /// orbits the camera (and disables auto-rotation on first use).
    func orbitBy(dx: Float, dy: Float) {
#if os(iOS)
        if arActive {
            // Horizontal drag yaws; vertical drag tilts (pitch). The model is auto-re-grounded each
            // frame (rotatedMinY), so tilting a sideways splat upright never sinks it through the floor.
            // Two-finger twist adds roll (modelTiltAdjust).
            arModelYaw += Angle(radians: Double(dx * Constants.cameraOrbitRadiansPerPoint))
            arModelPitch += Angle(radians: Double(dy * Constants.cameraOrbitRadiansPerPoint))
            return
        }
#endif
        userHasInteracted = true
        cameraYaw += Angle(radians: Double(dx * Constants.cameraOrbitRadiansPerPoint))
        let newPitch = cameraPitch.radians + Double(dy * Constants.cameraOrbitRadiansPerPoint)
        cameraPitch = Angle(radians: min(max(newPitch, -Constants.cameraPitchLimit.radians), Constants.cameraPitchLimit.radians))
    }

    /// Snapshots the pinch target (orbit distance + AR model real size) at gesture start, so the
    /// cumulative gesture scale in `zoomBy` maps relative to it.
    func pinchBegan() {
        pinchReferenceDistance = cameraDistance
#if os(iOS)
        pinchReferenceLength = arModelLengthMeters
#endif
    }

    /// Apply a pinch: in AR it resizes the placed model (real-world meters), otherwise it dollies the
    /// orbit camera. `scale` is the gesture's cumulative scale relative to the `pinchBegan` snapshot.
    func zoomBy(scale: Float) {
#if os(iOS)
        if arActive {
            arModelLengthMeters = min(max(pinchReferenceLength * scale, Self.arModelLengthRange.lowerBound),
                                      Self.arModelLengthRange.upperBound)
            return
        }
#endif
        userHasInteracted = true
        cameraDistance = min(max(pinchReferenceDistance / max(scale, 0.0001), Constants.cameraMinDistance), Constants.cameraMaxDistance)
    }

    /// Two-finger rotation: adjust the model's up-tilt (about its X axis) so the user can stand a
    /// sideways-loaded splat upright. `delta` is the incremental gesture rotation in radians.
    func tiltBy(radians delta: Float) {
        userHasInteracted = true
        modelTiltAdjust += Angle(radians: Double(delta))
    }

    /// Places (or moves) the model onto the floor under a normalized screen point (0...1, top-left).
    /// Used by tap-to-place; auto-placement uses the screen center each frame until first placed.
    func placeAt(normalizedPoint: CGPoint) {
#if os(iOS)
        guard relightControls?.environmentChoice == .arRoom,
              let frame = arProbeProvider?.currentFrame else { return }
        _ = tryPlaceOnFloor(normalizedPoint: normalizedPoint, frame: frame)
#endif
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
            // Scaniverse .spz scenes carry Y pointing down (identity leaves them upside-down,
            // verified on-device); the 180° X rotation stands them upright.
            upAxisCalibration = url.pathExtension.lowercased() == "spz"
                ? matrix4x4_rotation(radians: .pi, axis: SIMD3<Float>(1, 0, 0))
                : matrix4x4_rotation(radians: -.pi / 2, axis: SIMD3<Float>(1, 0, 0))
            let reader = try AutodetectSceneReader(url)
            let points = try await reader.readAll()
#if os(iOS)
            // Robust model bounds (rejecting floater outliers) -> normalize transform + footprint, so
            // the model can be placed life-size on the floor in AR with a footprint-sized shadow.
            let placement = Self.modelPlacement(for: Self.robustModelBounds(points))
            arModelNormalize = placement.normalize
            arModelFootprint = placement.footprint
            arModelAABBMin = placement.aabbMin
            arModelAABBMax = placement.aabbMax
#endif
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
        case .captureAndTrain:
            // Routed to CaptureFlowView in ContentView's navigationDestination; never reached here.
            break
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

        // Up-axis calibration is chosen per file at load time (Z-up .ply vs Y-up .spz).
        let upCalibration = upAxisCalibration

        // Two-finger twist pitches the model about the screen's horizontal axis (eye-space X), so a
        // splat lying flat (up-axis toward the camera) can be tipped upright. Applied leftmost = eye space.
        let screenRoll = matrix4x4_rotation(radians: Float(modelTiltAdjust.radians), axis: SIMD3<Float>(1, 0, 0))

        let viewport = MTLViewport(originX: 0, originY: 0, width: drawableSize.width, height: drawableSize.height, znear: 0, zfar: 1)

        return ModelRendererViewportDescriptor(viewport: viewport,
                                               projectionMatrix: projectionMatrix,
                                               viewMatrix: screenRoll * translationMatrix * pitchMatrix * yawMatrix * upCalibration,
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
            FrameProbe.shared.record(commandBuffer)
        }

        updateRotation()
        proceduralSplatController?.update()

        let drawableTexture = drawable.texture
        ensureHDRTexture(width: drawableTexture.width, height: drawableTexture.height)

        let splat = modelRenderer as? SplatRenderer

        FrameProbe.shared.configure(
            splats: splat?.splatCount ?? 0,
            // The renderer's own value, not the control's: relighting only
            // actually runs once the IBL environment has finished building, so
            // the control alone would report work that is not happening yet.
            relight: splat?.relightSettings.isEnabled ?? false,
            environment: relightControls.map { String(describing: $0.environmentChoice) } ?? "none",
            width: drawableTexture.width,
            height: drawableTexture.height)

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
        if let arFrame {
            // Auto-place on the floor under the screen center until the user has a placement; after
            // that, tap-to-place (placeAt) moves it. The model stays hidden until a floor is found.
            if arFloorPosition == nil {
                _ = tryPlaceOnFloor(normalizedPoint: CGPoint(x: 0.5, y: 0.5), frame: arFrame)
            }
            // Turntable showcase: advance the model's yaw over time (frame-rate independent) so it
            // rotates hands-free; the contact shadow and reflections follow because yaw is applied first.
            if relightControls?.turntable == true {
                let now = Date()
                if let last = lastTurntableTimestamp {
                    arModelYaw += Angle(radians: Double(Self.turntableSpeed) * now.timeIntervalSince(last))
                }
                lastTurntableTimestamp = now
            } else {
                lastTurntableTimestamp = nil
            }
            if let arVP = arViewport(for: arFrame, drawableSize: CGSize(width: drawableTexture.width,
                                                                        height: drawableTexture.height)),
               let splat,
               let background = prepareARBackground(frame: arFrame, drawableTexture: drawableTexture,
                                                    commandBuffer: commandBuffer) {
                // Darken the floor under the car (on the camera bg) before the splats composite over it.
                encodeGroundShadow(frame: arFrame, into: commandBuffer, destination: background)
                activeViewport = arVP
                splat.arBackgroundTexture = background
                arActive = true
            } else {
                splat?.arBackgroundTexture = nil
            }
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
#if os(iOS)
            settings.occlusionEnabled = arActive && arOcclusionEnabled   // hide splats behind real geometry
            settings.depthLinearize = arDepthLinearize
#endif
            settings.tintColor = controls.tintColor
            settings.tintStrength = Float(controls.tintStrength)
            settings.asgIndirectEnabled = controls.asgIndirectEnabled
            settings.asgIndirectStrength = Float(controls.asgIndirectStrength)
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
        // the placement + user manipulation so the model is re-placed fresh if AR is re-entered.
        arProbeProvider?.stop()
        arFloorPosition = nil
        arModelYaw = .zero
        arModelPitch = .zero
        modelTiltAdjust = .zero
        arModelLengthMeters = 4.5
        lastTurntableTimestamp = nil
        arActive = false
        arOcclusionEnabled = false
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

    /// Builds the AR viewport from the current ARCamera (world->camera view + ARKit projection, both
    /// oriented for the interface). The model is normalized to life-size, placed on the detected floor
    /// (`arFloorPosition`), sized in meters, and yawed by the user. Returns nil until placed/tracking.
    private func arViewport(for frame: ARFrame, drawableSize: CGSize) -> ModelRendererViewportDescriptor? {
        guard drawableSize.width > 0, drawableSize.height > 0 else { return nil }
        guard let floor = arFloorPosition else { return nil }   // not placed on the floor yet
        if case .notAvailable = frame.camera.trackingState { return nil }
        let camera = frame.camera
        let orientation = currentInterfaceOrientation()

        let viewWorld = camera.viewMatrix(for: orientation)   // world -> camera
        let projection = camera.projectionMatrix(for: orientation, viewportSize: drawableSize,
                                                  zNear: 0.01, zFar: 100)
        // Place the normalized model (base-centered on its own origin) on the floor point, size it in
        // real meters, and spin it by the user's yaw. arModelNormalize already maps Z-up->Y-up.
        let floorAnchor = matrix4x4_translation(floor.x, floor.y, floor.z)
        let yaw = matrix4x4_rotation(radians: Float(arModelYaw.radians), axis: SIMD3<Float>(0, 1, 0))
        let pitch = matrix4x4_rotation(radians: Float(arModelPitch.radians), axis: SIMD3<Float>(1, 0, 0))
        let roll = matrix4x4_rotation(radians: Float(modelTiltAdjust.radians), axis: SIMD3<Float>(0, 0, 1))
        let orient = pitch * roll
        // Drop the oriented model so its rotated AABB bottom rests on the floor (no sink-through).
        let reGround = matrix4x4_translation(0, -Self.rotatedMinY(orient, arModelAABBMin, arModelAABBMax), 0)
        let modelWorld = floorAnchor * yaw * matrix4x4_scale(arModelLengthMeters) * reGround * orient * arModelNormalize
        let viewMatrix = viewWorld * modelWorld

        let mtlViewport = MTLViewport(originX: 0, originY: 0,
                                      width: drawableSize.width, height: drawableSize.height,
                                      znear: 0, zfar: 1)
        return ModelRendererViewportDescriptor(viewport: mtlViewport,
                                               projectionMatrix: projection,
                                               viewMatrix: viewMatrix,
                                               screenSize: SIMD2(x: Int(drawableSize.width), y: Int(drawableSize.height)))
    }

    /// Darkens the camera-background floor under the placed model with a soft elliptical contact
    /// shadow (sized to the model footprint, oriented by yaw, at the floor anchor), so it looks
    /// grounded. Encoded after the camera conversion and before the splat resolve composites over it.
    private func encodeGroundShadow(frame: ARFrame, into commandBuffer: MTLCommandBuffer, destination: MTLTexture) {
        guard let shadow = groundShadowRenderer, let floor = arFloorPosition,
              drawableSize.width > 0, drawableSize.height > 0 else { return }
        let orientation = currentInterfaceOrientation()
        let view = frame.camera.viewMatrix(for: orientation)
        let projection = frame.camera.projectionMatrix(for: orientation, viewportSize: drawableSize,
                                                       zNear: 0.01, zFar: 100)
        let yaw = matrix4x4_rotation(radians: Float(arModelYaw.radians), axis: SIMD3<Float>(0, 1, 0))
        let semiX = max(arModelFootprint.x, 0.01) * arModelLengthMeters * 0.5 * Self.shadowMargin
        let semiZ = max(arModelFootprint.y, 0.01) * arModelLengthMeters * 0.5 * Self.shadowMargin
        let footprintScale = matrix_float4x4(columns: (SIMD4<Float>(semiX, 0, 0, 0),
                                                       SIMD4<Float>(0, 1, 0, 0),
                                                       SIMD4<Float>(0, 0, semiZ, 0),
                                                       SIMD4<Float>(0, 0, 0, 1)))
        let mvp = projection * view * matrix4x4_translation(floor.x, floor.y, floor.z) * yaw * footprintScale
        shadow.encode(into: commandBuffer, mvp: mvp,
                      strength: Self.shadowStrength, softness: Self.shadowSoftness, destination: destination)
    }

    /// Raycasts a world ray through a normalized screen point against detected/estimated horizontal
    /// planes; on a hit, sets `arFloorPosition` to that floor point. Returns whether it placed.
    private func tryPlaceOnFloor(normalizedPoint: CGPoint, frame: ARFrame) -> Bool {
        guard let session = arProbeProvider?.arSession, drawableSize.width > 0, drawableSize.height > 0 else { return false }
        if case .notAvailable = frame.camera.trackingState { return false }
        let orientation = currentInterfaceOrientation()
        let view = frame.camera.viewMatrix(for: orientation)
        let projection = frame.camera.projectionMatrix(for: orientation, viewportSize: drawableSize,
                                                       zNear: 0.01, zFar: 100)
        let invVP = (projection * view).inverse
        let ndcX = Float(normalizedPoint.x) * 2 - 1
        let ndcY = 1 - Float(normalizedPoint.y) * 2
        let nearH = invVP * SIMD4<Float>(ndcX, ndcY, 0, 1)   // Metal NDC near z = 0
        let farH  = invVP * SIMD4<Float>(ndcX, ndcY, 1, 1)   //            far  z = 1
        let near = SIMD3<Float>(nearH.x, nearH.y, nearH.z) / nearH.w
        let far  = SIMD3<Float>(farH.x, farH.y, farH.z) / farH.w
        let direction = simd_normalize(far - near)
        let query = ARRaycastQuery(origin: near, direction: direction,
                                   allowing: .estimatedPlane, alignment: .horizontal)
        guard let hit = session.raycast(query).first else { return false }
        let t = hit.worldTransform
        arFloorPosition = SIMD3<Float>(t.columns.3.x, t.columns.3.y, t.columns.3.z)
        return true
    }

    /// Outlier-robust model bounds (mean ± 3σ filter) in model space, for life-size AR placement —
    /// rejects scene-sized floater splats that would otherwise blow up the bounding box.
    private static func robustModelBounds(_ points: [SplatPoint]) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        guard !points.isEmpty else { return (SIMD3<Float>(repeating: -1), SIMD3<Float>(repeating: 1)) }
        let count = Float(points.count)
        var mean = SIMD3<Float>(repeating: 0)
        for p in points { mean += p.position }
        mean /= count
        var variance = SIMD3<Float>(repeating: 0)
        for p in points { let d = p.position - mean; variance += d * d }
        variance /= count
        let sigma = SIMD3<Float>(variance.x.squareRoot(), variance.y.squareRoot(), variance.z.squareRoot())
        let lo = mean - 3 * sigma
        let hi = mean + 3 * sigma
        var mn = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var mx = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in points {
            let q = p.position
            if q.x < lo.x || q.x > hi.x || q.y < lo.y || q.y > hi.y || q.z < lo.z || q.z > hi.z { continue }
            mn = simd_min(mn, q); mx = simd_max(mx, q)
        }
        return mn.x <= mx.x ? (mn, mx) : (lo, hi)
    }

    /// Builds the model-space -> normalized transform (Z-up→Y-up, base resting on y=0, centered in XZ,
    /// longest footprint dimension = 1 unit, so `arModelLengthMeters` reads as meters) plus the
    /// normalized footprint extents (world XZ) used to size the contact shadow.
    private static func modelPlacement(for bounds: (min: SIMD3<Float>, max: SIMD3<Float>))
        -> (normalize: matrix_float4x4, footprint: SIMD2<Float>, aabbMin: SIMD3<Float>, aabbMax: SIMD3<Float>) {
        let r = matrix4x4_rotation(radians: -.pi / 2, axis: SIMD3<Float>(1, 0, 0))   // Z-up -> Y-up
        // AABB after r maps model (x,y,z) -> world' (x, z, -y).
        let mn = bounds.min, mx = bounds.max
        let xMin = mn.x, xMax = mx.x        // world' x
        let yMin = mn.z, yMax = mx.z        // world' y (= model z)
        let zMin = -mx.y, zMax = -mn.y      // world' z (= -model y)
        let extX = xMax - xMin
        let extY = yMax - yMin
        let extZ = zMax - zMin
        let horizontal = max(extX, extZ)
        let normScale = horizontal > 1e-5 ? 1.0 / horizontal : 1.0
        let translate = matrix4x4_translation(-(xMin + xMax) * 0.5, -yMin, -(zMin + zMax) * 0.5)
        let normalize = matrix4x4_scale(normScale) * translate * r
        // Normalized AABB: centered in x/z, base at y=0, used to re-ground after user rotation.
        let aabbMin = SIMD3<Float>(-extX * normScale * 0.5, 0, -extZ * normScale * 0.5)
        let aabbMax = SIMD3<Float>(extX * normScale * 0.5, extY * normScale, extZ * normScale * 0.5)
        return (normalize, SIMD2<Float>(extX * normScale, extZ * normScale), aabbMin, aabbMax)
    }

    /// Lowest y of an AABB after applying a rotation — used to drop the rotated model onto the floor.
    private static func rotatedMinY(_ m: matrix_float4x4, _ mn: SIMD3<Float>, _ mx: SIMD3<Float>) -> Float {
        var minY = Float.greatestFiniteMagnitude
        for x in [mn.x, mx.x] {
            for y in [mn.y, mx.y] {
                for z in [mn.z, mx.z] {
                    minY = min(minY, (m * SIMD4<Float>(x, y, z, 1)).y)
                }
            }
        }
        return minY
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

        // Scene depth (LiDAR) for occlusion; nil on devices without it. Packed into the bg alpha.
        let depthMap = frame.smoothedSceneDepth?.depthMap ?? frame.sceneDepth?.depthMap
        let sceneDepth = depthMap.flatMap { depthTexture(from: $0) }
        arOcclusionEnabled = (sceneDepth != nil)
        // (A, B) so the resolve maps a splat's NDC depth to meters: depth = B / (A - ndc).
        let projection = frame.camera.projectionMatrix(for: currentInterfaceOrientation(),
                                                       viewportSize: viewportSize, zNear: 0.01, zFar: 100)
        let wz = projection.columns.2.w
        arDepthLinearize = abs(wz) > 1e-6 ? SIMD2(projection.columns.2.z / wz, projection.columns.3.z / wz) : .zero

        converter.encode(into: commandBuffer, luma: luma, chroma: chroma, sceneDepth: sceneDepth,
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

    /// Wraps an ARKit scene-depth buffer (single-plane Float32 meters) as an `.r32Float` MTLTexture.
    private func depthTexture(from pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        if cameraTextureCache == nil {
            CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cameraTextureCache)
        }
        guard let cache = cameraTextureCache else { return nil }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, pixelBuffer,
                                                               nil, .r32Float, width, height, 0, &cvTexture)
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
    /// Configurator repaint: paint color (linear RGB) and strength (0 = original color, 1 = full repaint).
    var tintColor: SIMD3<Float> = SIMD3(1, 1, 1)
    var tintStrength: Double = 0
    /// Ref-Gaussian ASG indirect (Stage D): consume the trained `ind_asg` tail (160 floats per
    /// splat) for per-splat secondary reflections. Only takes effect on scenes that carry the tail.
    var asgIndirectEnabled: Bool = false
    /// 0..1 blend factor between the IBL environment specular (direct) and the ASG indirect.
    /// Stands in for the paper's ray-traced visibility test that isn't feasible on-device.
    var asgIndirectStrength: Double = 0.5
    /// Hands-free showcase: slowly auto-rotate the AR-placed model (turntable). Drag still adjusts it.
    var turntable: Bool = false
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

/// A selectable paint color for the configurator (nil `linear` restores the original color).
private struct ConfiguratorPaint: Identifiable {
    let id = UUID()
    let name: String
    let swatch: Color
    let linear: SIMD3<Float>?
}

/// A selectable finish for the configurator; drives the roughness / reflection overrides.
private struct ConfiguratorFinish: Identifiable {
    let id = UUID()
    let name: String
    let trained: Bool
    let roughness: Double
    let reflection: Double
}

private let configuratorPaints: [ConfiguratorPaint] = [
    .init(name: "Original", swatch: .gray, linear: nil),
    .init(name: "Red", swatch: .red, linear: SIMD3(0.55, 0.02, 0.02)),
    .init(name: "Blue", swatch: .blue, linear: SIMD3(0.02, 0.06, 0.5)),
    .init(name: "Green", swatch: .green, linear: SIMD3(0.03, 0.3, 0.06)),
    .init(name: "Black", swatch: .black, linear: SIMD3(0.015, 0.015, 0.015)),
    .init(name: "White", swatch: .white, linear: SIMD3(0.8, 0.8, 0.82)),
    .init(name: "Silver", swatch: Color(white: 0.75), linear: SIMD3(0.6, 0.6, 0.62)),
    .init(name: "Yellow", swatch: .yellow, linear: SIMD3(0.6, 0.45, 0.02)),
]

private let configuratorFinishes: [ConfiguratorFinish] = [
    .init(name: "Original", trained: true, roughness: 0.4, reflection: 0.5),
    .init(name: "Matte", trained: false, roughness: 0.9, reflection: 0.05),
    .init(name: "Glossy", trained: false, roughness: 0.25, reflection: 0.4),
    .init(name: "Mirror", trained: false, roughness: 0.02, reflection: 0.95),
    .init(name: "Metallic", trained: false, roughness: 0.15, reflection: 0.9),
]

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

                            // Configurator: repaint color + finish preset ("try your variant in your room").
                            Text("Configurator").font(.caption.bold())
                            HStack(spacing: 6) {
                                ForEach(configuratorPaints) { paint in
                                    Button {
                                        if let linear = paint.linear {
                                            controls.tintColor = linear
                                            controls.tintStrength = 1
                                        } else {
                                            controls.tintStrength = 0
                                        }
                                    } label: {
                                        Circle()
                                            .fill(paint.swatch)
                                            .frame(width: 22, height: 22)
                                            .overlay(Circle().stroke(.white.opacity(0.5), lineWidth: 1))
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                            HStack(spacing: 6) {
                                ForEach(configuratorFinishes) { finish in
                                    Button(finish.name) {
                                        controls.useTrainedMaterial = finish.trained
                                        controls.roughness = finish.roughness
                                        controls.reflectionStrength = finish.reflection
                                    }
                                    .font(.caption2)
                                    .buttonStyle(.bordered)
                                }
                            }

                            Toggle("Turntable (AR)", isOn: $controls.turntable)
                                .font(.caption)

                            // Ref-Gaussian ASG indirect: consume the trained 160-float ind_asg
                            // tail to add per-splat secondary reflections (the paper's "indirect
                            // light" head). Slider stands in for the ray-traced visibility test.
                            Toggle("ASG indirect", isOn: $controls.asgIndirectEnabled)
                                .font(.caption)
                            if controls.asgIndirectEnabled {
                                labeledSlider("Indirect mix", value: $controls.asgIndirectStrength, range: 0...1)
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
    /// The underlying session, exposed so the renderer can raycast against detected planes for placement.
    var arSession: ARSession { session }

    func start() {
        guard Self.isSupported, !isRunning else { return }
        let configuration = ARWorldTrackingConfiguration()
        configuration.environmentTexturing = .automatic
        configuration.planeDetection = [.horizontal]   // floor/table detection for placing the model
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            configuration.frameSemantics.insert(.smoothedSceneDepth)   // LiDAR depth for occlusion
        }
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

/// Frame timing for the renderer, reported to the console.
///
/// The README calls this renderer real-time without ever saying at what. GPU
/// time comes from the command buffer's own clock rather than a wall-clock
/// difference, so it measures the work rather than how often the display asked
/// for it — a frame that finishes in 4 ms still presents at the 8.3 ms the
/// 120 Hz display allows, and only the first number says whether there is
/// headroom left.
final class FrameProbe: @unchecked Sendable {
    static let shared = FrameProbe()

    /// What the on-screen overlay shows, so a screen recording carries its own
    /// evidence rather than needing the console alongside it.
    @MainActor
    @Observable
    final class Readout {
        var line = ""
    }

    @MainActor static let readout = Readout()

    /// Frames per report. Two seconds or so at 60 Hz, long enough to average
    /// out a stray hitch and short enough to catch a change in the scene.
    private let window = 120
    /// The overlay updates far more often than the console does; a quarter of a
    /// second is slow enough to read and fast enough to follow a moving camera.
    private let overlayWindow = 15

    private let lock = NSLock()
    private var gpuSeconds: [Double] = []
    private var cpuSeconds: [Double] = []
    private var lastPresented: CFTimeInterval?
    private var recent: [Double] = []
    private var recentIntervals: [CFTimeInterval] = []
    private var label = ""

    /// Describe what is being rendered. Changing this flushes the current
    /// window, so a report never straddles two configurations.
    func configure(splats: Int, relight: Bool, environment: String, width: Int, height: Int) {
        let next = "splats=\(splats) relight=\(relight ? 1 : 0) env=\(environment) " +
                   "size=\(width)x\(height)"
        lock.lock()
        defer { lock.unlock() }
        guard next != label else { return }
        label = next
        gpuSeconds.removeAll(keepingCapacity: true)
        cpuSeconds.removeAll(keepingCapacity: true)
        recent.removeAll(keepingCapacity: true)
        recentIntervals.removeAll(keepingCapacity: true)
        lastPresented = nil
    }

    func record(_ commandBuffer: MTLCommandBuffer) {
        let gpu = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        guard gpu > 0 else { return }   // zero until the buffer has actually run
        let now = CACurrentMediaTime()

        lock.lock()
        var report: String?
        var overlay: String?
        gpuSeconds.append(gpu)
        if let last = lastPresented { cpuSeconds.append(now - last) }
        lastPresented = now
        recent.append(gpu)
        recentIntervals.append(now)
        if recent.count >= overlayWindow {
            overlay = overlayLine()
            recent.removeAll(keepingCapacity: true)
            recentIntervals.removeAll(keepingCapacity: true)
        }
        if gpuSeconds.count >= window {
            report = summary()
            gpuSeconds.removeAll(keepingCapacity: true)
            cpuSeconds.removeAll(keepingCapacity: true)
        }
        lock.unlock()

        if let report { print(report) }
        if let overlay {
            Task { @MainActor in Self.readout.line = overlay }
        }
    }

    /// Caller holds the lock.
    private func overlayLine() -> String {
        let gpu = recent.sorted()
        let median = gpu[gpu.count / 2] * 1000
        var fps = 0.0
        if let first = recentIntervals.first, let last = recentIntervals.last, last > first {
            fps = Double(recentIntervals.count - 1) / (last - first)
        }
        return String(format: "%@  ·  %.1f ms GPU  ·  %.0f fps", label, median, fps)
    }

    /// Caller holds the lock.
    private func summary() -> String {
        let gpu = gpuSeconds.sorted()
        let frame = cpuSeconds.sorted()
        func ms(_ xs: [Double], _ q: Double) -> Double {
            guard !xs.isEmpty else { return 0 }
            return xs[min(xs.count - 1, Int(q * Double(xs.count)))] * 1000
        }
        let interval = ms(frame, 0.5)
        return String(
            format: "RENDERPERF %@ gpu_ms=%.2f gpu_p95_ms=%.2f frame_ms=%.2f fps=%.1f n=%d",
            label, ms(gpu, 0.5), ms(gpu, 0.95), interval,
            interval > 0 ? 1000 / interval : 0, gpu.count)
    }
}

#endif // os(iOS) || os(macOS)
