#if os(iOS)
import ARKit
import SceneKit
import SwiftUI

/// Live AR camera feed, sharing the capture controller's ARSession so the preview and the captured
/// poses come from the same tracking state. An empty SceneKit scene renders only the passthrough.
private struct ARPreviewView: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView()
        view.session = session
        view.scene = SCNScene()
        view.automaticallyUpdatesLighting = true
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

/// End-to-end on-device pipeline: capture keyframes by tapping, train a Gaussian splat from them,
/// then hand the resulting PLY to the existing viewer by appending to the navigation path.
struct CaptureFlowView: View {
    let mode: CaptureMode
    @Binding var navigationPath: NavigationPath

    @StateObject private var capture = ARCaptureController()
    @StateObject private var training: GaussianTrainingController
    @Environment(\.dismiss) private var dismiss

    /// Object mode tuning is dominated by iPhone process-memory cap (~1.5-2GB without the
    /// increased-memory entitlement). The combination of (a) 30+ sensor-native frames held as
    /// float images, (b) per-frame masks at the same dims, and (c) a Gaussian splat state +
    /// Adam-optimizer scratch that grows with each densification round runs us right up to
    /// the limit. We chose:
    ///
    ///   * iters = 8000 → stopSplitAt = 4000 → ~34 densification rounds (up from ~24 at 6000
    ///     iters). The extra 10 rounds let needle-shaped splats keep splitting into rounder
    ///     children instead of being frozen as anisotropic streaks; without those rounds the
    ///     reconstruction stalls at iter 3000 with ~48k under-converged splats. 8000 also
    ///     fires the opacity reset twice (iters 3100 and 6100), leaving 1900 iters of pure
    ///     refinement after the last reset — enough for the inside-object splats to recover
    ///     their opacity and the outside-noise splats to fade. Going to 10000 iters fires a
    ///     third reset at 9100 with only 900 iters left to recover, which actually leaves
    ///     genuine splats unfinished.
    ///   * imageDownscale = 3.0 (vs room's 1.5) cuts image + mask CPU/GPU memory by ~75%
    ///     (~250MB → ~70MB for 30 sensor-native frames). The trainer still has plenty of
    ///     resolution to converge object detail thanks to msplat's coarse-to-fine schedule.
    ///
    /// Room mode runs 6000 iters at downscale 1.5 — bumped from 3000 / 2.0 once the LiDAR mesh
    /// init cloud started seeding the trainer with ~20-50k well-placed splats instead of the
    /// 50-200 sparse feature points; that init density supports a longer densification schedule
    /// (more rounds before opacity reset) and the lower downscale recovers wall-texture detail
    /// the trainer was previously losing. Room doesn't carry masks so the memory budget
    /// comfortably absorbs the extra image resolution.
    init(mode: CaptureMode, navigationPath: Binding<NavigationPath>) {
        self.mode = mode
        self._navigationPath = navigationPath
        let iters = (mode == .object) ? 8_000 : 6_000
        let downscale: Float = (mode == .object) ? 3.0 : 1.5
        self._training = StateObject(wrappedValue: GaussianTrainingController(
            totalIterations: iters, imageDownscale: downscale))
    }

    @State private var isMasking = false
    @State private var maskingProgress = 0
    @State private var maskingTotal = 0
    @State private var maskedPreview: UIImage?
    @State private var keptFrames = 0
    @State private var initPointCount = 0
    @State private var initPointSource: String = ""

    /// Below this, there are too few views for a usable reconstruction.
    private let minKeyframes = 10

    var body: some View {
        ZStack {
            if ARCaptureController.isSupported {
                ARPreviewView(session: capture.session)
                    .ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
                Text("AR world tracking isn't supported on this device.")
                    .foregroundStyle(.white)
                    .padding()
            }

            if isMasking {
                maskingOverlay
            } else {
                switch training.phase {
                case .idle:
                    captureOverlay
                case .failed(let message):
                    failureOverlay(message)
                default:
                    trainingOverlay
                }
            }
        }
        .navigationBarBackButtonHidden(training.phase != .idle || isMasking)
        .onAppear { capture.start() }
        .onDisappear { capture.stop() }
        .onChange(of: training.phase) { _, phase in
            if case .finished(let url) = phase {
                navigationPath.append(ModelIdentifier.gaussianSplat(url))
            }
        }
    }

    // MARK: Capture

    private var captureOverlay: some View {
        VStack {
            Text(mode == .object
                 ? "Orbit the object slowly at several heights — aim for 30+ frames from many angles."
                 : "Walk slowly around the room — frames are captured automatically. Tap the shutter for extra coverage.")
                .font(.subheadline)
                .foregroundStyle(.white)
                .padding(8)
                .background(.black.opacity(0.5), in: .rect(cornerRadius: 8))
                .padding(.top)

            Spacer()

            HStack(alignment: .center, spacing: 24) {
                thumbnail

                Button(action: { capture.captureKeyframe() }) {
                    Circle()
                        .fill(.white)
                        .frame(width: 70, height: 70)
                        .overlay(Circle().stroke(.white, lineWidth: 4).padding(4))
                }

                trainButton
            }
            .padding(.bottom, 32)
        }
    }

    private var thumbnail: some View {
        Group {
            if let image = capture.lastThumbnail {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Color.white.opacity(0.15)
            }
        }
        .frame(width: 56, height: 56)
        .clipShape(.rect(cornerRadius: 8))
        .overlay(alignment: .bottomTrailing) {
            Text("\(capture.keyframeCount)")
                .font(.caption.bold())
                .foregroundStyle(.white)
                .padding(4)
                .background(.black.opacity(0.6), in: .capsule)
                .padding(2)
        }
    }

    private var trainButton: some View {
        Button(action: beginTraining) {
            Text("Train")
                .font(.headline)
                .frame(width: 56, height: 56)
                .background(capture.keyframeCount >= minKeyframes ? .blue : .gray, in: .circle)
                .foregroundStyle(.white)
        }
        .disabled(capture.keyframeCount < minKeyframes)
    }

    // MARK: Training

    private var trainingOverlay: some View {
        VStack(spacing: 16) {
            ProgressView(value: training.progress) {
                Text(training.phase == .preparing ? "Preparing dataset…" : "Training")
            }
            .tint(.white)

            Text("Iteration \(training.iteration) / \(training.totalIterations)  ·  \(training.splatCount) splats")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.white)

            if keptFrames > 0 {
                Text("\(keptFrames) frames · \(initPointCount) init points (\(initPointSource))")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.white.opacity(0.75))
            }

            Button("Stop & use current result") { training.cancel() }
                .buttonStyle(.bordered)
                .tint(.white)
        }
        .padding(24)
        .background(.black.opacity(0.7), in: .rect(cornerRadius: 16))
        .padding(40)
    }

    private func failureOverlay(_ message: String) -> some View {
        VStack(spacing: 16) {
            Text(message)
                .multilineTextAlignment(.center)
                .foregroundStyle(.white)
            Button("Go Back") { dismiss() }
                .buttonStyle(.borderedProminent)
        }
        .padding(24)
        .background(.black.opacity(0.7), in: .rect(cornerRadius: 16))
        .padding(40)
    }

    private var maskingOverlay: some View {
        VStack(spacing: 12) {
            if let preview = maskedPreview {
                Image(uiImage: preview)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: 220, maxHeight: 220)
                    .clipShape(.rect(cornerRadius: 8))
            } else {
                ProgressView().tint(.white)
                    .frame(height: 60)
            }
            Text("Isolating object…")
                .foregroundStyle(.white)
            Text("\(maskingProgress) / \(maskingTotal)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.white)
        }
        .padding(20)
        .background(.black.opacity(0.7), in: .rect(cornerRadius: 16))
        .padding(40)
    }

    private func beginTraining() {
        let kfs = capture.keyframes
        // Build the room init cloud BEFORE pausing the session — `collectRoomInitCloud` reads
        // `session.currentFrame.anchors` for the LiDAR mesh, which is only populated while the
        // session is live. Sparse + mesh + per-keyframe depth voxelized together.
        let roomInit = capture.collectRoomInitCloud()
        capture.stop()
        let workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("capture-\(UUID().uuidString)", isDirectory: true)
        let outputPLY = FileManager.default.temporaryDirectory
            .appendingPathComponent("scene-\(UUID().uuidString).ply")

        guard mode == .object else {
            // Room mode: hand the LiDAR-augmented init cloud to the trainer. Falls back to
            // sparse-only on non-LiDAR devices because `collectRoomInitCloud` degrades to just
            // featurePoints when the mesh + depth sources are empty.
            keptFrames = kfs.count
            initPointCount = roomInit.points.count
            initPointSource = roomInit.source
            training.start(keyframes: kfs, points: roomInit.points,
                           workDir: workDir, outputPLY: outputPLY)
            return
        }
        // Object mode below keeps using the sparse cloud as a fallback when no LiDAR is
        // available; ObjectMaskProcessor still drives the dense in-object seed.
        let pts = capture.collectedPoints

        // Object mode: isolate the foreground in each frame and bake a white background, then
        // train with bgColor=white so empty regions match the masked input — splats only grow
        // on the object (NeRF-synthetic recipe). Frames where no foreground is detected are
        // dropped, because mixing masked and unmasked frames breaks the consistent-background
        // assumption the trainer relies on.
        isMasking = true
        maskingProgress = 0
        maskingTotal = kfs.count
        maskedPreview = nil
        Task {
            var masked: [CaptureKeyframe] = []
            var unprojected: [simd_float3] = []
            masked.reserveCapacity(kfs.count)
            for kf in kfs {
                let result = await Task.detached(priority: .userInitiated) {
                    ObjectMaskProcessor.mask(kf, bgR: 1, bgG: 1, bgB: 1)
                }.value
                if let result {
                    masked.append(result.keyframe)
                    unprojected.append(contentsOf: result.points)
                    // Show the latest mask as a live diagnostic so the user can spot Vision
                    // picking the wrong thing (a table edge instead of the object) early. JPEG
                    // bytes are sensor-native landscape (no EXIF), so wrap with the keyframe's
                    // orientation to display the right way up for the device's hold.
                    if let cg = UIImage(data: result.keyframe.jpeg)?.cgImage {
                        maskedPreview = UIImage(cgImage: cg, scale: 1, orientation: result.keyframe.orientation)
                    }
                }
                maskingProgress += 1
            }
            // On LiDAR devices we now have a dense in-object cloud; voxelise to a few thousand
            // unique init points. On non-LiDAR devices `unprojected` is empty and we fall back
            // to ARKit's sparse rawFeaturePoints — mostly on background surfaces, but still
            // better than random initialisation.
            let initPoints: [simd_float3]
            if unprojected.isEmpty {
                initPoints = pts
                initPointSource = "sparse features"
            } else {
                // 1 cm voxel matches the LiDAR depth noise floor and seeds the trainer with
                // ~10k well-placed object-surface points (vs. ~1.7k at 2cm). Standard 3DGS
                // expects 100k+ init splats from COLMAP; 10k is the on-device pragmatic point
                // — enough density for a 6k-iter run to crystallise into recognisable geometry,
                // small enough that densification stays well within memory limits now that the
                // per-frame buffer-churn OOM is fixed.
                initPoints = ObjectMaskProcessor.voxelize(unprojected, voxelSize: 0.01)
                initPointSource = "LiDAR"
            }
            keptFrames = masked.count
            initPointCount = initPoints.count
            isMasking = false
            // densifyGradThresh: standard 3DGS calibrates for full-image gradient density.
            // With foreground-mask-weighted loss the per-fg-pixel effective gradient drops
            // because `inv_n = 1/(H*W*3)` doesn't account for the masked pixel count, so
            // 0.0002 is too high (no splits trigger → plateau at init count). Calibration
            // history: 0.00005 caused ~12%/round growth → 80k splats at iter 2300 → OOM. 0.0001
            // halved that but still hit cap. 0.00015 (1.5x cut from default) targets ~6-8%/
            // round growth and a final ~50-100k splat count that fits comfortably in iPhone's
            // process-memory cap alongside the image/mask buffers.
            //
            // cullAlphaThresh: leave at the default (0.1). LiDAR-noise splats outside the
            // object volume SHOULD prune at the first opacity-reset; they don't carry any
            // gradient signal and just consume memory. Splats inside the object recover
            // opacity well above 0.1 within a few iters of each reset.
            training.start(keyframes: masked, points: initPoints,
                           workDir: workDir, outputPLY: outputPLY,
                           bgColor: (1, 1, 1),
                           densifyGradThresh: 0.00015)
        }
    }
}
#endif // os(iOS)
