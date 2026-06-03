#if os(iOS)
import QuickLook
import RealityKit
import SwiftUI

/// Apple `ObjectCaptureSession`-backed flow for the "Capture Object (AR)" button. This replaces
/// the previous msplat training pipeline that lived in `CaptureFlowView(mode: .object)` —
/// see the route-b memory note for the strategic pivot. Room captures still go through
/// `CaptureFlowView`; the msplat object-mode plumbing (mask loss, `ObjectMaskProcessor`,
/// `Camera::mask`) stays dormant for now since room mode treats an absent mask as a no-op.
///
/// Pipeline shape: `ObjectCaptureSession` drives the system-guided capture UI →
/// `PhotogrammetrySession` reconstructs a `.usdz` in the temp directory →
/// `QLPreviewController` shows it (and AR Quick Look launches AR from there for free).
struct AppleObjectCaptureFlow: View {
    @Binding var navigationPath: NavigationPath
    @State private var pipeline = AppleObjectCapturePipeline()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if !ObjectCaptureSession.isSupported {
                unsupportedOverlay
            } else {
                content
            }
        }
        // Block back-swipe while a capture or reconstruction is mid-flight; the temp directory
        // would leak and the photogrammetry task would be orphaned otherwise.
        .navigationBarBackButtonHidden(pipeline.isBusy)
        .onAppear { pipeline.startIfNeeded() }
        .onDisappear { pipeline.tearDown() }
    }

    @ViewBuilder
    private var content: some View {
        switch pipeline.phase {
        case .idle, .initializing:
            ProgressView("Preparing capture…")
                .tint(.white)
                .foregroundStyle(.white)
        case .capturing(let session):
            captureUI(session: session)
        case .reconstructing:
            reconstructionUI
        case .preview(let url):
            USDZPreviewView(url: url)
                .ignoresSafeArea()
        case .failed(let message):
            failureOverlay(message)
        }
    }

    // MARK: Capture UI

    /// The system view (`ObjectCaptureView`) already renders the bounding box, distance hints,
    /// scan-pass progress ring and feedback chips — we only add the transition buttons that
    /// drive the state machine (`startDetecting` → `startCapturing` → `beginNewScanPass…` →
    /// `finish`).
    private func captureUI(session: ObjectCaptureSession) -> some View {
        ZStack {
            ObjectCaptureView(session: session)
                .ignoresSafeArea()

            VStack {
                Spacer()
                captureControls(for: session)
                    .padding(.horizontal, 24)
                    .padding(.bottom, 32)
            }
        }
    }

    @ViewBuilder
    private func captureControls(for session: ObjectCaptureSession) -> some View {
        switch session.state {
        case .initializing:
            // The session is bringing AR + segmentation up. The system view shows its own
            // spinner; we just keep our button row empty so it doesn't flash.
            EmptyView()
        case .ready:
            VStack(spacing: 8) {
                Text("Aim at the object on a flat surface, then continue.")
                    .promptStyle()
                Button("Continue") { _ = session.startDetecting() }
                    .buttonStyle(.borderedProminent)
            }
        case .detecting:
            VStack(spacing: 8) {
                Text("Adjust the bounding box, then start the first pass.")
                    .promptStyle()
                HStack {
                    Button("Reset Box") { _ = session.resetDetection() }
                        .buttonStyle(.bordered)
                        .tint(.white)
                    Button("Start Capture") { session.startCapturing() }
                        .buttonStyle(.borderedProminent)
                }
            }
        case .capturing:
            capturingControls(for: session)
        case .finishing:
            ProgressView("Finalizing capture…")
                .tint(.white)
                .foregroundStyle(.white)
        case .completed:
            // Transient — `handleStateChange` flips phase to .reconstructing on this state.
            ProgressView()
                .tint(.white)
        case .failed(let err):
            Text("Capture failed: \(err.localizedDescription)")
                .promptStyle()
        @unknown default:
            EmptyView()
        }
    }

    @ViewBuilder
    private func capturingControls(for session: ObjectCaptureSession) -> some View {
        VStack(spacing: 8) {
            Text("Pass \(pipeline.passNumber) of 3 · \(session.numberOfShotsTaken) shots")
                .promptStyle()

            if session.userCompletedScanPass {
                // Apple's guidance: flip the object between passes 1→2 and 2→3 so the bottom
                // gets captured. The user knows their object best, so we expose both choices
                // plus a finish — they can stop after any pass that looks good.
                HStack(spacing: 12) {
                    Button("Flip & Continue") {
                        pipeline.passNumber += 1
                        session.beginNewScanPassAfterFlip()
                    }
                    .buttonStyle(.borderedProminent)

                    Button("Same Side") {
                        pipeline.passNumber += 1
                        session.beginNewScanPass()
                    }
                    .buttonStyle(.bordered)
                    .tint(.white)

                    Button("Finish") { session.finish() }
                        .buttonStyle(.bordered)
                        .tint(.white)
                }
            }
        }
    }

    // MARK: Reconstruction & preview

    private var reconstructionUI: some View {
        VStack(spacing: 16) {
            ProgressView(value: pipeline.reconstructionProgress) {
                Text("Reconstructing model…")
                    .foregroundStyle(.white)
            }
            .tint(.white)
            Text("\(Int(pipeline.reconstructionProgress * 100))%")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.white)
            Button("Cancel") { pipeline.cancelReconstruction() }
                .buttonStyle(.bordered)
                .tint(.white)
        }
        .padding(24)
        .background(.black.opacity(0.7), in: .rect(cornerRadius: 16))
        .padding(40)
    }

    // MARK: Overlays

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

    private var unsupportedOverlay: some View {
        VStack(spacing: 12) {
            Text("Object Capture isn't supported on this device.")
                .foregroundStyle(.white)
            Text("Requires a LiDAR-equipped iPhone running iOS 17 or later.")
                .font(.caption)
                .foregroundStyle(.white.opacity(0.75))
            Button("Go Back") { dismiss() }
                .buttonStyle(.borderedProminent)
        }
        .padding(24)
        .background(.black.opacity(0.7), in: .rect(cornerRadius: 16))
        .padding(40)
    }
}

// MARK: - Pipeline

/// State machine + lifecycle owner for the Object Capture flow. Holds the working temp directory,
/// the live `ObjectCaptureSession`, the `PhotogrammetrySession` reconstructing the USDZ, and the
/// async observers tying them to SwiftUI. `@Observable` so the view re-renders on `phase` /
/// `reconstructionProgress` / `passNumber` changes without an `@Published` boilerplate layer.
@MainActor
@Observable
final class AppleObjectCapturePipeline {
    enum Phase {
        case idle
        case initializing
        /// Live capture in progress. SwiftUI observes the session directly via the Observation
        /// framework, so its `state`, `numberOfShotsTaken`, `userCompletedScanPass`, etc. drive
        /// view updates without us republishing.
        case capturing(ObjectCaptureSession)
        case reconstructing
        case preview(URL)
        case failed(String)
    }

    private(set) var phase: Phase = .idle
    private(set) var reconstructionProgress: Double = 0
    /// 1-based; user-facing "Pass N of 3". Bumped when the user chooses to continue after a
    /// completed pass — the session itself doesn't expose a pass counter.
    var passNumber: Int = 1

    var isBusy: Bool {
        switch phase {
        case .capturing, .reconstructing, .initializing: true
        default: false
        }
    }

    private var workRoot: URL?
    private var imagesDirectory: URL?
    private var modelOutputURL: URL?
    private var session: ObjectCaptureSession?
    private var photogrammetrySession: PhotogrammetrySession?
    private var stateObserver: Task<Void, Never>?
    private var processingObserver: Task<Void, Never>?

    func startIfNeeded() {
        guard case .idle = phase else { return }
        guard ObjectCaptureSession.isSupported else {
            // The view's `unsupportedOverlay` already covers this case visually; we just keep
            // phase at .idle so we don't kick off any session work.
            return
        }
        phase = .initializing

        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("object-capture-\(UUID().uuidString)", isDirectory: true)
        let imagesDir = root.appendingPathComponent("images", isDirectory: true)
        let checkpointDir = root.appendingPathComponent("snapshots", isDirectory: true)
        let modelOut = root.appendingPathComponent("model.usdz")
        do {
            try FileManager.default.createDirectory(at: imagesDir, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: checkpointDir, withIntermediateDirectories: true)
        } catch {
            phase = .failed("Could not create capture directory: \(error.localizedDescription)")
            return
        }
        self.workRoot = root
        self.imagesDirectory = imagesDir
        self.modelOutputURL = modelOut

        let session = ObjectCaptureSession()
        var config = ObjectCaptureSession.Configuration()
        config.checkpointDirectory = checkpointDir
        // `isOverCaptureEnabled` captures extra detail images during each orbit; the on-device
        // reconstructor uses them as supplementary high-res inputs. Worth the small extra time
        // for the kind of object we're replacing the msplat path on (small, detail-rich).
        config.isOverCaptureEnabled = true
        session.start(imagesDirectory: imagesDir, configuration: config)
        self.session = session
        self.passNumber = 1
        phase = .capturing(session)
        observeSession(session)
    }

    private func observeSession(_ session: ObjectCaptureSession) {
        stateObserver?.cancel()
        stateObserver = Task { [weak self] in
            for await state in session.stateUpdates {
                guard let self else { return }
                await MainActor.run { self.handle(state: state) }
            }
        }
    }

    private func handle(state: ObjectCaptureSession.CaptureState) {
        switch state {
        case .completed:
            // Capture done; session has written all images to `imagesDirectory`. Tear the
            // session down before starting reconstruction so its ARKit + segmentation resources
            // are released — the on-device photogrammetry needs every MB it can get.
            session?.cancel()
            session = nil
            startReconstruction()
        case .failed(let err):
            phase = .failed("Capture failed: \(err.localizedDescription)")
        default:
            break
        }
    }

    private func startReconstruction() {
        guard let imagesDirectory, let modelOutputURL else { return }
        phase = .reconstructing
        reconstructionProgress = 0

        let pg: PhotogrammetrySession
        do {
            pg = try PhotogrammetrySession(input: imagesDirectory)
        } catch {
            phase = .failed("Could not start reconstruction: \(error.localizedDescription)")
            return
        }
        self.photogrammetrySession = pg

        processingObserver = Task { [weak self] in
            guard let self else { return }
            // `pg.outputs` is a throwing AsyncSequence — iterator errors surface here (as
            // opposed to per-request `.requestError` outputs, which are non-throwing).
            do {
                for try await output in pg.outputs {
                    await MainActor.run { self.handle(output: output, modelURL: modelOutputURL) }
                }
            } catch {
                await MainActor.run {
                    self.phase = .failed("Reconstruction failed: \(error.localizedDescription)")
                }
            }
        }

        // `.reduced` is the right detail level on-device: `.medium` roughly triples wall time
        // and only marginally improves AR-Quick-Look-scale viewing. `.full`/`.raw` are
        // Mac-only.
        let request = PhotogrammetrySession.Request.modelFile(url: modelOutputURL, detail: .reduced)
        do {
            try pg.process(requests: [request])
        } catch {
            phase = .failed("Could not start reconstruction: \(error.localizedDescription)")
        }
    }

    private func handle(output: PhotogrammetrySession.Output, modelURL: URL) {
        switch output {
        case .requestProgress(_, let fraction):
            reconstructionProgress = fraction
        case .processingComplete:
            phase = .preview(modelURL)
        case .requestError(_, let err):
            phase = .failed("Reconstruction failed: \(err.localizedDescription)")
        case .processingCancelled:
            // User-initiated; fall back to idle so they can start over without leaving the view.
            phase = .idle
        default:
            // .inputComplete / .requestComplete / .requestProgressInfo / .invalidSample /
            // .skippedSample / .stitchingIncomplete / .automaticDownsampling — informational;
            // the per-request progress + processingComplete already cover what the UI needs.
            break
        }
    }

    func cancelReconstruction() {
        photogrammetrySession?.cancel()
        processingObserver?.cancel()
        processingObserver = nil
        photogrammetrySession = nil
        phase = .idle
    }

    func tearDown() {
        stateObserver?.cancel()
        stateObserver = nil
        processingObserver?.cancel()
        processingObserver = nil
        photogrammetrySession?.cancel()
        photogrammetrySession = nil
        session?.cancel()
        session = nil
        // We deliberately do NOT delete `workRoot` here: the USDZ at `modelOutputURL` may still
        // be in use by the QLPreviewController if we're tearing down on a back-nav from preview.
        // The OS will reclaim the temp directory when it cycles.
    }
}

// MARK: - USDZ preview

/// Wrap `QLPreviewController` so the produced USDZ can be inspected in SwiftUI. Quick Look
/// detects `.usdz` and shows the built-in 3D viewer with an "AR" button → AR Quick Look — so
/// the user gets room-anchored AR placement for free without us spinning up our own ARView.
struct USDZPreviewView: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> QLPreviewController {
        let controller = QLPreviewController()
        controller.dataSource = context.coordinator
        return controller
    }

    func updateUIViewController(_ uiViewController: QLPreviewController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(url: url) }

    final class Coordinator: NSObject, QLPreviewControllerDataSource {
        let url: URL
        init(url: URL) { self.url = url }
        func numberOfPreviewItems(in controller: QLPreviewController) -> Int { 1 }
        func previewController(_ controller: QLPreviewController,
                               previewItemAt index: Int) -> QLPreviewItem {
            url as QLPreviewItem
        }
    }
}

// MARK: - Small helpers

private extension View {
    /// Shared chrome for the floating prompt strings layered over `ObjectCaptureView`.
    func promptStyle() -> some View {
        self
            .font(.subheadline)
            .multilineTextAlignment(.center)
            .foregroundStyle(.white)
            .padding(8)
            .background(.black.opacity(0.5), in: .rect(cornerRadius: 8))
    }
}

#endif // os(iOS)
