import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var isPickingFile = false
#if os(iOS)
    @State private var relightControls = RelightControls()
#endif

#if os(macOS)
    @Environment(\.openWindow) private var openWindow
#elseif os(iOS)
    @State private var navigationPath = NavigationPath()

    private func openWindow(value: ModelIdentifier) {
        navigationPath.append(value)
    }
#elseif os(visionOS)
    @Environment(\.openImmersiveSpace) var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) var dismissImmersiveSpace

    @State var immersiveSpaceIsShown = false

    private func openWindow(value: ModelIdentifier) {
        Task {
            switch await openImmersiveSpace(value: value) {
            case .opened:
                immersiveSpaceIsShown = true
            case .error, .userCancelled:
                break
            @unknown default:
                break
            }
        }
    }
#endif

    var body: some View {
#if os(macOS) || os(visionOS)
        mainView
#elseif os(iOS)
        NavigationStack(path: $navigationPath) {
            mainView
                .navigationDestination(for: ModelIdentifier.self) { modelIdentifier in
                    if case .captureAndTrain(.object) = modelIdentifier {
                        // Object capture is delegated to Apple's ObjectCaptureSession +
                        // PhotogrammetrySession → USDZ pipeline; see AppleObjectCaptureFlow.
                        AppleObjectCaptureFlow(navigationPath: $navigationPath)
                            .navigationTitle(modelIdentifier.description)
                    } else if case .captureAndTrain(let mode) = modelIdentifier {
                        // Room captures continue to use the msplat training pipeline.
                        CaptureFlowView(mode: mode, navigationPath: $navigationPath)
                            .navigationTitle(modelIdentifier.description)
                    } else {
                        MetalKitSceneView(modelIdentifier: modelIdentifier, relightControls: relightControls)
                            .navigationTitle(modelIdentifier.description)
                            .overlay(alignment: .bottom) {
                                if case .gaussianSplat = modelIdentifier {
                                    RelightControlsView(controls: relightControls)
                                        .padding()
                                }
                            }
                    }
                }
        }
#endif // os(iOS)
    }

    @ViewBuilder
    var mainView: some View {
        VStack {
            Spacer()

            Text("MetalSplatter SampleApp")

            Spacer()

            Button("Read Scene File") {
                isPickingFile = true
            }
            .padding()
            .buttonStyle(.borderedProminent)
            .disabled(isPickingFile)
#if os(visionOS)
            .disabled(immersiveSpaceIsShown)
#endif
            .fileImporter(isPresented: $isPickingFile,
                          allowedContentTypes: [
                            UTType(filenameExtension: "ply")!,
                            UTType(filenameExtension: "splat")!,
                            UTType(filenameExtension: "spz")!,
                          ]) {
                isPickingFile = false
                switch $0 {
                case .success(let url):
                    _ = url.startAccessingSecurityScopedResource()
                    Task {
                        // This is a sample app. In a real app, this should be more tightly scoped, not using a silly timer.
                        try await Task.sleep(for: .seconds(10))
                        url.stopAccessingSecurityScopedResource()
                    }
                    openWindow(value: ModelIdentifier.gaussianSplat(url))
                case .failure:
                    break
                }
            }

#if os(iOS)
            Button("Capture Room (AR)") {
                openWindow(value: ModelIdentifier.captureAndTrain(.room))
            }
            .padding()
            .buttonStyle(.borderedProminent)
            .disabled(!ARCaptureController.isSupported)

            Button("Capture Object (AR)") {
                openWindow(value: ModelIdentifier.captureAndTrain(.object))
            }
            .padding()
            .buttonStyle(.borderedProminent)
            .disabled(!ARCaptureController.isSupported)
#endif

            Button("Procedural Splat") {
                openWindow(value: ModelIdentifier.proceduralSplat)
            }
            .padding()
            .buttonStyle(.borderedProminent)
#if os(visionOS)
            .disabled(immersiveSpaceIsShown)
#endif

            Button("Show Sample Box") {
                openWindow(value: ModelIdentifier.sampleBox)
            }
            .padding()
            .buttonStyle(.borderedProminent)
#if os(visionOS)
            .disabled(immersiveSpaceIsShown)
#endif

            Spacer()

#if os(visionOS)
            Button("Dismiss Immersive Space") {
                Task {
                    await dismissImmersiveSpace()
                    immersiveSpaceIsShown = false
                }
            }
            .disabled(!immersiveSpaceIsShown)

            Spacer()
#endif // os(visionOS)
        }
    }
}
