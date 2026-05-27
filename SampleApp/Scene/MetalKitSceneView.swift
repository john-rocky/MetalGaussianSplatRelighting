#if os(iOS) || os(macOS)

import SwiftUI
import MetalKit

#if os(macOS)
private typealias ViewRepresentable = NSViewRepresentable
#elseif os(iOS)
private typealias ViewRepresentable = UIViewRepresentable
#endif

struct MetalKitSceneView: ViewRepresentable {
    var modelIdentifier: ModelIdentifier?
    var relightControls: RelightControls? = nil

    // Gesture callbacks are delivered on the main thread, and the renderer is @MainActor, so the
    // coordinator is main-actor isolated too (required under Swift 6 strict concurrency).
    @MainActor
    class Coordinator: NSObject {
        var renderer: MetalKitSceneRenderer?
        private var panLastTranslation: CGPoint = .zero
        private var pinchReferenceDistance: Float = Constants.cameraInitialDistance

#if os(iOS)
        @objc func handlePan(_ g: UIPanGestureRecognizer) {
            let t = g.translation(in: g.view)
            if g.state == .began { panLastTranslation = .zero }
            renderer?.orbitBy(dx: Float(t.x - panLastTranslation.x),
                              dy: Float(t.y - panLastTranslation.y))
            panLastTranslation = t
        }
        @objc func handlePinch(_ g: UIPinchGestureRecognizer) {
            if g.state == .began { pinchReferenceDistance = renderer?.cameraDistance ?? Constants.cameraInitialDistance }
            renderer?.zoomBy(scale: Float(g.scale), referenceDistance: pinchReferenceDistance)
        }
#elseif os(macOS)
        @objc func handlePan(_ g: NSPanGestureRecognizer) {
            let t = g.translation(in: g.view)
            if g.state == .began { panLastTranslation = .zero }
            // AppKit's y axis is up-positive; negate to match the iOS drag feel.
            renderer?.orbitBy(dx: Float(t.x - panLastTranslation.x),
                              dy: Float(-(t.y - panLastTranslation.y)))
            panLastTranslation = t
        }
        @objc func handleMagnify(_ g: NSMagnificationGestureRecognizer) {
            if g.state == .began { pinchReferenceDistance = renderer?.cameraDistance ?? Constants.cameraInitialDistance }
            renderer?.zoomBy(scale: Float(1 + g.magnification), referenceDistance: pinchReferenceDistance)
        }
#endif
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

#if os(macOS)
    func makeNSView(context: NSViewRepresentableContext<MetalKitSceneView>) -> MTKView {
        makeView(context.coordinator)
    }
#elseif os(iOS)
    func makeUIView(context: UIViewRepresentableContext<MetalKitSceneView>) -> MTKView {
        makeView(context.coordinator)
    }
#endif

    private func makeView(_ coordinator: Coordinator) -> MTKView {
        let metalKitView = MTKView()

        if let metalDevice = MTLCreateSystemDefaultDevice() {
            metalKitView.device = metalDevice
        }

        let renderer = MetalKitSceneRenderer(metalKitView)
        renderer?.relightControls = relightControls
        coordinator.renderer = renderer
        metalKitView.delegate = renderer

        // Interactive orbit camera: drag to rotate, pinch to zoom.
#if os(iOS)
        metalKitView.addGestureRecognizer(UIPanGestureRecognizer(target: coordinator, action: #selector(Coordinator.handlePan(_:))))
        metalKitView.addGestureRecognizer(UIPinchGestureRecognizer(target: coordinator, action: #selector(Coordinator.handlePinch(_:))))
#elseif os(macOS)
        metalKitView.addGestureRecognizer(NSPanGestureRecognizer(target: coordinator, action: #selector(Coordinator.handlePan(_:))))
        metalKitView.addGestureRecognizer(NSMagnificationGestureRecognizer(target: coordinator, action: #selector(Coordinator.handleMagnify(_:))))
#endif

        Task {
            do {
                try await renderer?.load(modelIdentifier)
            } catch {
                print("Error loading model: \(error.localizedDescription)")
            }
        }

        return metalKitView
    }

#if os(macOS)
    func updateNSView(_ view: MTKView, context: NSViewRepresentableContext<MetalKitSceneView>) {
        updateView(context.coordinator)
    }
#elseif os(iOS)
    func updateUIView(_ view: MTKView, context: UIViewRepresentableContext<MetalKitSceneView>) {
        updateView(context.coordinator)
    }
#endif

    private func updateView(_ coordinator: Coordinator) {
        guard let renderer = coordinator.renderer else { return }
        renderer.relightControls = relightControls
        Task {
            do {
                try await renderer.load(modelIdentifier)
            } catch {
                print("Error loading model: \(error.localizedDescription)")
            }
        }
    }
}

#endif // os(iOS) || os(macOS)
