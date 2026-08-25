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

#if os(iOS)
        @objc func handlePan(_ g: UIPanGestureRecognizer) {
            let t = g.translation(in: g.view)
            if g.state == .began { panLastTranslation = .zero }
            renderer?.orbitBy(dx: Float(t.x - panLastTranslation.x),
                              dy: Float(t.y - panLastTranslation.y))
            panLastTranslation = t
        }
        @objc func handlePinch(_ g: UIPinchGestureRecognizer) {
            if g.state == .began { renderer?.pinchBegan() }
            renderer?.zoomBy(scale: Float(g.scale))
        }
        @objc func handleTap(_ g: UITapGestureRecognizer) {
            guard let view = g.view, view.bounds.width > 0, view.bounds.height > 0 else { return }
            let p = g.location(in: view)
            renderer?.placeAt(normalizedPoint: CGPoint(x: p.x / view.bounds.width, y: p.y / view.bounds.height))
        }
        // Two-finger twist: stand a sideways-loaded splat upright (adjusts the model up-tilt).
        @objc func handleRotation(_ g: UIRotationGestureRecognizer) {
            renderer?.tiltBy(radians: Float(g.rotation))
            g.rotation = 0
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
            if g.state == .began { renderer?.pinchBegan() }
            renderer?.zoomBy(scale: Float(1 + g.magnification))
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

        // MTKView caps at 60 by default, and inside that cap the GPU clocks down
        // to just meet it — so a frame that could finish in 3 ms reports 15 ms
        // and every scene under the cap looks identical. BENCH_FPS raises the
        // ceiling so a measurement sees the work rather than the power policy.
        if let fps = ProcessInfo.processInfo.environment["BENCH_FPS"], let n = Int(fps) {
            metalKitView.preferredFramesPerSecond = n
        }

        let renderer = MetalKitSceneRenderer(metalKitView)
        renderer?.relightControls = relightControls
        coordinator.renderer = renderer
        metalKitView.delegate = renderer

        // Interactive orbit camera: drag to rotate, pinch to zoom.
#if os(iOS)
        metalKitView.addGestureRecognizer(UIPanGestureRecognizer(target: coordinator, action: #selector(Coordinator.handlePan(_:))))
        let pinch = UIPinchGestureRecognizer(target: coordinator, action: #selector(Coordinator.handlePinch(_:)))
        pinch.delegate = coordinator
        metalKitView.addGestureRecognizer(pinch)
        // Two-finger twist to stand the model upright; recognized simultaneously with pinch-zoom.
        let rotate = UIRotationGestureRecognizer(target: coordinator, action: #selector(Coordinator.handleRotation(_:)))
        rotate.delegate = coordinator
        metalKitView.addGestureRecognizer(rotate)
        metalKitView.addGestureRecognizer(UITapGestureRecognizer(target: coordinator, action: #selector(Coordinator.handleTap(_:))))
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

#if os(iOS)
extension MetalKitSceneView.Coordinator: UIGestureRecognizerDelegate {
    // Let pinch-zoom and the two-finger twist run at the same time.
    func gestureRecognizer(_ g: UIGestureRecognizer,
                           shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool { true }
}
#endif

#endif // os(iOS) || os(macOS)
