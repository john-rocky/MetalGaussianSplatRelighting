import Foundation

/// What the capture-and-train flow should isolate.
enum CaptureMode: String, Codable, Hashable, Sendable {
    /// Train a splat of everything in view (a whole room, scene, etc.).
    case room
    /// Use on-device foreground segmentation to keep only the salient object and
    /// bake a white background, so the trainer fits the object alone.
    case object
}

enum ModelIdentifier: Equatable, Hashable, Codable, CustomStringConvertible {
    case gaussianSplat(URL)
    case proceduralSplat
    case sampleBox
    /// Capture with ARKit, train a Gaussian splat on-device, then view the result.
    case captureAndTrain(CaptureMode)

    var description: String {
        switch self {
        case .gaussianSplat(let url):
            "Gaussian Splat: \(url.path)"
        case .proceduralSplat:
            "Procedural Splat"
        case .sampleBox:
            "Sample Box"
        case .captureAndTrain(.room):
            "Capture Room"
        case .captureAndTrain(.object):
            "Capture Object"
        }
    }
}
