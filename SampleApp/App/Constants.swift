import Foundation
import SwiftUI

enum Constants {
    static let maxSimultaneousRenders = 3
    static let rotationPerSecond = Angle(degrees: 7)
    static let rotationAxis = SIMD3<Float>(0, 1, 0)
#if !os(visionOS)
    static let fovy = Angle(degrees: 65)
#endif
    static let modelCenterZ: Float = -8

    // Orbit camera: drag to rotate (yaw/pitch), pinch to zoom (distance from the model).
    static let cameraInitialDistance: Float = 8
    static let cameraMinDistance: Float = 0.15
    static let cameraMaxDistance: Float = 30
    static let cameraInitialPitch = Angle(degrees: -10)   // slight downward tilt looks natural
    static let cameraPitchLimit = Angle(degrees: 89)      // avoid flipping over the poles
    static let cameraOrbitRadiansPerPoint: Float = 0.01   // drag sensitivity (radians per screen point)

    // Procedural splat geometry
    static let proceduralCubeSize: Float = 1.0
    static let proceduralCubeDistance: Float = 1.0
    static let proceduralCubeGridSizes: [Int] = [10, 20, 50]
    static let proceduralCubeSplatRelativeRadius: Float = 0.1
    static let proceduralCubeSwapDelay: TimeInterval = 2.0
}

