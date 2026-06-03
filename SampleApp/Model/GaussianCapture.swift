#if os(iOS)
import ARKit
import Combine
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreVideo
import Foundation
import Msplat
import UIKit
import Vision
import simd

// MARK: - Captured keyframe

/// One captured training view: the camera image plus its pose and intrinsics, in the form
/// msplat's Nerfstudio loader expects. The camera-to-world matrix is stored already in
/// row-major order (the loader reads `transform_matrix[r][c]`), so no conversion is needed at
/// write time. ARKit's camera space (+X right, +Y up, looking down -Z) matches the OpenGL
/// convention the loader assumes, so the pose needs no axis remapping.
struct CaptureKeyframe: Sendable {
    let jpeg: Data
    let width: Int
    let height: Int
    /// 4x4 camera-to-world, row-major, OpenGL/ARKit convention.
    let transformRowMajor: [Float]
    /// Pinhole intrinsics in pixels, matching `width`/`height`. ARKit's `capturedImage` is
    /// always in sensor-native landscape regardless of how the device is held, and so are the
    /// intrinsics and `camera.transform` — we keep all three in that frame so they're internally
    /// consistent for the trainer.
    let fx: Float, fy: Float, cx: Float, cy: Float
    /// EXIF orientation tag carried by the JPEG. The pixel data stays in sensor-native
    /// landscape (so msplat's `CGImageSourceCreateImageAtIndex` reads pixels that agree with
    /// `fx/fy/cx/cy`), and this tag is what makes `UIImage(data:)` previews show the right way
    /// up and what we pass to Vision so its segmentation ML model sees naturally-oriented input.
    let orientation: UIImage.Orientation
    /// LiDAR depth captured with the frame, used by object mode to seed init points inside the
    /// object. Nil on non-LiDAR devices and on the masked output keyframe (we drop it once the
    /// unprojection has been done — no point keeping ~200 KB per frame alive through training).
    let depth: DepthSnapshot?
    /// Optional foreground mask (single-channel grayscale PNG, same dimensions as the JPEG).
    /// When present, the writer emits it next to the image and adds a `mask_path` entry to
    /// `transforms.json`; msplat then applies per-pixel loss masking so background pixels
    /// don't drive densification. Nil for room-mode captures (no segmentation).
    let maskPNG: Data?
}

/// `UIImage.Orientation` and `CGImagePropertyOrientation` carry the same EXIF semantics under
/// different raw values; this bridges them for handing the keyframe's orientation to Vision.
extension CGImagePropertyOrientation {
    init(_ ui: UIImage.Orientation) {
        switch ui {
        case .up:             self = .up
        case .down:           self = .down
        case .left:           self = .left
        case .right:          self = .right
        case .upMirrored:     self = .upMirrored
        case .downMirrored:   self = .downMirrored
        case .leftMirrored:   self = .leftMirrored
        case .rightMirrored:  self = .rightMirrored
        @unknown default:     self = .up
        }
    }
}

// MARK: - AR capture

/// Owns a world-tracking ARSession and turns manual taps into training keyframes. The matching
/// `ARPreviewView` shares this session so the live camera feed and the captured poses come from
/// the exact same tracking state.
@MainActor
final class ARCaptureController: ObservableObject {
    /// Whether this device supports the world tracking the capture relies on.
    static var isSupported: Bool { ARWorldTrackingConfiguration.isSupported }

    let session = ARSession()

    @Published private(set) var keyframeCount = 0
    /// Thumbnail of the most recent capture, for a quick visual confirmation in the UI.
    @Published private(set) var lastThumbnail: UIImage?

    private(set) var keyframes: [CaptureKeyframe] = []
    /// World-space sparse points accumulated from ARKit, deduplicated by feature identifier, used
    /// to seed the trainer's initial point cloud (better than random initialization).
    private var featurePoints: [UInt64: simd_float3] = [:]
    private let ciContext = CIContext()
    private var isRunning = false

    func start() {
        guard Self.isSupported, !isRunning else { return }
        let configuration = ARWorldTrackingConfiguration()
        configuration.worldAlignment = .gravity
        // Enable LiDAR depth (no-op on non-LiDAR devices). Object mode unprojects masked depth
        // pixels per keyframe to seed the trainer with a dense in-object init cloud.
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            configuration.frameSemantics.insert(.smoothedSceneDepth)
        }
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        isRunning = true
    }

    func stop() {
        guard isRunning else { return }
        session.pause()
        isRunning = false
    }

    /// Records the current frame as a keyframe. Returns false if no usable frame is available yet.
    @discardableResult
    func captureKeyframe() -> Bool {
        guard let frame = session.currentFrame else { return false }
        // ARKit reuses the captured-image buffer, so convert to JPEG immediately rather than retaining it.
        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: CGRect(x: 0, y: 0, width: width, height: height)) else {
            return false
        }
        // Save sensor-native JPEG bytes verbatim. Using `UIImage(cgImage:scale:orientation:)`
        // with a non-`.up` value here would silently bake the rotation into the pixels on iOS
        // 14+ — the JPEG bytes would no longer match the sensor-native intrinsics, and Vision's
        // orientation hint would compound on top to double-rotate the input. Orientation lives
        // on the keyframe as metadata only; previews wrap with it on display.
        let orientation = Self.currentImageOrientation()
        guard let jpeg = UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.9) else { return false }
        let displayImage = UIImage(cgImage: cgImage, scale: 1, orientation: orientation)

        // Intrinsics are defined for `imageResolution`, which equals the captured-image dimensions.
        let k = frame.camera.intrinsics
        keyframes.append(CaptureKeyframe(
            jpeg: jpeg,
            width: width,
            height: height,
            transformRowMajor: Self.rowMajor(frame.camera.transform),
            fx: k.columns.0.x, fy: k.columns.1.y, cx: k.columns.2.x, cy: k.columns.2.y,
            orientation: orientation,
            depth: DepthSnapshot(from: frame.smoothedSceneDepth ?? frame.sceneDepth),
            maskPNG: nil))   // room-mode captures carry no mask

        if let cloud = frame.rawFeaturePoints {
            for i in cloud.points.indices {
                featurePoints[cloud.identifiers[i]] = cloud.points[i]
            }
        }

        lastThumbnail = displayImage
        keyframeCount = keyframes.count
        return true
    }

    /// Accumulated world-space points for trainer initialization.
    var collectedPoints: [simd_float3] { Array(featurePoints.values) }

    func reset() {
        keyframes.removeAll()
        featurePoints.removeAll()
        lastThumbnail = nil
        keyframeCount = 0
    }

    /// simd stores matrices column-major; emit true row-major M[r][c] = columns[c][r].
    /// `simd_float4x4.columns` is a tuple, not an array — index it via a local array.
    private static func rowMajor(_ m: simd_float4x4) -> [Float] {
        let cols = [m.columns.0, m.columns.1, m.columns.2, m.columns.3]
        var out = [Float](repeating: 0, count: 16)
        for r in 0..<4 {
            for c in 0..<4 {
                out[r * 4 + c] = cols[c][r]
            }
        }
        return out
    }

    /// Picks the EXIF orientation that, when written into a back-camera JPEG (which is always
    /// sensor-native landscape), makes the image display naturally for the device's current
    /// hold orientation. Default is portrait (`.right`).
    private static func currentImageOrientation() -> UIImage.Orientation {
        let interface: UIInterfaceOrientation
        if let scene = UIApplication.shared.connectedScenes
            .compactMap({ $0 as? UIWindowScene })
            .first(where: { $0.activationState == .foregroundActive }) {
            interface = scene.interfaceOrientation
        } else {
            interface = .portrait
        }
        switch interface {
        case .portrait:            return .right
        case .portraitUpsideDown:  return .left
        case .landscapeLeft:       return .down
        case .landscapeRight:      return .up
        @unknown default:          return .right
        }
    }
}

// MARK: - Nerfstudio dataset writer

/// Serializes captured keyframes into a Nerfstudio-format directory (`transforms.json`, `images/`,
/// and an optional `points3D.ply`) that `GaussianDataset` can load directly.
enum NerfstudioDatasetWriter {
    static func write(keyframes: [CaptureKeyframe], points: [simd_float3], to root: URL) throws {
        guard !keyframes.isEmpty else {
            // Defensive: surfaces as a `.failed` phase via GaussianTrainingController's catch.
            // Typical cause is foreground segmentation finding nothing in any captured frame
            // (reflective/transparent objects, very small subjects, harsh lighting).
            throw NSError(domain: "NerfstudioDatasetWriter", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "No frames to write — the foreground detector didn't find an object in any captured frame. Try better lighting, fill more of the frame, or recapture."
            ])
        }
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: root.path) {
            try fileManager.removeItem(at: root)
        }
        let imagesDir = root.appendingPathComponent("images")
        try fileManager.createDirectory(at: imagesDir, withIntermediateDirectories: true)

        var frames: [[String: Any]] = []
        frames.reserveCapacity(keyframes.count)
        for (index, keyframe) in keyframes.enumerated() {
            let name = String(format: "frame_%04d.jpg", index)
            try keyframe.jpeg.write(to: imagesDir.appendingPathComponent(name))
            var entry: [String: Any] = [
                "file_path": "images/\(name)",
                "w": keyframe.width,
                "h": keyframe.height,
                "fl_x": Double(keyframe.fx),
                "fl_y": Double(keyframe.fy),
                "cx": Double(keyframe.cx),
                "cy": Double(keyframe.cy),
                "transform_matrix": reshape(keyframe.transformRowMajor),
            ]
            // Foreground mask for loss masking. msplat's load_nerfstudio reads `mask_path` and
            // routes it through Camera::loadMask → getGPUMask → ssim_*_kernel, weighting the
            // SSIM/L1 loss per pixel. Only present for object-mode captures (the
            // `ObjectMaskProcessor` attaches a PNG of the soft Vision mask per frame).
            if let maskPNG = keyframe.maskPNG {
                let maskName = String(format: "frame_%04d_mask.png", index)
                try maskPNG.write(to: imagesDir.appendingPathComponent(maskName))
                entry["mask_path"] = "images/\(maskName)"
            }
            frames.append(entry)
        }

        // Global intrinsics double as defaults; per-frame values above override them if they drift.
        let first = keyframes[0]
        let manifest: [String: Any] = [
            "w": first.width,
            "h": first.height,
            "fl_x": Double(first.fx),
            "fl_y": Double(first.fy),
            "cx": Double(first.cx),
            "cy": Double(first.cy),
            "frames": frames,
        ]

        if !points.isEmpty {
            // The loader auto-detects "points3D.ply" in the root, so no ply_file_path entry is needed.
            try writePointCloud(points, to: root.appendingPathComponent("points3D.ply"))
        }

        let data = try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: root.appendingPathComponent("transforms.json"))
    }

    /// 16-element row-major array -> 4x4 nested array for JSON.
    private static func reshape(_ flat: [Float]) -> [[Double]] {
        stride(from: 0, to: 16, by: 4).map { row in
            (0..<4).map { Double(flat[row + $0]) }
        }
    }

    /// Writes a minimal binary little-endian PLY (positions only; the loader defaults missing
    /// colors to gray). iOS is little-endian, so native float bytes are already LE.
    private static func writePointCloud(_ points: [simd_float3], to url: URL) throws {
        var header = "ply\n"
        header += "format binary_little_endian 1.0\n"
        header += "element vertex \(points.count)\n"
        header += "property float x\n"
        header += "property float y\n"
        header += "property float z\n"
        header += "end_header\n"

        var data = Data(header.utf8)
        data.reserveCapacity(data.count + points.count * 12)
        for point in points {
            for value in [point.x, point.y, point.z] {
                withUnsafeBytes(of: value) { data.append(contentsOf: $0) }
            }
        }
        try data.write(to: url)
    }
}

// MARK: - Object isolation

/// LiDAR depth snapshot copied out of an ARFrame at capture time. Stored as plain arrays
/// (rather than ARKit's CVPixelBuffer) so it's value-type Sendable and survives off-main
/// processing without surprise pointer aliasing into ARKit's reused buffers.
struct DepthSnapshot: Sendable {
    let width: Int
    let height: Int
    /// Row-major perpendicular distance in metres along the camera's optical-axis Z.
    let values: [Float]
    /// Per-pixel confidence (0 = low, 1 = medium, 2 = high), same shape as `values`.
    let confidence: [UInt8]

    init?(from depthData: ARDepthData?) {
        guard let depthData else { return nil }
        let map = depthData.depthMap
        guard CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32 else { return nil }
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        let w = CVPixelBufferGetWidth(map)
        let h = CVPixelBufferGetHeight(map)
        let depthStride = CVPixelBufferGetBytesPerRow(map) / MemoryLayout<Float>.size
        guard let base = CVPixelBufferGetBaseAddress(map)?.assumingMemoryBound(to: Float.self) else { return nil }
        var vals = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                vals[y * w + x] = base[y * depthStride + x]
            }
        }

        var conf = [UInt8](repeating: 2, count: w * h)
        if let cmap = depthData.confidenceMap {
            CVPixelBufferLockBaseAddress(cmap, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(cmap, .readOnly) }
            let cstride = CVPixelBufferGetBytesPerRow(cmap)
            if let cbase = CVPixelBufferGetBaseAddress(cmap)?.assumingMemoryBound(to: UInt8.self) {
                for y in 0..<h {
                    for x in 0..<w {
                        conf[y * w + x] = cbase[y * cstride + x]
                    }
                }
            }
        }

        self.width = w
        self.height = h
        self.values = vals
        self.confidence = conf
    }
}

/// Output of `ObjectMaskProcessor.mask`: the keyframe with its background baked out, plus any
/// world-space points unprojected from masked LiDAR depth pixels (empty on non-LiDAR devices).
struct MaskResult: Sendable {
    let keyframe: CaptureKeyframe
    let points: [simd_float3]
}

/// Two jobs per keyframe in one pass: (1) replace everything outside the salient foreground
/// instance with a solid colour, so the trainer's `bgColor` matches the pixels it should render
/// as "nothing" (Blender / NeRF-synthetic recipe — splats only grow on the object); (2) if LiDAR
/// depth was captured with the frame, unproject the masked pixels into world-space points to
/// seed the trainer with a *dense cloud inside the object*. The second one is what makes object
/// mode actually converge to crisp geometry rather than a fuzzy haze — ARKit's sparse
/// `rawFeaturePoints` are concentrated on textured backgrounds, not on the object you care about.
enum ObjectMaskProcessor {
    private static let ciContext = CIContext()

    static func mask(_ keyframe: CaptureKeyframe, bgR: Float, bgG: Float, bgB: Float) -> MaskResult? {
        // Hand the JPEG straight to Vision with the EXIF orientation tag — the segmentation ML
        // model was trained on naturally-oriented images, so a sideways input (which is what
        // sensor-native landscape looks like when the user is in portrait) dramatically degrades
        // mask quality. Vision's outputs (the mask & masked-image buffers) come back in the
        // *source image's pixel space*, i.e. still sensor-native, so the downstream pipeline
        // stays internally consistent with the keyframe's intrinsics and pose.
        let handler = VNImageRequestHandler(data: keyframe.jpeg,
                                            orientation: CGImagePropertyOrientation(keyframe.orientation),
                                            options: [:])
        let request = VNGenerateForegroundInstanceMaskRequest()
        do {
            try handler.perform([request])
        } catch {
            return nil
        }
        guard let observation = request.results?.first,
              !observation.allInstances.isEmpty else { return nil }

        // Pick only the dominant detected instance, so a co-detected table edge or hand doesn't
        // bleed into the foreground and feed the trainer inconsistent silhouettes per frame.
        let instances = dominantInstance(observation)

        // 1. Source-resolution binary mask. With an orientation hint, some Vision builds return
        //    the mask in the corrected (display) frame instead of the raw pixel-buffer frame;
        //    that would misalign it with the JPEG bytes, intrinsics, depth — all sensor-native
        //    landscape — and produce a composite where the mask outline doesn't match the
        //    image content (visible as "mask is rotated relative to the original" in previews).
        //    Detect the mismatch by dimensions and rotate the mask back to sensor-native.
        guard let rawMaskBuffer = try? observation.generateScaledMaskForImage(forInstances: instances,
                                                                              from: handler) else {
            return nil
        }
        let maskBuffer: CVPixelBuffer
        if CVPixelBufferGetWidth(rawMaskBuffer) == keyframe.width &&
           CVPixelBufferGetHeight(rawMaskBuffer) == keyframe.height {
            maskBuffer = rawMaskBuffer
        } else {
            let rawCI = CIImage(cvPixelBuffer: rawMaskBuffer)
            let inverse = inverseOrientation(CGImagePropertyOrientation(keyframe.orientation))
            guard let nb = renderMaskBuffer(rawCI.oriented(inverse),
                                            width: keyframe.width,
                                            height: keyframe.height) else { return nil }
            maskBuffer = nb
        }

        // 2. Soft-edge composite. Vision's mask is effectively binary, and compositing with a
        //    hard 0 → 1 transition at the silhouette becomes a step function in the trained
        //    image — 3DGS reads that as near-infinite gradient and densifies into millions of
        //    micro-splats along the boundary, OOMing the iPhone around iter ~2k. Blender's
        //    NeRF-synthetic datasets avoid this for free with anti-aliased α at render time; we
        //    replicate it with a Gaussian-blurred mask + `CIBlendWithMask`. The source JPEG is
        //    loaded via `CGImageSource` (which does not apply EXIF, matching msplat's loader
        //    exactly) to guarantee sensor-native pixel layout — `CIImage(data:)` applies EXIF
        //    on some SDK versions, which would rotate inputCI relative to the mask.
        let maskCI = CIImage(cvPixelBuffer: maskBuffer)
        let blur = CIFilter.gaussianBlur()
        blur.inputImage = maskCI
        blur.radius = 8.0   // pixels of edge softness at source resolution (~4 px in trainer)
        let softMask = (blur.outputImage ?? maskCI).cropped(to: maskCI.extent)

        guard let cgSource = CGImageSourceCreateWithData(keyframe.jpeg as CFData, nil),
              let cgInput = CGImageSourceCreateImageAtIndex(cgSource, 0, nil) else { return nil }
        let inputCI = CIImage(cgImage: cgInput)
        let backgroundCI = CIImage(color: CIColor(red: CGFloat(bgR),
                                                  green: CGFloat(bgG),
                                                  blue: CGFloat(bgB)))
            .cropped(to: inputCI.extent)
        let blend = CIFilter.blendWithMask()
        blend.inputImage = inputCI
        blend.backgroundImage = backgroundCI
        blend.maskImage = softMask
        guard let composed = blend.outputImage else { return nil }

        // 3. Save the composed image AND the soft mask at SENSOR-NATIVE dimensions (no crop).
        //
        //    The earlier per-frame tight crop was the OOM root cause. msplat's
        //    `ensure_forward` reallocates 6 working buffers (`out_img`, `final_Ts`, `final_idx`,
        //    `loss_intermediates`, `ssim_h_buf`, `v_rendered` — ~152 bytes/pixel total) every
        //    time `(img_height, img_width)` changes between frames. With ~30 per-frame bboxes
        //    we'd reallocate ~30 distinct working-buffer sets; Metal can't release the old
        //    ones until their pending command buffers complete, so peak GPU memory ran into
        //    the GBs and iOS killed the process mid-densification.
        //
        //    Keeping all frames at the same dims (matching room mode) means `ensure_forward`
        //    allocates exactly once. The trainer still ignores background pixels via the mask-
        //    weighted loss kernel, so reconstruction quality doesn't degrade — only some
        //    per-step compute on regions whose loss-weight is zero.
        guard let fullCG = ciContext.createCGImage(composed, from: composed.extent) else { return nil }
        guard let jpeg = UIImage(cgImage: fullCG).jpegData(compressionQuality: 0.9) else { return nil }

        // The soft mask drives msplat's per-pixel loss weight via `mask_path` in transforms.json
        // — background pixels (mask=0) contribute neither loss nor gradient, so densification
        // stays confined to the object. The Gaussian-blurred edge yields a smooth weighting
        // transition across the silhouette (vs. a binary mask whose hard 0→1 cut still drives
        // some edge gradients through SSIM windows that span the boundary).
        guard let fullMaskCG = ciContext.createCGImage(softMask, from: softMask.extent) else { return nil }
        let maskPNG = UIImage(cgImage: fullMaskCG).pngData()

        let baked = CaptureKeyframe(jpeg: jpeg,
                                    width: keyframe.width,
                                    height: keyframe.height,
                                    transformRowMajor: keyframe.transformRowMajor,
                                    fx: keyframe.fx,
                                    fy: keyframe.fy,
                                    cx: keyframe.cx,
                                    cy: keyframe.cy,
                                    orientation: keyframe.orientation,
                                    depth: nil,
                                    maskPNG: maskPNG)

        // 4. Unproject masked depth pixels using the ORIGINAL (uncropped) intrinsics — world
        //    points are crop-invariant and live in the same ARKit world frame as the cameras.
        var points: [simd_float3] = []
        if let depth = keyframe.depth {
            points = unproject(depth: depth, maskBuffer: maskBuffer, keyframe: keyframe)
        }

        return MaskResult(keyframe: baked, points: points)
    }

    /// Picks the IndexSet of just the largest detected instance (by pixel count). Avoids the
    /// "object + table + hand" multi-mask soup that breaks consistent-background training.
    private static func dominantInstance(_ observation: VNInstanceMaskObservation) -> IndexSet {
        let instances = observation.allInstances
        if instances.count <= 1 { return instances }

        let buffer = observation.instanceMask
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let w = CVPixelBufferGetWidth(buffer)
        let h = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        guard let base = CVPixelBufferGetBaseAddress(buffer)?.assumingMemoryBound(to: UInt8.self) else {
            return instances
        }

        var counts: [Int: Int] = [:]
        for y in 0..<h {
            let row = y * stride
            for x in 0..<w {
                let idx = Int(base[row + x])
                if idx > 0 { counts[idx, default: 0] += 1 }
            }
        }
        if let largest = counts.max(by: { $0.value < $1.value }) {
            return IndexSet(integer: largest.key)
        }
        return instances
    }

    /// Tightest axis-aligned bbox of foreground pixels in the mask, with `marginRatio` slack
    /// on each side, clipped to the buffer bounds. Returns nil if the mask is empty.
    private static func tightCrop(maskBuffer: CVPixelBuffer, marginRatio: Float) -> (x: Int, y: Int, width: Int, height: Int)? {
        CVPixelBufferLockBaseAddress(maskBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(maskBuffer, .readOnly) }
        let w = CVPixelBufferGetWidth(maskBuffer)
        let h = CVPixelBufferGetHeight(maskBuffer)
        let stride = CVPixelBufferGetBytesPerRow(maskBuffer)
        guard let base = CVPixelBufferGetBaseAddress(maskBuffer)?.assumingMemoryBound(to: UInt8.self) else {
            return nil
        }

        var minX = w, maxX = -1, minY = h, maxY = -1
        for y in 0..<h {
            let row = y * stride
            for x in 0..<w {
                if base[row + x] >= 128 {
                    if x < minX { minX = x }
                    if x > maxX { maxX = x }
                    if y < minY { minY = y }
                    if y > maxY { maxY = y }
                }
            }
        }
        guard maxX >= minX, maxY >= minY else { return nil }

        let bw = maxX - minX + 1
        let bh = maxY - minY + 1
        let mx = Int(Float(bw) * marginRatio)
        let my = Int(Float(bh) * marginRatio)
        let x0 = max(0, minX - mx)
        let y0 = max(0, minY - my)
        let x1 = min(w, maxX + 1 + mx)
        let y1 = min(h, maxY + 1 + my)
        return (x0, y0, x1 - x0, y1 - y0)
    }

    /// Walks the depth map, keeps only depth pixels whose foreground-mask sample is set and whose
    /// confidence is at least medium, and unprojects each surviving pixel to world space.
    private static func unproject(depth: DepthSnapshot, maskBuffer: CVPixelBuffer,
                                  keyframe: CaptureKeyframe) -> [simd_float3] {
        CVPixelBufferLockBaseAddress(maskBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(maskBuffer, .readOnly) }
        let maskW = CVPixelBufferGetWidth(maskBuffer)
        let maskH = CVPixelBufferGetHeight(maskBuffer)
        let maskStride = CVPixelBufferGetBytesPerRow(maskBuffer)
        guard let maskBase = CVPixelBufferGetBaseAddress(maskBuffer)?.assumingMemoryBound(to: UInt8.self) else {
            return []
        }

        let imgW = Float(keyframe.width)
        let imgH = Float(keyframe.height)
        let fx = keyframe.fx, fy = keyframe.fy, cx = keyframe.cx, cy = keyframe.cy
        let m = keyframe.transformRowMajor
        var points: [simd_float3] = []
        points.reserveCapacity(depth.width * depth.height / 4)

        for dv in 0..<depth.height {
            let rowBase = dv * depth.width
            for du in 0..<depth.width {
                let d = depth.values[rowBase + du]
                // smoothedSceneDepth is z-distance in metres; gate against junk values and far hits.
                guard d.isFinite, d > 0.1, d < 5.0 else { continue }
                if depth.confidence[rowBase + du] < 1 { continue }

                // Depth map shares the captured image's orientation, so just scale up to image pixels.
                let u = (Float(du) + 0.5) * imgW / Float(depth.width)
                let v = (Float(dv) + 0.5) * imgH / Float(depth.height)
                let mu = Int(u * Float(maskW) / imgW)
                let mv = Int(v * Float(maskH) / imgH)
                guard mu >= 0, mu < maskW, mv >= 0, mv < maskH else { continue }
                if maskBase[mv * maskStride + mu] < 128 { continue }

                // Pinhole unprojection in ARKit/OpenGL camera convention (image y is down -> camera y is up).
                let cX = (u - cx) * d / fx
                let cY = -(v - cy) * d / fy
                let cZ: Float = -d
                points.append(simd_float3(
                    m[0]  * cX + m[1]  * cY + m[2]  * cZ + m[3],
                    m[4]  * cX + m[5]  * cY + m[6]  * cZ + m[7],
                    m[8]  * cX + m[9]  * cY + m[10] * cZ + m[11]
                ))
            }
        }
        return points
    }

    /// EXIF orientation inverse, used to undo Vision's mask reorientation back to sensor-native.
    private static func inverseOrientation(_ o: CGImagePropertyOrientation) -> CGImagePropertyOrientation {
        switch o {
        case .up:            return .up
        case .right:         return .left
        case .down:          return .down
        case .left:          return .right
        case .upMirrored:    return .upMirrored
        case .downMirrored:  return .downMirrored
        case .leftMirrored:  return .rightMirrored
        case .rightMirrored: return .leftMirrored
        }
    }

    /// Renders a single-channel CIImage into a freshly-allocated grayscale CVPixelBuffer.
    /// Used to bake a reoriented mask back into a buffer the depth-unprojection and tight-crop
    /// helpers can sample directly (they walk raw pixel rows, not CIImage extents).
    private static func renderMaskBuffer(_ ci: CIImage, width: Int, height: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs: CFDictionary = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        let status = CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_OneComponent8, attrs, &pb)
        guard status == kCVReturnSuccess, let pb else { return nil }
        ciContext.render(ci, to: pb)
        return pb
    }

    /// Deduplicates points onto a `voxelSize`-metre grid. Trims 100k+-scale raw clouds to a few
    /// thousand well-spaced init points without losing the object's geometric extent.
    static func voxelize(_ points: [simd_float3], voxelSize: Float) -> [simd_float3] {
        var seen = Set<SIMD3<Int32>>()
        seen.reserveCapacity(points.count / 4)
        var out: [simd_float3] = []
        out.reserveCapacity(points.count / 4)
        for p in points {
            let key = SIMD3<Int32>(Int32(p.x / voxelSize),
                                   Int32(p.y / voxelSize),
                                   Int32(p.z / voxelSize))
            if seen.insert(key).inserted {
                out.append(p)
            }
        }
        return out
    }
}

// MARK: - Training

/// Drives msplat's trainer step-by-step on a background task, publishing progress for the UI and
/// exporting a PLY when finished. The dataset write happens here too so all heavy work stays off
/// the main actor.
@MainActor
final class GaussianTrainingController: ObservableObject {
    enum Phase: Equatable {
        case idle
        case preparing
        case training
        case finished(URL)
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var iteration = 0
    @Published private(set) var splatCount = 0

    let totalIterations: Int
    /// Image downscale divisor applied at load (2.0 = half resolution), to keep memory and step
    /// time reasonable on-device. Raise for quality, lower toward 1.0 for full resolution.
    let imageDownscale: Float

    private var task: Task<Void, Never>?

    /// 6000 iterations: enough for splats to shrink past their fat initial covariance and for
    /// opacities to converge after the alpha-reset cycle (`refineEvery * resetAlphaEvery = 3000`
    /// — training that stops AT that boundary leaves splats with freshly-reset opacities, which
    /// looks like the colorful "splat soup" before geometry has crystallised). Standard 3DGS
    /// runs 30k; 6k is the on-device pragmatic point that pairs well with the LiDAR-seeded
    /// init density and the per-pixel mask-weighted loss (background pixels contribute zero
    /// gradient, so densification stays naturally bounded and peak memory is predictable).
    init(totalIterations: Int = 6000, imageDownscale: Float = 2.0) {
        self.totalIterations = totalIterations
        self.imageDownscale = imageDownscale
    }

    var progress: Double {
        totalIterations > 0 ? min(1, Double(iteration) / Double(totalIterations)) : 0
    }

    func cancel() { task?.cancel() }

    func start(keyframes: [CaptureKeyframe], points: [simd_float3],
               workDir: URL, outputPLY: URL,
               bgColor: (Float, Float, Float)? = nil,
               densifyGradThresh: Float? = nil,
               cullAlphaThresh: Float? = nil) {
        guard task == nil else { return }
        phase = .preparing
        let total = totalIterations
        let downscale = imageDownscale

        task = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                try NerfstudioDatasetWriter.write(keyframes: keyframes, points: points, to: workDir)
            } catch {
                await self?.set(phase: .failed("Couldn't write the capture dataset: \(error.localizedDescription)"))
                return
            }

            let dataset = GaussianDataset(path: workDir.path, downscaleFactor: downscale, evalMode: false)
            guard dataset.numTrain > 0 else {
                await self?.set(phase: .failed("No training views were loaded from the capture."))
                return
            }

            var config = TrainingConfig()
            config.iterations = Int32(total)
            if let bgColor { config.bgColor = bgColor }
            // Object-mode tuning: mask-weighted loss concentrates the gradient signal on the
            // foreground subset (~20-30% of pixels), so msplat's `densifyGradThresh` and
            // `cullAlphaThresh` — both calibrated for full-image standard 3DGS — need to be
            // relaxed to compensate. Without this the trainer plateaus at the init splat
            // count: too few splats cross the densify threshold, and the post-reset opacity
            // sits right at the cull floor so any borderline splat gets pruned immediately.
            if let densifyGradThresh { config.densifyGradThresh = densifyGradThresh }
            if let cullAlphaThresh { config.cullAlphaThresh = cullAlphaThresh }
            let trainer = GaussianTrainer(dataset: dataset, config: config)
            await self?.set(phase: .training)

            while !Task.isCancelled && trainer.iteration < total {
                let stats = trainer.step()
                if stats.iteration % 10 == 0 {
                    await self?.update(iteration: stats.iteration, splatCount: stats.splatCount)
                }
            }

            // Export whatever has been trained so far, even if the user cancelled early.
            trainer.exportPly(to: outputPLY.path)
            msplatSync()
            await self?.update(iteration: trainer.iteration, splatCount: trainer.splatCount)
            await self?.set(phase: .finished(outputPLY))
        }
    }

    private func update(iteration: Int, splatCount: Int) {
        self.iteration = iteration
        self.splatCount = splatCount
    }

    private func set(phase: Phase) {
        self.phase = phase
    }
}
#endif // os(iOS)
