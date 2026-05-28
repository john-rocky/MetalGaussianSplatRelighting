# Metal Gaussian Splat Relighting

**Real-time relightable 3D Gaussian Splatting on Apple platforms, in Swift + Metal.**

<p align="center">
  <img src="https://github.com/user-attachments/assets/c34f552e-cc8b-47aa-bbb9-c10583419a7e" width="300" alt="Live relighting: rotating the HDR environment and the reflections respond on-device">
</p>

Load a [Ref-Gaussian](https://github.com/fudan-zvg/ref-gaussian)–trained scene and relight it live on an
iPhone: swap and rotate the HDR environment and watch the reflections and shading respond, with the
environment drawn behind the object as a skybox. Or drop the object into your real room in **AR**, lit by
the room's actual lighting and composited over the live camera. Built on top of
[scier/MetalSplatter](https://github.com/scier/MetalSplatter) (which renders splats but does not relight),
this fork adds a full **split-sum image-based-lighting (IBL)** pipeline and **deferred physically-based
shading** for relightable assets.

> Status: research/portfolio project. The relighting algorithms are from published work (Ref-Gaussian,
> UE4 split-sum IBL); the contribution here is porting and optimizing them into a real-time on-device
> Metal pipeline.

## What it adds over MetalSplatter

- **Split-sum IBL relighting** — GPU-precomputed prefiltered specular cubemap + diffuse irradiance
  cubemap + BRDF (DFG) integration LUT, evaluated per pixel.
- **Deferred PBR shading** — per-splat material (normal / roughness / reflection strength / albedo) is
  blended into a tile-memory G-buffer, then shaded once per pixel. This matches Ref-Gaussian's
  `render_surfel`: `final = (1 − reflectance)·base_color + specular`, with the specular F0 tinted by the
  learned `ori_color` albedo.
- **HDR environment + skybox** — load an equirectangular HDR; it lights the object *and* is drawn as the
  background (reconstructed per pixel from the inverse view-projection), so reflections match the scene
  behind the object. Switch between bundled environments and rotate them live.
- **AR mode** — place the relit splat in your real room with ARKit: the live camera feed is the
  background, the object is lit by (and reflects) the room via ARKit environment probes, and it sits on
  the detected floor at **real-world scale** (with a soft contact shadow) so you can walk around it.
  Tap to place, pinch to resize, drag to spin.
- **Live configurator** — repaint the object to a new color and switch its finish (matte / glossy /
  mirror / metallic). The recolor changes only the body's hue (keeping its shading, and leaving white /
  neutral markings untouched), so the *same captured asset* shows in a new color and is still correctly
  relit and reflects the environment — something baked-lighting captures can't do. The natural
  AR-commerce "see the variant you'd buy, in your room" feature.
- **Ref-Gaussian 2D-surfel assets** — SplatIO reads the non-standard 281-float `.ply` (per-splat PBR
  material, surfel normal reconstructed from the rotation quaternion).
- **Interactive orbit camera** — drag to rotate, pinch to zoom, with a Z-up→Y-up calibration for
  Ref-NeRF / Blender datasets.

## How it works

```
Ref-Gaussian .ply ──▶ SplatIO ──▶ per-splat material (N, roughness, reflectance, albedo)
                                        │
HDR equirect ──▶ IBL precompute ──▶ prefiltered cube + irradiance cube + BRDF LUT
                                        │
                          ┌─────────────┴───────────────┐
                          ▼                              ▼
              multi-stage G-buffer pass         postprocess pass
           (blend color/normal/material      (per-pixel split-sum IBL +
            into tile imageblock memory)       skybox composite)
```

Reflections and the skybox sample the same environment in a single consistent frame, so they always
agree. Normals are oriented per-pixel toward the camera (matching Ref-Gaussian's `flip_align_view`).

## AR mode

Selecting **AR / Room** runs an `ARWorldTrackingConfiguration` with automatic environment texturing and
reuses the same deferred-relighting pipeline:

- ARKit's `AREnvironmentProbeAnchor` provides a full surround **cubemap** of the room, which is fed
  straight into the IBL precompute — so the object is lit by, and reflects, your actual room.
- The captured camera frame (YCbCr) is converted to a linear-RGB background and composited *in the same
  resolve pass* that normally draws the skybox — splats sit over the live camera with no extra blend.
- The model is placed on a detected horizontal plane by raycast (auto under the screen center, or
  tap-to-place) and viewed through the `ARCamera` view/projection, so it holds its spot on the floor as
  you walk around it. It is normalized from its (outlier-robust) bounds and scaled to a real-world size
  in meters, so it appears life-size; pinch resizes and drag spins it.
- A soft elliptical **contact shadow** (sized to the model's footprint) is blended onto the floor under
  the model before the splats composite over it, so it reads as grounded rather than floating.

A note on correctness: a single rear camera only sees its frustum, so reflections of directions behind
the camera can't be measured — ARKit fills those in with machine-learning estimation. This reads well on
glossy/rough materials (the reflection is blurred anyway) and improves as you look around the room.

## Build & run

1. Open `SampleApp/MetalSplatter_SampleApp.xcodeproj`.
2. For iOS, set your development team and bundle ID under Signing & Capabilities.
3. Build & run (Release recommended; large files load far faster than in Debug).
4. Open a Ref-Gaussian `.ply`. Use the on-screen panel to toggle relighting, swap/rotate the
   environment, adjust intensity, and inspect debug channels (normal / roughness / reflectance /
   albedo / prefiltered env / irradiance).
5. Pick the **AR / Room** environment to place the object in your real room (the app requests camera
   access). Point at the floor so it drops in at real size (or tap to place), then walk around it;
   pinch to resize and drag to spin. Keep relighting on. Life-size objects need open space.

A reflective object (e.g. the Shiny-Blender `car` or a chrome sphere) shows the relighting best; a matte
object intentionally reflects very little.

## Credits & license

- Built on [scier/MetalSplatter](https://github.com/scier/MetalSplatter) (MIT). The original library,
  PLYIO, SplatIO, and sample-app scaffolding are by [Sean Cier](https://github.com/scier).
- Relighting model from [Ref-Gaussian](https://github.com/fudan-zvg/ref-gaussian); split-sum IBL from
  Karis / Epic Games (UE4).
- Bundled HDR environments from [Poly Haven](https://polyhaven.com) (CC0).

Licensed under the MIT License (see [LICENSE](LICENSE)), preserving the upstream copyright.
