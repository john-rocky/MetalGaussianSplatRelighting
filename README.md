# Metal Gaussian Splat Relighting

**Real-time relightable 3D Gaussian Splatting on Apple platforms, in Swift + Metal.**

Load a [Ref-Gaussian](https://github.com/fudan-zvg/ref-gaussian)–trained scene and relight it live on an
iPhone: swap and rotate the HDR environment and watch the reflections and shading respond, with the
environment drawn behind the object as a skybox. Built on top of
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

## Build & run

1. Open `SampleApp/MetalSplatter_SampleApp.xcodeproj`.
2. For iOS, set your development team and bundle ID under Signing & Capabilities.
3. Build & run (Release recommended; large files load far faster than in Debug).
4. Open a Ref-Gaussian `.ply`. Use the on-screen panel to toggle relighting, swap/rotate the
   environment, adjust intensity, and inspect debug channels (normal / roughness / reflectance /
   albedo / prefiltered env / irradiance).

A reflective object (e.g. the Shiny-Blender `car` or a chrome sphere) shows the relighting best; a matte
object intentionally reflects very little.

## Credits & license

- Built on [scier/MetalSplatter](https://github.com/scier/MetalSplatter) (MIT). The original library,
  PLYIO, SplatIO, and sample-app scaffolding are by [Sean Cier](https://github.com/scier).
- Relighting model from [Ref-Gaussian](https://github.com/fudan-zvg/ref-gaussian); split-sum IBL from
  Karis / Epic Games (UE4).
- Bundled HDR environments from [Poly Haven](https://polyhaven.com) (CC0).

Licensed under the MIT License (see [LICENSE](LICENSE)), preserving the upstream copyright.
