# ai4anim-webgpu

A TypeScript three.js WebGPU inference port of the biped + quadruped neural-motion-matching demos from [facebookresearch/ai4animationpy](https://github.com/facebookresearch/ai4animationpy) by [Paul Starke](https://github.com/paulstarke) and [Sebastian Starke](https://github.com/sebastianstarke).

**Live demo:** [motionsynth.sweriko.com](https://motionsynth.sweriko.com)

<video src="https://github.com/sweriko/ai4anim-webgpu/raw/main/docs/neuralmotion.mp4" controls muted playsinline width="100%"></video>

## What it does

- Runs the upstream biped + quadruped neural networks directly in the browser
- Batched compute: one dispatch per prediction tick across every active agent, with mixed-precision matmul kernels (fp16 weights, fp32 accumulator).
- All agents share a single instanced `SkinnedMesh`
- basic controll interface

## setup

```bash
npm install
npm run dev     
```

## Attribution & license

This is a derivative work of AI4AnimationPy. See [`NOTICE`](./NOTICE) for full attribution and [`LICENSE`](./LICENSE) for the license terms (CC BY-NC 4.0 — non-commercial use only). The "ai4anim-" prefix is descriptive; this project is not affiliated with or endorsed by Meta or the original authors.
