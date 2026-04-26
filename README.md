# ai4anim-webgpu

A TypeScript three.js WebGPU inference port of the biped + quadruped neural-motion-matching demos from [facebookresearch/ai4animationpy](https://github.com/facebookresearch/ai4animationpy) by [Paul Starke](https://github.com/paulstarke) and [Sebastian Starke](https://github.com/sebastianstarke).

**Live demo:** [motionsynth.sweriko.com](https://motionsynth.sweriko.com)

<img width="1375" height="773" alt="image" src="https://github.com/user-attachments/assets/ac3abd00-18ba-4e70-97be-1c96e9805598" />


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
