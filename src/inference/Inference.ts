/** TSL-based batched WebGPU inference with vec4 matmul and optional fp16 weights.
 *
 *  Pipeline (deterministic, matches Program.py with noise=0 seed=0):
 *    1. x_norm = (x - inMean) / inStd                       [IN]
 *    2. h1 = ELU(Linear(x_norm, est_L1))                    [HIDDEN]
 *    3. h2 = ELU(Linear(h1, est_L2))                        [HIDDEN]
 *    4. p  = Softmax_perGroup(Linear(h2, est_L3), 16)       [LATENT]
 *    5. repeat denoiser_iterations (=3):
 *         xin = concat(p, x_norm)                           [LATENT + IN]
 *         h1  = ELU(Linear(xin, den_L1))
 *         h2  = ELU(Linear(h1,  den_L2))
 *         p   = Softmax_perGroup(Linear(h2, den_L3), 16)
 *    6. dec_in = concat(p, x_norm)
 *    7. Decoder (batched over SEQ timesteps, FiLM on time):
 *         y1 = ELU(FiLM_Linear(dec_in, dec_L1))
 *         y2 = ELU(FiLM_Linear(y1,     dec_L2))
 *         y3 =     FiLM_Linear(y2,     dec_L3)
 *    8. y = y3 * outStd + outMean                           [SEQ, OUT]
 *
 *  Every tensor has a leading agent dim. One dispatch per kernel covers
 *  every active agent at once.
 *
 *  Matmul:
 *    - Weight matrices are packed row-vec4-aligned. Inner dim rounded up
 *      to a multiple of 4 with trailing zeros (contributes 0 to the dot).
 *    - Kernel's inner loop iterates `ceil(inDim / 4)` times; each iteration
 *      reads one vec4 of weights and one vec4 of activation (built from
 *      4 consecutive scalar x reads), then `acc += dot(xv, wv)`.
 *    - Per-agent input / concat scratch buffers use the padded stride so
 *      consecutive-scalar reads within an iteration stay inside the agent.
 *    - Precision switch: fp32 weights stored as `array<vec4<f32>>`;
 *      fp16 weights stored as `array<u32>` (two halves per u32), unpacked
 *      with `unpackHalf2x16` on read. Accumulator is f32 in both.
 *
 *  fp16 does NOT require the `shader-f16` adapter feature — `unpack2x16float`
 *  is a core WGSL builtin available on every WebGPU adapter.
 */

import {
  Fn, If, Loop, dot, exp, float, instanceIndex, select, storage, uint,
  unpackHalf2x16, vec4,
} from "three/tsl";
import type { WebGPURenderer, ComputeNode, StorageBufferNode } from "three/webgpu";
import { StorageBufferAttribute } from "three/webgpu";
import { Bundle } from "../model/bundle.js";
import {
  packWeight, padScalarF32, pad4, vec4Count,
  type WeightPrecision, type PackedWeight,
} from "./weights.js";

export type { WeightPrecision };

export interface InferenceOptions {
  /** Max simultaneously active agents (buffers are pre-allocated for this). */
  maxAgents?: number;
  /** Denoiser iteration count (default from bundle meta). */
  iterations?: number;
  /** Weight precision. Default fp16. fp16 uses packed u32 + unpackHalf2x16,
   *  so it requires no adapter feature beyond core WebGPU. */
  precision?: WeightPrecision;
}

const id = (s: string) => s.replace(/[^A-Za-z0-9_]/g, "_");

function scratchF32(count: number, label: string): {
  node: StorageBufferNode<"float">; attr: StorageBufferAttribute;
} {
  const attr = new StorageBufferAttribute(count, 1);
  const node = storage(attr, "float", count);
  node.setName(id(label));
  return { node, attr };
}

function readOnlyF32(data: Float32Array, label: string): StorageBufferNode<"float"> {
  const attr = new StorageBufferAttribute(data, 1);
  const node = storage(attr, "float", data.length).toReadOnly();
  node.setName(id(label));
  return node;
}

/** Upload a packed weight buffer as the right TSL storage type. */
function uploadWeight(
  pw: PackedWeight, rows: number, label: string,
): { node: StorageBufferNode<"vec4"> | StorageBufferNode<"uint">; kind: WeightPrecision } {
  if (pw.precision === "fp32") {
    const attr = new StorageBufferAttribute(pw.data as Float32Array, 4);
    const count = rows * pw.inDimV4;
    const node = storage(attr, "vec4", count).toReadOnly();
    node.setName(id(label));
    return { node, kind: "fp32" };
  }
  // fp16 — one u32 per element, 2 halves per u32.
  const attr = new StorageBufferAttribute(pw.data as Uint32Array, 1);
  const count = rows * pw.u32PerRow;
  const node = storage(attr, "uint", count).toReadOnly();
  node.setName(id(label));
  return { node, kind: "fp16" };
}

export class Inference {
  readonly bundle: Bundle;
  readonly renderer: WebGPURenderer;
  readonly iterations: number;
  readonly maxAgents: number;
  readonly precision: WeightPrecision;

  readonly IN: number;
  readonly OUT: number;
  readonly LATENT: number;
  readonly HIDDEN: number;
  readonly SEQ: number;
  readonly GROUP: number;

  /** Padded per-agent strides (multiples of 4) — kernels use these when a
   *  tensor feeds a vec4 matmul. */
  readonly paddedIN: number;
  readonly paddedCONCAT: number;
  readonly paddedDEC_IN: number;   // LATENT + IN = CONCAT

  private readonly pipeline: ComputeNode[];
  /** Each kernel's per-agent work-unit count. Used to rescale `.count` on
   *  each run() so idle slots contribute zero GPU work and zero dispatched
   *  workgroups. Three handles the rest via its early-return machinery +
   *  dispatch-size cache in WebGPUBackend. */
  private readonly perAgentScale: number[];
  /** Active-agent count the pipeline was last sized for. Avoids redundant
   *  `.count` writes on every frame when the roster didn't change. */
  private lastActiveCount = 0;

  /** Raw input view into xRaw (padded stride). Agents write their slice here. */
  readonly xRawArray: Float32Array;
  /** Padded per-agent input stride (in float units). Agents write at slot*inputStride. */
  readonly inputStride: number;

  private readonly xRawAttr: StorageBufferAttribute;
  readonly decOutAttr: StorageBufferAttribute;

  static async create(
    bundle: Bundle, renderer: WebGPURenderer, opts: InferenceOptions = {},
  ): Promise<Inference> {
    const self = new Inference(bundle, renderer, opts);
    const dumpTarget = (globalThis as unknown as { __dumpWGSL?: string }).__dumpWGSL;
    if (dumpTarget) {
      const backend = renderer.backend as unknown as {
        createProgram(p: { code: string; stage: string; name: string }): void;
      };
      const orig = backend.createProgram.bind(backend);
      backend.createProgram = (program) => {
        const tag = `${program.stage}${program.name ? "_" + program.name : ""}`;
        if (tag.includes(dumpTarget)) {
          console.log(`[wgsl ${tag}]\n${program.code}\n[/wgsl]`);
        }
        return orig(program);
      };
    }
    await renderer.computeAsync(self.pipeline);
    return self;
  }

  private constructor(
    bundle: Bundle, renderer: WebGPURenderer, opts: InferenceOptions,
  ) {
    this.bundle = bundle;
    this.renderer = renderer;
    this.iterations = opts.iterations ?? bundle.meta.model.denoiser_iterations;
    this.maxAgents = Math.max(1, opts.maxAgents ?? 1);
    this.precision = opts.precision ?? "fp16";

    const m = bundle.meta.model;
    this.IN = m.input_dim;
    this.OUT = m.output_dim;
    this.LATENT = m.latent_dim;
    this.HIDDEN = m.latent_dim;
    this.SEQ = m.sequence_length;
    this.GROUP = m.codebook_dims;

    this.paddedIN = pad4(this.IN);
    this.paddedCONCAT = pad4(this.LATENT + this.IN);
    this.paddedDEC_IN = this.paddedCONCAT;
    this.inputStride = this.paddedIN;

    const N = this.maxAgents;
    const prec = this.precision;

    // -- Weights: packed (vec4-aligned rows, optional fp16 halves) ----------
    const WN: Record<string, StorageBufferNode<"vec4"> | StorageBufferNode<"uint">> = {};
    const pack = (name: string, rows: number, inDim: number, label: string) => {
      const pw = packWeight(bundle.get(name), rows, inDim, prec);
      WN[name] = uploadWeight(pw, rows, label).node;
    };
    pack("estimator.l1.weight", this.HIDDEN, this.IN, "estW_l1");
    pack("estimator.l2.weight", this.HIDDEN, this.HIDDEN, "estW_l2");
    pack("estimator.l3.weight", this.LATENT, this.HIDDEN, "estW_l3");
    pack("denoiser.l1.weight", this.HIDDEN, this.IN + this.LATENT, "denW_l1");
    pack("denoiser.l2.weight", this.HIDDEN, this.HIDDEN, "denW_l2");
    pack("denoiser.l3.weight", this.LATENT, this.HIDDEN, "denW_l3");
    pack("decoder.l1.linear.weight", this.HIDDEN, this.IN + this.LATENT, "decW_l1");
    pack("decoder.l2.linear.weight", this.HIDDEN, this.HIDDEN, "decW_l2");
    pack("decoder.l3.linear.weight", this.OUT, this.HIDDEN, "decW_l3");

    // Biases stay fp32 scalar (small, first accumulator touch).
    const B: Record<string, StorageBufferNode<"float">> = {};
    for (const who of ["estimator", "denoiser"]) {
      for (const L of ["l1", "l2", "l3"]) {
        B[`${who}.${L}.bias`] = readOnlyF32(bundle.get(`${who}.${L}.bias`), `${who}_${L}_b`);
      }
    }
    for (const L of ["l1", "l2", "l3"]) {
      B[`decoder.${L}.linear.bias`] = readOnlyF32(
        bundle.get(`decoder.${L}.linear.bias`), `dec_${L}_b`);
    }

    // FiLM coefficients — scalar fp32, padded to multiple of 4 (small, but
    // matmul inner loop reads them in vec4 chunks).
    const F: Record<string, StorageBufferNode<"float">> = {};
    const padFilm = (name: string, n: number, label: string): StorageBufferNode<"float"> => {
      const padded = padScalarF32(bundle.get(name), n);
      return readOnlyF32(padded, label);
    };
    const filmDim = { l1: this.IN + this.LATENT, l2: this.HIDDEN, l3: this.HIDDEN };
    for (const L of ["l1", "l2", "l3"] as const) {
      for (const who of ["scale", "shift"] as const) {
        F[`decoder.${L}.film.${who}.weight`] = padFilm(
          `decoder.${L}.film.${who}.weight`, filmDim[L], `dec_${L}_${who}_W`);
        F[`decoder.${L}.film.${who}.bias`] = padFilm(
          `decoder.${L}.film.${who}.bias`, filmDim[L], `dec_${L}_${who}_b`);
      }
    }

    // Stats — fp32 scalar.
    const inMean = readOnlyF32(bundle.get("input_stats.mean"), "inMean");
    const inStd  = readOnlyF32(bundle.get("input_stats.std"),  "inStd");
    const outMean = readOnlyF32(bundle.get("output_stats.mean"), "outMean");
    const outStd  = readOnlyF32(bundle.get("output_stats.std"),  "outStd");

    // Pre-normalize the 16 timestamps once.
    const rawT = bundle.get("timing");
    const tMean = bundle.get("time_stats.mean")[0];
    const tStd  = bundle.get("time_stats.std")[0];
    const tNormalized = new Float32Array(rawT.length);
    for (let i = 0; i < rawT.length; i++) tNormalized[i] = (rawT[i] - tMean) / tStd;
    const timings = readOnlyF32(tNormalized, "timings_norm");

    // -- Scratch buffers. Per-agent strides marked padded where vec4 reads --
    this.xRawAttr = new StorageBufferAttribute(N * this.paddedIN, 1);
    this.xRawArray = this.xRawAttr.array as Float32Array;
    const xRaw = storage(this.xRawAttr, "float", N * this.paddedIN);
    xRaw.setName("xRaw");

    const { node: xNorm }  = scratchF32(N * this.paddedIN, "xNorm");
    const { node: h1 }     = scratchF32(N * this.HIDDEN, "h1");
    const { node: h2 }     = scratchF32(N * this.HIDDEN, "h2");
    const { node: p }      = scratchF32(N * this.LATENT, "p");
    const { node: concat } = scratchF32(N * this.paddedCONCAT, "concat");
    const { node: dec1 }   = scratchF32(N * this.SEQ * this.HIDDEN, "dec1");
    const { node: dec2 }   = scratchF32(N * this.SEQ * this.HIDDEN, "dec2");

    this.decOutAttr = new StorageBufferAttribute(N * this.SEQ * this.OUT, 1);
    const decOut = storage(this.decOutAttr, "float", N * this.SEQ * this.OUT);
    decOut.setName("decOut");

    // -- Compute graph -----------------------------------------------------
    // `collect(node, perAgent)` stores the kernel + its per-agent scale so
    // run() can rescale `.count` with the active agent count each frame.
    const pipeline: ComputeNode[] = [];
    const perAgentScale: number[] = [];
    const collect = (node: ComputeNode, perAgent: number): void => {
      pipeline.push(node);
      perAgentScale.push(perAgent);
    };

    // Normalize writes to the padded xNorm using the real IN stride only for
    // the live slots. Padding slots stay zero-initialized.
    collect(this.buildNormalizePadded(
      "normalize", this.IN, this.paddedIN, xRaw, inMean, inStd, xNorm), this.IN);

    // Estimator (inDim is paddedIN, HIDDEN, HIDDEN).
    collect(this.buildLinear("est_l1",
      this.paddedIN, this.HIDDEN, xNorm,
      WN["estimator.l1.weight"]!, B["estimator.l1.bias"]!, h1, true), this.HIDDEN);
    collect(this.buildLinear("est_l2",
      this.HIDDEN, this.HIDDEN, h1,
      WN["estimator.l2.weight"]!, B["estimator.l2.bias"]!, h2, true), this.HIDDEN);
    collect(this.buildLinear("est_l3",
      this.HIDDEN, this.LATENT, h2,
      WN["estimator.l3.weight"]!, B["estimator.l3.bias"]!, p, false), this.LATENT);
    collect(this.buildSoftmaxInPlace("est_softmax", this.LATENT, this.GROUP, p),
      this.LATENT / this.GROUP);

    for (let it = 0; it < this.iterations; it++) {
      collect(this.buildConcatPadded(`den${it}_concat`,
        this.LATENT, this.IN, this.paddedIN, this.paddedCONCAT, p, xNorm, concat),
        this.LATENT + this.IN);
      collect(this.buildLinear(`den${it}_l1`,
        this.paddedCONCAT, this.HIDDEN, concat,
        WN["denoiser.l1.weight"]!, B["denoiser.l1.bias"]!, h1, true), this.HIDDEN);
      collect(this.buildLinear(`den${it}_l2`,
        this.HIDDEN, this.HIDDEN, h1,
        WN["denoiser.l2.weight"]!, B["denoiser.l2.bias"]!, h2, true), this.HIDDEN);
      collect(this.buildLinear(`den${it}_l3`,
        this.HIDDEN, this.LATENT, h2,
        WN["denoiser.l3.weight"]!, B["denoiser.l3.bias"]!, p, false), this.LATENT);
      collect(this.buildSoftmaxInPlace(`den${it}_softmax`, this.LATENT, this.GROUP, p),
        this.LATENT / this.GROUP);
    }

    collect(this.buildConcatPadded("dec_concat",
      this.LATENT, this.IN, this.paddedIN, this.paddedCONCAT, p, xNorm, concat),
      this.LATENT + this.IN);

    collect(this.buildFilmLinear("dec_l1",
      this.paddedCONCAT, this.HIDDEN, this.SEQ, true, true,
      concat, timings,
      F["decoder.l1.film.scale.weight"]!, F["decoder.l1.film.scale.bias"]!,
      F["decoder.l1.film.shift.weight"]!, F["decoder.l1.film.shift.bias"]!,
      WN["decoder.l1.linear.weight"]!,    B["decoder.l1.linear.bias"]!,
      dec1), this.SEQ * this.HIDDEN);
    collect(this.buildFilmLinear("dec_l2",
      this.HIDDEN, this.HIDDEN, this.SEQ, true, false,
      dec1, timings,
      F["decoder.l2.film.scale.weight"]!, F["decoder.l2.film.scale.bias"]!,
      F["decoder.l2.film.shift.weight"]!, F["decoder.l2.film.shift.bias"]!,
      WN["decoder.l2.linear.weight"]!,    B["decoder.l2.linear.bias"]!,
      dec2), this.SEQ * this.HIDDEN);
    collect(this.buildFilmLinear("dec_l3",
      this.HIDDEN, this.OUT, this.SEQ, false, false,
      dec2, timings,
      F["decoder.l3.film.scale.weight"]!, F["decoder.l3.film.scale.bias"]!,
      F["decoder.l3.film.shift.weight"]!, F["decoder.l3.film.shift.bias"]!,
      WN["decoder.l3.linear.weight"]!,    B["decoder.l3.linear.bias"]!,
      decOut), this.SEQ * this.OUT);

    collect(this.buildDenormalize("denormalize",
      this.OUT, this.SEQ * this.OUT, outMean, outStd, decOut),
      this.SEQ * this.OUT);

    this.pipeline = pipeline;
    this.perAgentScale = perAgentScale;
  }

  /** Per-agent output stride (floats). */
  get outputStride(): number { return this.SEQ * this.OUT; }

  // ---------------------------------------------------------------------
  // Kernel builders
  // ---------------------------------------------------------------------

  /** One thread per scalar in the output buffer; uses padded per-agent stride
   *  so xNorm's padding slots stay zero across agent boundaries. */
  private buildNormalizePadded(
    label: string, dim: number, paddedDim: number,
    x: StorageBufferNode<"float">, mean: StorageBufferNode<"float">,
    std: StorageBufferNode<"float">, y: StorageBufferNode<"float">,
  ): ComputeNode {
    const dimU = uint(dim);
    const paddedU = uint(paddedDim);
    const kernel = Fn(() => {
      // Dispatch count = N * dim (real floats only). Thread idx maps to a real
      // slot in (agent, feature-within-agent) space.
      const idx = instanceIndex;
      const agent = idx.div(dimU);
      const f = idx.mod(dimU);
      const dst = agent.mul(paddedU).add(f);
      const v = x.element(dst).sub(mean.element(f)).div(std.element(f));
      y.element(dst).assign(v);
    })().compute(this.maxAgents * dim, [64]);
    kernel.setName(id(label));
    return kernel;
  }

  /** y[idx] = y[idx] * std[idx%dim] + mean[idx%dim]  — y is SEQ*OUT per agent,
   *  no padding needed since OUT=352 is already a multiple of 4 and SEQ*OUT=5632. */
  private buildDenormalize(
    label: string, dim: number, total: number,
    mean: StorageBufferNode<"float">, std: StorageBufferNode<"float">, y: StorageBufferNode<"float">,
  ): ComputeNode {
    const dimU = uint(dim);
    const kernel = Fn(() => {
      const i = instanceIndex;
      const f = i.mod(dimU);
      const v = y.element(i).mul(std.element(f)).add(mean.element(f));
      y.element(i).assign(v);
    })().compute(this.maxAgents * total, [64]);
    kernel.setName(id(label));
    return kernel;
  }

  /** Concat(p[N,aLen], xNorm[N, paddedB]) → concat[N, paddedOut].  `aLen` is
   *  the real latent count; `bLen` is the real input count; padded strides
   *  for B and O are supplied explicitly. Dispatches N * (aLen + bLen) real
   *  slots — unused padding slots in `O` stay zero from initial allocation. */
  private buildConcatPadded(
    label: string, aLen: number, bLen: number,
    paddedB: number, paddedOut: number,
    A: StorageBufferNode<"float">, B: StorageBufferNode<"float">, O: StorageBufferNode<"float">,
  ): ComputeNode {
    const real = aLen + bLen;
    const realU = uint(real);
    const aLenU = uint(aLen);
    const paddedBU = uint(paddedB);
    const paddedOutU = uint(paddedOut);
    const kernel = Fn(() => {
      const idx = instanceIndex;
      const agent = idx.div(realU);
      const pos = idx.mod(realU);
      const dst = agent.mul(paddedOutU).add(pos);
      If(pos.lessThan(aLenU), () => {
        O.element(dst).assign(A.element(agent.mul(aLenU).add(pos)));
      }).Else(() => {
        O.element(dst).assign(B.element(agent.mul(paddedBU).add(pos.sub(aLenU))));
      });
    })().compute(this.maxAgents * real, [64]);
    kernel.setName(id(label));
    return kernel;
  }

  /** Vec4-packed matmul. Per-thread: y[agent, o] = bias[o] + Σ_i dot(xv4[i], wv4[o,i]).
   *
   *  `inStride` is the per-agent padded stride of `x` (multiple of 4). Trailing
   *  zeros in W's padded rows contribute nothing to the dot, so the loop
   *  bound is just `inStride / 4` — the real inDim is recoverable from
   *  inStride - (0..3) and doesn't need to be threaded through. */
  private buildLinear(
    label: string,
    inStride: number, outDim: number,
    x: StorageBufferNode<"float">,
    W: StorageBufferNode<"vec4"> | StorageBufferNode<"uint">,
    b: StorageBufferNode<"float">,
    y: StorageBufferNode<"float">,
    activate: boolean,
  ): ComputeNode {
    const inStrideU = uint(inStride);
    const outDimU = uint(outDim);
    const inDimV4 = vec4Count(inStride);        // loop iteration count (paddedIn/4)
    const precision: WeightPrecision = this.precision;

    const kernel = Fn(() => {
      const idx = instanceIndex;
      const batch = idx.div(outDimU);
      const o = idx.mod(outDimU);
      const acc = float(0).toVar();
      acc.assign(b.element(o));

      const xRow = batch.mul(inStrideU);
      // wRow is in vec4-elements (fp32) or u32-pairs (fp16). Both cases:
      // one row covers `inDimV4` "slots" conceptually; actual storage index
      // is `o * inDimV4` for fp32 or `o * inDimV4 * 2` for fp16.
      const wRowF32 = o.mul(inDimV4);
      const wRowF16 = o.mul(inDimV4 * 2);

      Loop(inDimV4, ({ i }) => {
        const xi = xRow.add(i.mul(4));
        const xv = vec4(
          x.element(xi),
          x.element(xi.add(1)),
          x.element(xi.add(2)),
          x.element(xi.add(3)),
        );
        if (precision === "fp32") {
          // Must materialize wRowF32 + i as a var so the inner expression doesn't
          // rely on `i` still being in scope when TSL emits the indexing math.
          // See historical bug: nested Loop shadowing. Here no inner loop, but
          // playing safe.
          const wv = (W as StorageBufferNode<"vec4">).element(wRowF32.add(i));
          acc.addAssign(dot(xv, wv));
        } else {
          // fp16: 2 u32s per vec4, each u32 holds 2 halves.
          const u0 = (W as StorageBufferNode<"uint">).element(wRowF16.add(i.mul(2)));
          const u1 = (W as StorageBufferNode<"uint">).element(wRowF16.add(i.mul(2)).add(1));
          // unpackHalf2x16 returns vec2<f32>; build vec4 from (vec2, vec2).
          // TS generic overload for that form isn't declared, so cast through.
          const lo = unpackHalf2x16(u0) as unknown as Parameters<typeof vec4>[0];
          const hi = unpackHalf2x16(u1) as unknown as Parameters<typeof vec4>[0];
          const wv = (vec4 as unknown as (a: unknown, b: unknown) => ReturnType<typeof vec4>)(lo, hi);
          acc.addAssign(dot(xv, wv));
        }
      });

      if (activate) {
        acc.assign(select(acc.greaterThan(0), acc, exp(acc).sub(1)));
      }
      y.element(idx).assign(acc);
    })().compute(this.maxAgents * outDim, [64]);
    kernel.setName(id(label));
    return kernel;
  }

  /** Decoder FiLM+Linear. Like buildLinear but the z-input is FiLM-modulated
   *  per iteration with scale[i] = scaleW[i] * t[s] + scaleB[i] and shift
   *  similarly, evaluated in vec4 chunks for cache-friendly scalar reads. */
  private buildFilmLinear(
    label: string,
    inStride: number, outDim: number, seq: number,
    activate: boolean, tileInput: boolean,
    z: StorageBufferNode<"float">, t: StorageBufferNode<"float">,
    scaleW: StorageBufferNode<"float">, scaleB: StorageBufferNode<"float">,
    shiftW: StorageBufferNode<"float">, shiftB: StorageBufferNode<"float">,
    W: StorageBufferNode<"vec4"> | StorageBufferNode<"uint">,
    b: StorageBufferNode<"float">,
    y: StorageBufferNode<"float">,
  ): ComputeNode {
    const inStrideU = uint(inStride);
    const outDimU = uint(outDim);
    const seqOutU = uint(seq * outDim);
    const inDimV4 = vec4Count(inStride);
    const precision: WeightPrecision = this.precision;

    const kernel = Fn(() => {
      const idx = instanceIndex;
      const batch = idx.div(seqOutU);
      const rem = idx.mod(seqOutU);
      const s = rem.div(outDimU);
      const o = rem.mod(outDimU);
      const tv = t.element(s);
      // z per-agent stride is `inStride` (scalar floats), tile-same-or-per-step.
      const zBase = tileInput
        ? batch.mul(inStrideU)
        : batch.mul(uint(seq)).mul(inStrideU).add(s.mul(inStrideU));
      const acc = float(0).toVar();
      acc.assign(b.element(o));

      const wRowF32 = o.mul(inDimV4);
      const wRowF16 = o.mul(inDimV4 * 2);

      Loop(inDimV4, ({ i }) => {
        const fi = i.mul(4);
        const zi = zBase.add(fi);
        const zRaw = vec4(
          z.element(zi),
          z.element(zi.add(1)),
          z.element(zi.add(2)),
          z.element(zi.add(3)),
        );
        const sW = vec4(
          scaleW.element(fi),
          scaleW.element(fi.add(1)),
          scaleW.element(fi.add(2)),
          scaleW.element(fi.add(3)),
        );
        const sB = vec4(
          scaleB.element(fi),
          scaleB.element(fi.add(1)),
          scaleB.element(fi.add(2)),
          scaleB.element(fi.add(3)),
        );
        const hW = vec4(
          shiftW.element(fi),
          shiftW.element(fi.add(1)),
          shiftW.element(fi.add(2)),
          shiftW.element(fi.add(3)),
        );
        const hB = vec4(
          shiftB.element(fi),
          shiftB.element(fi.add(1)),
          shiftB.element(fi.add(2)),
          shiftB.element(fi.add(3)),
        );
        const sc = sW.mul(tv).add(sB);
        const sh = hW.mul(tv).add(hB);
        const zv = sc.mul(zRaw).add(sh);
        if (precision === "fp32") {
          const wv = (W as StorageBufferNode<"vec4">).element(wRowF32.add(i));
          acc.addAssign(dot(zv, wv));
        } else {
          const u0 = (W as StorageBufferNode<"uint">).element(wRowF16.add(i.mul(2)));
          const u1 = (W as StorageBufferNode<"uint">).element(wRowF16.add(i.mul(2)).add(1));
          const lo = unpackHalf2x16(u0) as unknown as Parameters<typeof vec4>[0];
          const hi = unpackHalf2x16(u1) as unknown as Parameters<typeof vec4>[0];
          const wv = (vec4 as unknown as (a: unknown, b: unknown) => ReturnType<typeof vec4>)(lo, hi);
          acc.addAssign(dot(zv, wv));
        }
      });

      if (activate) {
        acc.assign(select(acc.greaterThan(0), acc, exp(acc).sub(1)));
      }
      y.element(idx).assign(acc);
    })().compute(this.maxAgents * seq * outDim, [64]);
    kernel.setName(id(label));
    return kernel;
  }

  /** In-place per-group softmax. Groups are per-agent disjoint, so in-place
   *  is safe. Dispatch N * (latent / groupSize) threads. */
  private buildSoftmaxInPlace(
    label: string, total: number, groupSize: number, buf: StorageBufferNode<"float">,
  ): ComputeNode {
    const groupsPerAgent = total / groupSize;
    const gs = uint(groupSize);
    const kernel = Fn(() => {
      const g = instanceIndex;
      const base = g.mul(gs);
      const m = float(0).toVar();
      m.assign(buf.element(base));
      Loop({ start: 1, end: groupSize, type: "int" }, ({ i }) => {
        const v = buf.element(base.add(i));
        If(v.greaterThan(m), () => { m.assign(v); });
      });
      const sum = float(0).toVar();
      Loop(groupSize, ({ i }) => {
        const elemIdx = base.add(i);
        const e = exp(buf.element(elemIdx).sub(m));
        buf.element(elemIdx).assign(e);
        sum.addAssign(e);
      });
      const inv = float(1).div(sum);
      Loop(groupSize, ({ i }) => {
        const elemIdx = base.add(i);
        buf.element(elemIdx).assign(buf.element(elemIdx).mul(inv));
      });
    })().compute(this.maxAgents * groupsPerAgent, [64]);
    kernel.setName(id(label));
    return kernel;
  }

  // ---------------------------------------------------------------------
  /** Resize every kernel's dispatch so it only covers `activeCount` agents.
   *  Three.js reads `ComputeNode.count` each dispatch to compute workgroup
   *  count AND to emit an early-return guard in the kernel (via
   *  `builder.allowEarlyReturns`), so this single knob shrinks both the
   *  number of workgroups launched and the work each thread does. */
  private setActiveCount(activeCount: number): void {
    if (activeCount === this.lastActiveCount) return;
    this.lastActiveCount = activeCount;
    const clamped = Math.max(0, Math.min(activeCount, this.maxAgents));
    for (let i = 0; i < this.pipeline.length; i++) {
      (this.pipeline[i] as unknown as { count: number }).count =
        clamped * this.perAgentScale[i];
    }
  }

  /** Run one batched inference step for `activeCount` active agents.
   *  Returns the denormalized output slice [activeCount * SEQ * OUT] along
   *  with timing breakdown in milliseconds. Slots beyond activeCount are
   *  not touched by the GPU this pass. */
  async run(activeCount: number = this.maxAgents): Promise<{
    output: Float32Array; computeMs: number; readbackMs: number;
  }> {
    this.xRawAttr.needsUpdate = true;
    this.setActiveCount(activeCount);

    const t0 = performance.now();
    await this.renderer.computeAsync(this.pipeline);
    const t1 = performance.now();
    const ab = await this.renderer.getArrayBufferAsync(this.decOutAttr);
    const t2 = performance.now();

    return {
      output: new Float32Array(ab),
      computeMs: t1 - t0,
      readbackMs: t2 - t1,
    };
  }
}
