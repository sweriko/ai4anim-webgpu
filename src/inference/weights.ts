/** Weight packing + padding utilities for batched matmul kernels.
 *
 *  The Linear / FiLM-Linear kernels iterate `ceil(inDim / 4)` times and read
 *  one vec4 of weights + a vec4 built from 4 consecutive x scalars per
 *  iteration. For that to stay correct:
 *
 *    1. Weight rows must be padded to a multiple of 4 with trailing zeros
 *       — zeros in the tail contribute nothing to the dot product, but the
 *       inner dim becomes vec4-aligned.
 *    2. Per-agent input / scratch buffers must use the padded stride so
 *       scalar reads within an iteration never cross into another agent.
 *
 *  Two precision variants are packed by the same entry points:
 *
 *    - fp32:  Float32Array, stored as `array<vec4<f32>>`.
 *    - fp16:  Uint32Array with two halves packed per u32, stored as
 *             `array<u32>`. The kernel uses `unpackHalf2x16` to recover
 *             two `vec2<f32>`s per u32, concatenates to a `vec4<f32>`, and
 *             dots against an f32 x-vector — a mixed-precision pattern
 *             (fp16 inputs, fp32 accumulator) that avoids catastrophic
 *             precision loss on long accumulations.
 */

import { DataUtils } from "three";

export type WeightPrecision = "fp32" | "fp16";

/** Floats per vec4 lane (for indexing). */
export const V4 = 4;

/** Round `n` up to the next multiple of `V4`. */
export const pad4 = (n: number): number => Math.ceil(n / V4) * V4;

/** Number of vec4 slots needed to hold `n` scalars, padded up. */
export const vec4Count = (n: number): number => Math.ceil(n / V4);

export interface PackedWeight {
  /** Precision of the packed data. */
  precision: WeightPrecision;
  /** Typed array that backs the StorageBufferAttribute.
   *  fp32: Float32Array of length rows * paddedInDim.
   *  fp16: Uint32Array of length rows * (paddedInDim / 2). */
  data: Float32Array | Uint32Array;
  /** Row stride in element units (vec4 for fp32, u32 for fp16). */
  rowStride: number;
  /** Padded inner dimension (always multiple of 4). */
  paddedInDim: number;
  /** ceil(paddedInDim / 4). The matmul kernel's inner-loop count. */
  inDimV4: number;
  /** For fp16, how many u32s hold one row (= paddedInDim / 2). */
  u32PerRow: number;
}

/** Pack a row-major float32 matrix of shape `[rows, inDim]` into a vec4-
 *  aligned buffer at the requested precision. */
export function packWeight(
  src: Float32Array, rows: number, inDim: number, precision: WeightPrecision,
): PackedWeight {
  if (src.length !== rows * inDim) {
    throw new Error(`packWeight shape mismatch: array ${src.length} != ${rows}*${inDim}`);
  }
  const padded = pad4(inDim);
  const inDimV4 = padded / V4;

  if (precision === "fp32") {
    const data = new Float32Array(rows * padded);
    for (let r = 0; r < rows; r++) {
      // Copy the real inDim floats; remaining 0..3 slots stay zero.
      data.set(src.subarray(r * inDim, (r + 1) * inDim), r * padded);
    }
    return {
      precision: "fp32", data, rowStride: padded, paddedInDim: padded,
      inDimV4, u32PerRow: 0,
    };
  }

  // fp16: two halves per u32, little-endian within the u32 (halves[i*2] in
  // low bits, halves[i*2+1] in high bits — matches `unpack2x16float`).
  const u32PerRow = padded / 2;   // padded is multiple of 4 ⇒ multiple of 2
  const data = new Uint32Array(rows * u32PerRow);
  for (let r = 0; r < rows; r++) {
    const rowBase = r * u32PerRow;
    const srcBase = r * inDim;
    for (let i = 0; i < padded; i += 2) {
      const aVal = i < inDim ? src[srcBase + i] : 0;
      const bVal = (i + 1) < inDim ? src[srcBase + i + 1] : 0;
      const aHalf = DataUtils.toHalfFloat(aVal);
      const bHalf = DataUtils.toHalfFloat(bVal);
      data[rowBase + i / 2] = (aHalf & 0xffff) | ((bHalf & 0xffff) << 16);
    }
  }
  return {
    precision: "fp16", data, rowStride: u32PerRow, paddedInDim: padded,
    inDimV4, u32PerRow,
  };
}

/** Pack a flat vector of `n` fp32 values into a padded Float32Array (size
 *  `pad4(n)`) — used for FiLM coefficient buffers that need vec4 alignment
 *  but aren't worth converting to fp16. */
export function padScalarF32(src: Float32Array, n: number): Float32Array {
  const padded = pad4(n);
  if (padded === n) return src;   // already aligned; reuse
  const out = new Float32Array(padded);
  out.set(src);
  return out;
}

