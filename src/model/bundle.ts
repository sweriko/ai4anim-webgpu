/** Loads an exported `<kind>.bin` + `<kind>.json` bundle (the pre-baked
 *  bundles in `public/` were exported from upstream AI4AnimationPy
 *  trainings; see the project's NOTICE).
 *
 *  Per-entry storage is mixed precision: f16 for the 9 large linear weight
 *  matrices, f32 for everything else (biases, FiLM coefs, stats, timing,
 *  guidances). `get()` returns Float32Array regardless — f16 entries are
 *  lazily up-converted at call time. f32 entries remain zero-copy views.
 *  Do not mutate the returned arrays. */

import { DataUtils } from "three";

export type EntryDtype = "f16" | "f32";

export interface BundleEntry {
  offset: number;
  count: number;
  shape: number[];
  /** On-disk precision. Defaults to f32 when absent (older bundles). */
  dtype?: EntryDtype;
}

export interface GuidanceMeta {
  entry: string;
}

export type ModelKind = "biped" | "quadruped";

/** Per-bone input feature blocks in feed-order. Subset of {pos, axisZ, axisY, vel}. */
export type InputFeatureTag = "pos" | "axisZ" | "axisY" | "vel";

export interface BipedSkeletonMeta {
  bone_names: string[];
  bone_count: number;
  ground: {
    left_ankle: number; left_ball: number;
    right_ankle: number; right_ball: number;
  };
  legs: {
    left_hip: number; left_knee: number; left_ankle: number; left_ball: number;
    right_hip: number; right_knee: number; right_ankle: number; right_ball: number;
  };
  contact_labels: [string, string, string, string];
}

export interface QuadrupedIKChain {
  label: string;
  source: number;
  target: number;
  /** Index into the 4-contact vector that gates this chain. */
  contact_index: number;
}

export interface QuadrupedSkeletonMeta {
  bone_names: string[];
  /** One parent name per bone (null for root). The Dog GLB lacks some leaf
   *  bones (e.g. HeadSite) so the runtime synthesizes them from this list. */
  parent_names: (string | null)[];
  bone_count: number;
  ik_chains: QuadrupedIKChain[];
  contact_labels: [string, string, string, string];
}

export interface BipedControlMeta {
  sequence_length: number;
  sequence_window: number;
  prediction_fps: number;
  sequence_fps: number;
  min_timescale: number;
  max_timescale: number;
  synchronization_sensitivity: number;
  timescale_sensitivity: number;
  contact_power: number;
  contact_threshold: number;
  trajectory_correction: number;
}

export interface QuadrupedControlMeta extends BipedControlMeta {
  locomotion_modes: { walk: number; pace: number; trot: number; canter: number };
  pid: { kp: number; ki: number; kd: number };
  input_deadzone: number;
  action_trigger_speed_max: number;
}

export interface ModelMeta {
  kind: ModelKind;
  input_dim: number;
  output_dim: number;
  latent_dim: number;
  codebook_channels: number;
  codebook_dims: number;
  sequence_length: number;
  sequence_window: number;
  denoiser_iterations: number;
  input_per_bone: InputFeatureTag[];
}

export interface BundleMeta {
  model: ModelMeta;
  skeleton: BipedSkeletonMeta | QuadrupedSkeletonMeta;
  control: BipedControlMeta | QuadrupedControlMeta;
  guidances: string[];
  bin: Record<string, BundleEntry>;
}

/** Distinguisher for narrowing inside callers. */
export function isQuadruped(meta: BundleMeta): meta is BundleMeta & {
  skeleton: QuadrupedSkeletonMeta;
  control: QuadrupedControlMeta;
} {
  return meta.model.kind === "quadruped";
}

export function isBiped(meta: BundleMeta): meta is BundleMeta & {
  skeleton: BipedSkeletonMeta;
  control: BipedControlMeta;
} {
  return meta.model.kind === "biped";
}

export class Bundle {
  private constructor(
    public readonly meta: BundleMeta,
    public readonly buf: ArrayBuffer,
  ) {}

  /** Load the bundle named `<kind>.json` + `<kind>.bin` from `base`. */
  static async load(base = "/", kind: ModelKind = "biped"): Promise<Bundle> {
    const metaUrl = `${base}${kind}.json`;
    const binUrl = `${base}${kind}.bin`;
    const [meta, bin] = await Promise.all([
      fetch(metaUrl).then((r) => {
        if (!r.ok) throw new Error(`failed to load ${metaUrl} (${r.status})`);
        return r.json() as Promise<BundleMeta>;
      }),
      fetch(binUrl).then((r) => {
        if (!r.ok) throw new Error(`failed to load ${binUrl} (${r.status})`);
        return r.arrayBuffer();
      }),
    ]);
    // Back-compat: older bundles may be missing the new schema fields.
    if (!meta.model.kind) {
      (meta.model as ModelMeta).kind = kind;
    }
    if (!meta.model.input_per_bone) {
      meta.model.input_per_bone = kind === "biped"
        ? ["pos", "axisZ", "axisY", "vel"]
        : ["pos", "vel"];
    }
    return new Bundle(meta, bin);
  }

  /** Float32Array for a named tensor. f32 entries are zero-copy views;
   *  f16 entries are decoded into a fresh Float32Array on each call. */
  get(key: string): Float32Array {
    const e = this.meta.bin[key];
    if (!e) throw new Error(`bundle entry not found: ${key}`);
    const dtype: EntryDtype = e.dtype ?? "f32";
    if (dtype === "f32") {
      return new Float32Array(this.buf, e.offset, e.count);
    }
    // f16: read packed halves, expand to fp32 via three's lookup table.
    const halves = new Uint16Array(this.buf, e.offset, e.count);
    const out = new Float32Array(e.count);
    for (let i = 0; i < e.count; i++) {
      out[i] = DataUtils.fromHalfFloat(halves[i]);
    }
    return out;
  }

  shape(key: string): number[] {
    const e = this.meta.bin[key];
    if (!e) throw new Error(`bundle entry not found: ${key}`);
    return e.shape;
  }

  /** Guidance positions for a style, shape [bone_count, 3] flattened. */
  guidance(name: string): Float32Array {
    return this.get(`guidance.${name}`);
  }
}
