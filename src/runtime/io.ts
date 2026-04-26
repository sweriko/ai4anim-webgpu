/** FeedTensor / ReadTensor equivalents for the neural motion matching models.
 *
 *  Both biped (Geno, 23 bones) and quadruped (Dog, 27 bones) share the same
 *  output layout but differ on input features:
 *    biped     — per-bone: pos + axisZ + axisY + vel   (12 floats)
 *    quadruped — per-bone: pos + vel                    (6 floats)
 *
 *  Bone count, per-bone feature order, and total dims are all taken from the
 *  bundle so neither layout is hardcoded here — the two exporters stamp the
 *  schema into `model.input_per_bone` and `skeleton.bone_count`.
 *
 *  Input (flat, per-field):
 *    per-bone features (N_bones × k, root-local)
 *    future root (16 × 6, root-local): pos.xz + axisZ.xz + vel.xz
 *    guidance (N_bones × 3, root-local): pos
 *  Output per frame (16 frames packed contiguously):
 *    root delta (3): dx, dyaw, dz
 *    bone TR   (N_bones × 9): pos(3) + axisZ(3) + axisY(3)
 *    bone vel  (N_bones × 3)
 *    contacts  (4)
 *    guidance  (N_bones × 3)
 */
import * as Mat from "../math/mat4.js";
import type { Mat4 } from "../math/mat4.js";
import type { Vec3 } from "../math/vec3.js";
import type { Actor } from "./actor.js";
import type { RootSeries } from "./root_series.js";
import type { BundleMeta, InputFeatureTag } from "../model/bundle.js";

export interface PredictedFrame {
  rootTransform: Mat4;
  rootVelocity: Vec3;
  bonePositions: Vec3[];
  boneAxesZ: Vec3[];
  boneAxesY: Vec3[];
  boneVelocities: Vec3[];
  contacts: Float32Array;
  guidances: Vec3[];
}

export interface PredictionResult {
  frames: PredictedFrame[];
}

const SEQUENCE_FPS = 30;

/** Per-bundle IO helper. Cached shapes, feature schema, and buffer size so the
 *  hot path avoids re-reading bundle meta each tick. */
export class IO {
  readonly seq: number;
  readonly bones: number;
  readonly outDim: number;
  readonly inDim: number;
  private readonly perBone: InputFeatureTag[];
  private readonly perBoneFloats: number;

  constructor(meta: BundleMeta) {
    this.seq = meta.model.sequence_length;
    this.bones = meta.skeleton.bone_count;
    this.outDim = meta.model.output_dim;
    this.inDim = meta.model.input_dim;
    this.perBone = meta.model.input_per_bone;
    this.perBoneFloats = this.perBone.length * 3;
    // Sanity: recomputed input_dim must match the bundle.
    const expected = this.bones * this.perBoneFloats + this.seq * 6 + this.bones * 3;
    if (expected !== this.inDim) {
      throw new Error(
        `IO: input_dim mismatch: bundle says ${this.inDim} but computed ${expected} `
        + `from bone_count=${this.bones}, per_bone=[${this.perBone.join(",")}], seq=${this.seq}`);
    }
  }

  /** Build the flat input vector for this model. */
  feed(actor: Actor, rootControl: RootSeries, guidancePositions: Vec3[]): Float32Array {
    const buf = new Float32Array(this.inDim);
    let p = 0;

    const root = actor.root;

    // Cache per-bone root-local features we might emit.
    const bonePos: Vec3[] = [];
    const boneAxZ: Vec3[] = [];
    const boneAxY: Vec3[] = [];
    const boneVel: Vec3[] = [];
    const needAxZ = this.perBone.includes("axisZ");
    const needAxY = this.perBone.includes("axisY");
    for (let i = 0; i < this.bones; i++) {
      const boneLocal = transformationTo(root, actor.transforms[i]);
      bonePos.push(Mat.getPosition(boneLocal));
      if (needAxZ) boneAxZ.push(Mat.getAxisZ(boneLocal));
      if (needAxY) boneAxY.push(Mat.getAxisY(boneLocal));
      boneVel.push(Mat.inverseTransformDirection(root, actor.velocities[i]));
    }

    // Per-bone feature blocks — one field (all bones) at a time, in the order
    // declared by `model.input_per_bone`. Not interleaved per-bone.
    for (const tag of this.perBone) {
      const src = tag === "pos" ? bonePos
                : tag === "axisZ" ? boneAxZ
                : tag === "axisY" ? boneAxY
                : boneVel;
      for (let i = 0; i < this.bones; i++) {
        buf[p++] = src[i][0]; buf[p++] = src[i][1]; buf[p++] = src[i][2];
      }
    }

    // Future root (16 × 6): pos.xz, axisZ.xz, vel.xz
    const rootPosXZ: [number, number][] = [];
    const rootAxZXZ: [number, number][] = [];
    const rootVelXZ: [number, number][] = [];
    for (let s = 0; s < this.seq; s++) {
      const rootSampleLocal = transformationTo(root, rootControl.transforms[s]);
      const pos = Mat.getPosition(rootSampleLocal);
      const axZ = Mat.getAxisZ(rootSampleLocal);
      const velLocal = Mat.inverseTransformDirection(root, rootControl.velocities[s]);
      rootPosXZ.push([pos[0], pos[2]]);
      rootAxZXZ.push([axZ[0], axZ[2]]);
      rootVelXZ.push([velLocal[0], velLocal[2]]);
    }
    for (let s = 0; s < this.seq; s++) { buf[p++] = rootPosXZ[s][0]; buf[p++] = rootPosXZ[s][1]; }
    for (let s = 0; s < this.seq; s++) { buf[p++] = rootAxZXZ[s][0]; buf[p++] = rootAxZXZ[s][1]; }
    for (let s = 0; s < this.seq; s++) { buf[p++] = rootVelXZ[s][0]; buf[p++] = rootVelXZ[s][1]; }

    // Guidance (N × 3)
    for (let i = 0; i < this.bones; i++) {
      const g = guidancePositions[i];
      buf[p++] = g[0]; buf[p++] = g[1]; buf[p++] = g[2];
    }

    if (p !== this.inDim) throw new Error(`feed: filled ${p}, expected ${this.inDim}`);
    return buf;
  }

  /** Decode the 16×outDim output tensor into world-space PredictedFrames. */
  read(output: Float32Array, actorRoot: Mat4): PredictionResult {
    const SEQ = this.seq;
    const BONES = this.bones;
    const OUT = this.outDim;
    if (output.length !== SEQ * OUT) {
      throw new Error(`output length ${output.length} != ${SEQ * OUT}`);
    }

    const rootVecs: [number, number, number][] = [];
    const bonePosLocal: Vec3[][] = [];
    const boneAxZLocal: Vec3[][] = [];
    const boneAxYLocal: Vec3[][] = [];
    const boneVelLocal: Vec3[][] = [];
    const contacts: Float32Array[] = [];
    const guidances: Vec3[][] = [];

    for (let f = 0; f < SEQ; f++) {
      let p = f * OUT;
      rootVecs.push([output[p++], output[p++], output[p++]]);

      const pos: Vec3[] = [];
      for (let i = 0; i < BONES; i++) pos.push([output[p++], output[p++], output[p++]]);
      bonePosLocal.push(pos);

      const axZ: Vec3[] = [];
      for (let i = 0; i < BONES; i++) axZ.push([output[p++], output[p++], output[p++]]);
      const axY: Vec3[] = [];
      for (let i = 0; i < BONES; i++) axY.push([output[p++], output[p++], output[p++]]);
      boneAxZLocal.push(axZ);
      boneAxYLocal.push(axY);

      const vel: Vec3[] = [];
      for (let i = 0; i < BONES; i++) vel.push([output[p++], output[p++], output[p++]]);
      boneVelLocal.push(vel);

      const c = new Float32Array(4);
      for (let i = 0; i < 4; i++) c[i] = output[p++];
      contacts.push(c);

      const g: Vec3[] = [];
      for (let i = 0; i < BONES; i++) g.push([output[p++], output[p++], output[p++]]);
      guidances.push(g);
    }

    // Integrate root deltas and lift into world space.
    const cum = { dx: 0, dyaw: 0, dz: 0 };
    const rootTransformsWorld: Mat4[] = [];
    const rootVelocitiesWorld: Vec3[] = [];
    for (let f = 0; f < SEQ; f++) {
      if (f > 0) {
        cum.dx += rootVecs[f][0];
        cum.dyaw += rootVecs[f][1];
        cum.dz += rootVecs[f][2];
      }
      const deltaLocal = deltaXZ(cum.dx, cum.dyaw, cum.dz);
      const frameRoot = transformationFrom(actorRoot, deltaLocal);
      rootTransformsWorld.push(frameRoot);

      const velLocal: Vec3 = [
        rootVecs[f][0] * SEQUENCE_FPS, 0, rootVecs[f][2] * SEQUENCE_FPS,
      ];
      rootVelocitiesWorld.push(Mat.transformDirection(frameRoot, velLocal));
    }

    const frames: PredictedFrame[] = [];
    for (let f = 0; f < SEQ; f++) {
      const frameRoot = rootTransformsWorld[f];
      const bonePositions: Vec3[] = [];
      const boneAxesZ: Vec3[] = [];
      const boneAxesY: Vec3[] = [];
      const boneVelocities: Vec3[] = [];
      for (let i = 0; i < BONES; i++) {
        bonePositions.push(Mat.transformPoint(frameRoot, bonePosLocal[f][i]));
        boneAxesZ.push(Mat.transformDirection(frameRoot, boneAxZLocal[f][i]));
        boneAxesY.push(Mat.transformDirection(frameRoot, boneAxYLocal[f][i]));
        boneVelocities.push(Mat.transformDirection(frameRoot, boneVelLocal[f][i]));
      }
      frames.push({
        rootTransform: frameRoot,
        rootVelocity: rootVelocitiesWorld[f],
        bonePositions, boneAxesZ, boneAxesY, boneVelocities,
        contacts: contacts[f],
        guidances: guidances[f],
      });
    }

    return { frames };
  }
}

// ------------------------------------------------------------------
//  Helpers
// ------------------------------------------------------------------

function transformationTo(space: Mat4, x: Mat4): Mat4 {
  const inv = Mat.create();
  Mat.invertAffine(inv, space);
  const out = Mat.create();
  Mat.multiply(out, inv, x);
  return out;
}

function transformationFrom(space: Mat4, x: Mat4): Mat4 {
  const out = Mat.create();
  Mat.multiply(out, space, x);
  return out;
}

/** Build a yaw-only 4×4 from (dx, dyawDeg, dz). `dyaw` is DEGREES — matches
 *  Python's Vector3.SignedAngle output that the model was trained on. */
function deltaXZ(dx: number, dyawDeg: number, dz: number): Mat4 {
  const m = Mat.create();
  const rad = dyawDeg * Math.PI / 180;
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  Mat.identity(m);
  m[0] = c; m[2] = -s;
  m[8] = s; m[10] = c;
  m[12] = dx; m[13] = 0; m[14] = dz;
  return m;
}
