/** Predicted sequence — 16-frame future of root + motion + contacts + guidance.
 *  Sampled every frame with sub-frame interpolation to drive the actor.
 */
import * as M from "../math/mat4.js";
import * as Q from "../math/quat.js";
import * as V from "../math/vec3.js";
import type { Mat4 } from "../math/mat4.js";
import type { Quat } from "../math/quat.js";
import type { Vec3 } from "../math/vec3.js";

export class Sequence {
  /** Wall-clock time when this sequence was produced by inference.
   *  `(totalTime - creationTime) * timescale` is the virtual sample point. */
  creationTime = 0;
  timestamps: Float32Array = new Float32Array(0);
  rootTransforms: Mat4[] = [];           // [SEQ]
  rootVelocities: Vec3[] = [];
  bonePositions: Vec3[][] = [];          // [SEQ][bones]
  boneQuaternions: Quat[][] = [];
  boneVelocities: Vec3[][] = [];
  contacts: Float32Array[] = [];         // [SEQ][4]
  guidances: Vec3[][] = [];              // [SEQ][bones]

  private getIndexPair(t: number): { a: number; b: number; w: number } {
    const n = this.timestamps.length;
    const first = this.timestamps[0];
    const last = this.timestamps[n - 1];
    let ratio = ((t - first) / (last - first)) * (n - 1);
    if (ratio < 0) ratio = 0;
    if (ratio > n - 1) ratio = n - 1;
    const a = Math.floor(ratio);
    const b = Math.min(a + 1, n - 1);
    const w = a === b ? 0 : (t - this.timestamps[a]) / (this.timestamps[b] - this.timestamps[a]);
    return { a, b, w: Math.max(0, Math.min(1, w)) };
  }

  sampleRoot(t: number, out: Mat4): Mat4 {
    const { a, b, w } = this.getIndexPair(t);
    const tmp = M.interpolate(this.rootTransforms[a], this.rootTransforms[b], w);
    M.copy(out, tmp);
    return out;
  }

  // NMMAgent is the only consumer; it always samples into pre-allocated
  // scratch to keep the 60Hz animate loop allocation-free. Non-allocating
  // variants are the only Sequence API.
  sampleBonePositionsInto(t: number, out: Vec3[]): void {
    const { a, b, w } = this.getIndexPair(t);
    const arrA = this.bonePositions[a];
    const arrB = this.bonePositions[b];
    for (let i = 0; i < out.length; i++) V.lerpInto(out[i], arrA[i], arrB[i], w);
  }
  sampleBoneQuaternionsInto(t: number, out: Quat[]): void {
    const { a, b, w } = this.getIndexPair(t);
    const arrA = this.boneQuaternions[a];
    const arrB = this.boneQuaternions[b];
    for (let i = 0; i < out.length; i++) Q.slerpInto(out[i], arrA[i], arrB[i], w);
  }
  sampleBoneVelocitiesInto(t: number, out: Vec3[]): void {
    const { a, b, w } = this.getIndexPair(t);
    const arrA = this.boneVelocities[a];
    const arrB = this.boneVelocities[b];
    for (let i = 0; i < out.length; i++) V.lerpInto(out[i], arrA[i], arrB[i], w);
  }
  sampleContactsInto(t: number, out: Float32Array): void {
    const { a, b, w } = this.getIndexPair(t);
    const arrA = this.contacts[a];
    const arrB = this.contacts[b];
    for (let i = 0; i < out.length; i++) out[i] = arrA[i] + (arrB[i] - arrA[i]) * w;
  }

  /** Root-lock when both feet are grounded (avg contact > 0.75). */
  getRootLock(): number {
    let sum = 0, n = 0;
    for (const c of this.contacts) { for (let i = 0; i < c.length; i++) { sum += c[i]; n++; } }
    return n > 0 && sum / n > 0.75 ? 1.0 : 0.0;
  }

  /** Arc length of the predicted root trajectory. */
  getLength(): number {
    let len = 0;
    for (let i = 1; i < this.rootTransforms.length; i++) {
      len += V.distance(
        M.getPosition(this.rootTransforms[i - 1]),
        M.getPosition(this.rootTransforms[i]));
    }
    return len;
  }
}
