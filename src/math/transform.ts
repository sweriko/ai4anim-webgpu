/** Transform helpers mirroring `ai4animation.Math.Transform`. */
import * as M from "./mat4.js";
import * as R from "./rotation.js";
import type { Mat4 } from "./mat4.js";
import type { Vec3 } from "./vec3.js";

/** Build a TR matrix from translation + forward-and-up axes (Rotation.Look). */
export const tr = (out: Mat4, pos: Vec3, z: Vec3, y: Vec3): Mat4 => {
  M.fromTR(out, pos, R.look(z, y));
  return out;
};

/** From a position and an already-built 3×3 rotation. */
export const trFromRot = (out: Mat4, pos: Vec3, rot: R.Rot3x3): Mat4 => {
  M.fromTR(out, pos, rot);
  return out;
};

/** Transform `x` into the local space of `space`. */
export const transformationTo = (out: Mat4, x: Mat4, space: Mat4): Mat4 => {
  const inv = new Float32Array(16);
  M.invertAffine(inv as Mat4, space);
  M.multiply(out, inv as Mat4, x);
  return out;
};

/** Transform `x` from the local space of `space` into the parent's space. */
export const transformationFrom = (out: Mat4, x: Mat4, space: Mat4): Mat4 => {
  M.multiply(out, space, x);
  return out;
};

/** Root-delta helper: `delta = [dx, dyaw, dz]` → 4×4 with Y-rotation only. */
export const deltaXZ = (out: Mat4, delta: Vec3): Mat4 => {
  const c = Math.cos(delta[1]);
  const s = Math.sin(delta[1]);
  M.identity(out);
  out[0] = c; out[2] = -s;
  out[8] = s; out[10] = c;
  out[12] = delta[0]; out[13] = 0; out[14] = delta[2];
  return out;
};

/** Linear-interpolate and re-normalize rotation (Transform.Interpolate). */
export const interpolate = (a: Mat4, b: Mat4, t: number): Mat4 =>
  M.interpolate(a, b, t);
