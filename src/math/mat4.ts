/** 4×4 homogeneous transform, stored as Float32Array(16), column-major.
 *
 * Index m[col*4 + row]. Three.js Matrix4.elements uses the same layout, so we
 * can copy directly. Axes live in columns 0/1/2 (X/Y/Z), translation in col 3.
 *
 * This mirrors the Python AI4Animation convention where `matrix[..., :3, 2]`
 * extracts the Z-axis (column 2 of the 3×3 upper-left).
 */
import * as V from "./vec3.js";
import type { Vec3 } from "./vec3.js";

export type Mat4 = Float32Array;

export const create = (): Mat4 => {
  const m = new Float32Array(16);
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
  return m;
};

export const copy = (out: Mat4, a: Mat4): Mat4 => {
  out.set(a);
  return out;
};

export const clone = (a: Mat4): Mat4 => {
  const m = new Float32Array(16);
  m.set(a);
  return m;
};

export const identity = (out: Mat4): Mat4 => {
  out.fill(0);
  out[0] = 1; out[5] = 1; out[10] = 1; out[15] = 1;
  return out;
};

/** Build a TR matrix from translation vector t and 3×3 rotation r (column-major 9-entry). */
export const fromTR = (out: Mat4, t: Vec3, r: ArrayLike<number>): Mat4 => {
  out[0] = r[0]; out[1] = r[1]; out[2] = r[2]; out[3] = 0;
  out[4] = r[3]; out[5] = r[4]; out[6] = r[5]; out[7] = 0;
  out[8] = r[6]; out[9] = r[7]; out[10] = r[8]; out[11] = 0;
  out[12] = t[0]; out[13] = t[1]; out[14] = t[2]; out[15] = 1;
  return out;
};

export const getPosition = (m: Mat4): Vec3 => [m[12], m[13], m[14]];
export const getAxisX = (m: Mat4): Vec3 => [m[0], m[1], m[2]];
export const getAxisY = (m: Mat4): Vec3 => [m[4], m[5], m[6]];
export const getAxisZ = (m: Mat4): Vec3 => [m[8], m[9], m[10]];

export const setPosition = (m: Mat4, t: Vec3): void => {
  m[12] = t[0]; m[13] = t[1]; m[14] = t[2];
};

export const setRotation3x3 = (m: Mat4, r: ArrayLike<number>): void => {
  m[0] = r[0]; m[1] = r[1]; m[2] = r[2];
  m[4] = r[3]; m[5] = r[4]; m[6] = r[5];
  m[8] = r[6]; m[9] = r[7]; m[10] = r[8];
};

/** out = a * b (column-major multiply). */
export const multiply = (out: Mat4, a: Mat4, b: Mat4): Mat4 => {
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

  let b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  out[0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
  out[1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
  out[2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
  out[3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

  b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
  out[4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
  out[5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
  out[6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
  out[7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

  b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
  out[8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
  out[9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
  out[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
  out[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

  b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
  out[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
  out[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
  out[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
  out[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

  return out;
};

/** Inverse of an affine TR matrix (rotation+translation, no scale). */
export const invertAffine = (out: Mat4, m: Mat4): Mat4 => {
  // R^T and -R^T * t
  out[0] = m[0]; out[1] = m[4]; out[2] = m[8]; out[3] = 0;
  out[4] = m[1]; out[5] = m[5]; out[6] = m[9]; out[7] = 0;
  out[8] = m[2]; out[9] = m[6]; out[10] = m[10]; out[11] = 0;
  const tx = m[12], ty = m[13], tz = m[14];
  out[12] = -(out[0] * tx + out[4] * ty + out[8] * tz);
  out[13] = -(out[1] * tx + out[5] * ty + out[9] * tz);
  out[14] = -(out[2] * tx + out[6] * ty + out[10] * tz);
  out[15] = 1;
  return out;
};

export const transformPoint = (m: Mat4, v: Vec3): Vec3 => [
  m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12],
  m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13],
  m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14],
];

export const transformDirection = (m: Mat4, v: Vec3): Vec3 => [
  m[0] * v[0] + m[4] * v[1] + m[8] * v[2],
  m[1] * v[0] + m[5] * v[1] + m[9] * v[2],
  m[2] * v[0] + m[6] * v[1] + m[10] * v[2],
];

/** Inverse-transform a point by an affine matrix (world→local). */
export const inverseTransformPoint = (m: Mat4, v: Vec3): Vec3 => {
  const dx = v[0] - m[12], dy = v[1] - m[13], dz = v[2] - m[14];
  return [
    m[0] * dx + m[1] * dy + m[2] * dz,
    m[4] * dx + m[5] * dy + m[6] * dz,
    m[8] * dx + m[9] * dy + m[10] * dz,
  ];
};

export const inverseTransformDirection = (m: Mat4, v: Vec3): Vec3 => [
  m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
  m[4] * v[0] + m[5] * v[1] + m[6] * v[2],
  m[8] * v[0] + m[9] * v[1] + m[10] * v[2],
];

/** Linear-interpolate column-major 4×4 matrices and re-normalize rotation. */
export const interpolate = (a: Mat4, b: Mat4, t: number): Mat4 => {
  const out = new Float32Array(16);
  for (let i = 0; i < 16; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  reNormalizeRotation(out);
  return out;
};

/** In-place: out = lerp(a, b, t), then re-orthonormalize the rotation. */
export const interpolateInto = (out: Mat4, a: Mat4, b: Mat4, t: number): void => {
  for (let i = 0; i < 16; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  reNormalizeRotation(out);
};

/** In-place: the result of interpolate(a, b, t) written into `a` — useful when
 *  you want to overwrite one of the inputs without a temporary. */
export const interpolateIntoFirst = (a: Mat4, a0: Mat4, b: Mat4, t: number): void => {
  for (let i = 0; i < 16; i++) a[i] = a0[i] + (b[i] - a0[i]) * t;
  reNormalizeRotation(a);
};

/** Re-orthonormalize the 3×3 upper-left via Rotation.Look-equivalent
 *  (take Y and Z axes from the current rotation, rebuild X as cross(Y,Z)). */
export const reNormalizeRotation = (m: Mat4): void => {
  const y = V.normalize([m[4], m[5], m[6]]);
  let z = V.normalize([m[8], m[9], m[10]]);
  let x = V.normalize(V.cross(y, z));
  // Recompute orthogonalized z so the basis is exactly orthonormal.
  z = V.normalize(V.cross(x, y));
  m[0] = x[0]; m[1] = x[1]; m[2] = x[2];
  m[4] = y[0]; m[5] = y[1]; m[6] = y[2];
  m[8] = z[0]; m[9] = z[1]; m[10] = z[2];
};
