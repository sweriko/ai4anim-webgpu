/** Quaternion — stored as [x, y, z, w]. Matches Python convention (Quaternion.py). */
import * as V from "./vec3.js";
import type { Vec3 } from "./vec3.js";

export type Quat = [number, number, number, number];

export const identity = (): Quat => [0, 0, 0, 1];

export const fromAxisAngle = (axis: Vec3, angle: number): Quat => {
  const h = angle * 0.5;
  const s = Math.sin(h);
  const n = V.normalize(axis);
  return [n[0] * s, n[1] * s, n[2] * s, Math.cos(h)];
};

export const multiply = (a: Quat, b: Quat): Quat => [
  a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
  a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
  a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
  a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
];

export const conjugate = (q: Quat): Quat => [-q[0], -q[1], -q[2], q[3]];

export const rotateVec3 = (q: Quat, v: Vec3): Vec3 => {
  // v' = q * v * q^-1. For a unit quaternion q = (xyz, w), use the optimized form.
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const vx = v[0], vy = v[1], vz = v[2];
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + y * tz - z * ty,
    vy + w * ty + z * tx - x * tz,
    vz + w * tz + x * ty - y * tx,
  ];
};

export const normalize = (q: Quat): Quat => {
  const l = Math.hypot(q[0], q[1], q[2], q[3]);
  return l > 1e-8 ? [q[0] / l, q[1] / l, q[2] / l, q[3] / l] : [0, 0, 0, 1];
};

/** In-place shortest-path slerp — writes into `out`, zero allocation. */
export const slerpInto = (out: Quat, a: Quat, b: Quat, t: number): void => {
  let bx = b[0], by = b[1], bz = b[2], bw = b[3];
  let d = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw;
  if (d < 0) { bx = -bx; by = -by; bz = -bz; bw = -bw; d = -d; }
  if (d > 0.9995) {
    const x = a[0] + (bx - a[0]) * t;
    const y = a[1] + (by - a[1]) * t;
    const z = a[2] + (bz - a[2]) * t;
    const w = a[3] + (bw - a[3]) * t;
    const l = Math.hypot(x, y, z, w);
    if (l > 1e-8) {
      out[0] = x / l; out[1] = y / l; out[2] = z / l; out[3] = w / l;
    } else {
      out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 1;
    }
    return;
  }
  const theta0 = Math.acos(d);
  const theta = theta0 * t;
  const sinTheta = Math.sin(theta);
  const sinTheta0 = Math.sin(theta0);
  const s0 = Math.cos(theta) - d * sinTheta / sinTheta0;
  const s1 = sinTheta / sinTheta0;
  out[0] = s0 * a[0] + s1 * bx;
  out[1] = s0 * a[1] + s1 * by;
  out[2] = s0 * a[2] + s1 * bz;
  out[3] = s0 * a[3] + s1 * bw;
};

/** Shortest-path slerp. */
export const slerp = (a: Quat, b: Quat, t: number): Quat => {
  let bx = b[0], by = b[1], bz = b[2], bw = b[3];
  let d = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw;
  if (d < 0) { bx = -bx; by = -by; bz = -bz; bw = -bw; d = -d; }
  if (d > 0.9995) {
    return normalize([
      a[0] + (bx - a[0]) * t,
      a[1] + (by - a[1]) * t,
      a[2] + (bz - a[2]) * t,
      a[3] + (bw - a[3]) * t,
    ]);
  }
  const theta0 = Math.acos(d);
  const theta = theta0 * t;
  const sinTheta = Math.sin(theta);
  const sinTheta0 = Math.sin(theta0);
  const s0 = Math.cos(theta) - d * sinTheta / sinTheta0;
  const s1 = sinTheta / sinTheta0;
  return [
    s0 * a[0] + s1 * bx,
    s0 * a[1] + s1 * by,
    s0 * a[2] + s1 * bz,
    s0 * a[3] + s1 * bw,
  ];
};
