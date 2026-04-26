/** Vec3 — 3-component vector, stored as [x, y, z]. Matches Python Y-up. */
export type Vec3 = [number, number, number];

export const vec3 = (x = 0, y = 0, z = 0): Vec3 => [x, y, z];
export const zero = (): Vec3 => [0, 0, 0];
export const up = (): Vec3 => [0, 1, 0];
export const unitX = (): Vec3 => [1, 0, 0];

export const copy = (a: Vec3): Vec3 => [a[0], a[1], a[2]];
export const set = (out: Vec3, x: number, y: number, z: number): void => {
  out[0] = x; out[1] = y; out[2] = z;
};
export const assign = (out: Vec3, a: Vec3): void => {
  out[0] = a[0]; out[1] = a[1]; out[2] = a[2];
};

export const add = (a: Vec3, b: Vec3): Vec3 =>
  [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: Vec3, b: Vec3): Vec3 =>
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a: Vec3, s: number): Vec3 =>
  [a[0] * s, a[1] * s, a[2] * s];
export const mul = (a: Vec3, b: Vec3): Vec3 =>
  [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
export const neg = (a: Vec3): Vec3 => [-a[0], -a[1], -a[2]];

export const dot = (a: Vec3, b: Vec3): number =>
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

export const length = (a: Vec3): number => Math.hypot(a[0], a[1], a[2]);
export const lengthSq = (a: Vec3): number =>
  a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
export const distance = (a: Vec3, b: Vec3): number =>
  Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

export const normalize = (a: Vec3): Vec3 => {
  const l = length(a);
  return l > 1e-8 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0];
};

export const clampMagnitude = (a: Vec3, max: number): Vec3 => {
  const l = length(a);
  if (l <= max || l < 1e-8) return copy(a);
  const s = max / l;
  return [a[0] * s, a[1] * s, a[2] * s];
};

export const lerp = (a: Vec3, b: Vec3, t: number): Vec3 => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];

/** In-place lerp — writes into `out`, zero allocation. */
export const lerpInto = (out: Vec3, a: Vec3, b: Vec3, t: number): void => {
  out[0] = a[0] + (b[0] - a[0]) * t;
  out[1] = a[1] + (b[1] - a[1]) * t;
  out[2] = a[2] + (b[2] - a[2]) * t;
};

/** Exponential-decay lerp: Python Vector3.LerpDt(a, b, dt, rate). */
export const lerpDt = (a: Vec3, b: Vec3, dt: number, rate: number): Vec3 =>
  lerp(a, b, 1 - Math.exp(-dt * rate));

export const slerp = (a: Vec3, b: Vec3, t: number): Vec3 => {
  const na = normalize(a);
  const nb = normalize(b);
  const d = Math.max(-1, Math.min(1, dot(na, nb)));
  if (d > 0.9995) return normalize(lerp(na, nb, t));
  const theta = Math.acos(d) * t;
  const perp = normalize(sub(nb, scale(na, d)));
  return add(scale(na, Math.cos(theta)), scale(perp, Math.sin(theta)));
};

export const slerpDt = (a: Vec3, b: Vec3, dt: number, rate: number): Vec3 =>
  slerp(a, b, 1 - Math.exp(-dt * rate));

/** Signed angle about a given axis, in DEGREES. Matches Vector3.SignedAngle
 *  which does `Rad2Deg(atan2(...))` (Vector3.py:107). */
export const signedAngle = (a: Vec3, b: Vec3, axis: Vec3): number => {
  const c = cross(a, b);
  const y = dot(c, axis);
  const x = dot(a, b);
  return Math.atan2(y, x) * 180 / Math.PI;
};
