export * as Vec from "./vec3.js";
export * as Quat from "./quat.js";
export * as Mat from "./mat4.js";
export * as Rot from "./rotation.js";
export * as Xf from "./transform.js";
export type { Vec3 } from "./vec3.js";
export type { Quat as QuatT } from "./quat.js";
export type { Mat4 } from "./mat4.js";
export type { Rot3x3 } from "./rotation.js";

export const interpolateDt = (a: number, b: number, dt: number, rate: number): number =>
  a + (b - a) * (1 - Math.exp(-dt * rate));

export const clamp = (x: number, lo: number, hi: number): number =>
  x < lo ? lo : x > hi ? hi : x;

/** Smoothstep with power + threshold. Matches `Utility.SmoothStep`:
 *    x = clamp(x, 0, 1)
 *    t = clamp((x - threshold) / (1 - threshold), 0, 1)
 *    s = 3t² - 2t³   (Hermite cubic)
 *    return s ** power */
export const smoothStep = (x: number, threshold: number, power: number): number => {
  const xc = x < 0 ? 0 : x > 1 ? 1 : x;
  let t = (xc - threshold) / (1 - threshold);
  if (t < 0) t = 0; else if (t > 1) t = 1;
  const s = 3 * t * t - 2 * t * t * t;
  return Math.pow(s, power);
};
