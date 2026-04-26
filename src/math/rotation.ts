/** Rotation builders that match Python `ai4animation.Math.Rotation`.
 *
 * Rotation.Look(z, y) returns a 3×3 whose columns are [cross(y,z), y, z].
 * We export the 3×3 as a flat 9-entry column-major array so it can drop into
 * Mat4.fromTR directly.
 */
import * as V from "./vec3.js";
import type { Vec3 } from "./vec3.js";

export type Rot3x3 = [
  number, number, number,
  number, number, number,
  number, number, number,
];

/** Build a basis from forward (z) and up (y) vectors. Exact port of
 *  Python's `Rotation.Look(z, y)` (Rotation.py:94-98): only z and y are
 *  normalized; X = cross(y, z) is left un-normalized so non-orthogonal
 *  inputs produce the same non-orthogonal basis Python would. */
export const look = (z: Vec3, y: Vec3): Rot3x3 => {
  const nz = V.normalize(z);
  const ny = V.normalize(y);
  const x = V.cross(ny, nz);
  return [
    x[0], x[1], x[2],
    ny[0], ny[1], ny[2],
    nz[0], nz[1], nz[2],
  ];
};

/** Z-axis only; Y is forced to world up. */
export const lookPlanar = (z: Vec3): Rot3x3 => {
  const up: Vec3 = [0, 1, 0];
  const nz = V.normalize([z[0], 0, z[2]]);
  if (V.lengthSq(nz) < 1e-8) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const x = V.normalize(V.cross(up, nz));
  return [
    x[0], x[1], x[2],
    up[0], up[1], up[2],
    nz[0], nz[1], nz[2],
  ];
};

export const identity = (): Rot3x3 => [1, 0, 0, 0, 1, 0, 0, 0, 1];
