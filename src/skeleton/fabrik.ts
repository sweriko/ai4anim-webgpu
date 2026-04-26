/** FABRIK (Forward And Backward Reaching Inverse Kinematics) solver with an
 *  optional pole-target constraint. Direct port of `ai4animation/IK/FABRIK.py`.
 *
 *  All internal work happens in the local space of the chain's root bone at
 *  solve() time — same as Python's `self.Root = Bones[0].GetTransform().copy()`
 *  — so pole/target math is numerically stable regardless of where in the
 *  world the chain lives.
 */
import * as M from "../math/mat4.js";
import * as V from "../math/vec3.js";
import type { Mat4 } from "../math/mat4.js";
import type { Vec3 } from "../math/vec3.js";
import type { Actor } from "../runtime/actor.js";

type Rot3x3 = [number, number, number, number, number, number, number, number, number];

/** Build the bone chain from `source` to `target` by walking parent pointers. */
function buildChain(actor: Actor, source: number, target: number): number[] {
  const out: number[] = [];
  let pivot = target;
  out.push(pivot);
  while (pivot !== source) {
    const p = actor.parents[pivot];
    if (p < 0) {
      throw new Error(`FABRIK: no chain from bone ${source} to ${target}`);
    }
    pivot = p;
    out.push(pivot);
  }
  out.reverse();
  return out;
}

/** Quaternion (xyzw) that rotates unit vector u onto unit vector v. */
function quatFromTo(u: Vec3, v: Vec3): [number, number, number, number] {
  const nu = V.normalize(u);
  const nv = V.normalize(v);
  const d = V.dot(nu, nv);
  // Anti-parallel case: pick any perpendicular axis, 180°.
  if (d < -0.999999) {
    const axis = Math.abs(nu[0]) < 0.9 ? V.cross(nu, [1, 0, 0]) : V.cross(nu, [0, 1, 0]);
    const a = V.normalize(axis);
    return [a[0], a[1], a[2], 0];
  }
  const c = V.cross(nu, nv);
  const w = Math.sqrt((d + 1) * 0.5);
  const inv = 1 / (2 * w);
  return [c[0] * inv, c[1] * inv, c[2] * inv, w];
}

/** xyzw quaternion → 3×3 column-major rotation matrix. */
function quatToMat3(q: [number, number, number, number]): Rot3x3 {
  const x = q[0], y = q[1], z = q[2], w = q[3];
  return [
    1 - 2 * (y * y + z * z),
    2 * (x * y + z * w),
    2 * (x * z - y * w),
    2 * (x * y - z * w),
    1 - 2 * (x * x + z * z),
    2 * (y * z + x * w),
    2 * (x * z + y * w),
    2 * (y * z - x * w),
    1 - 2 * (x * x + y * y),
  ];
}

/** `rot3x3(a) * rot3x3(b)` — row-of-a · col-of-b, column-major. */
function mat3Multiply(a: Rot3x3, b: Rot3x3): Rot3x3 {
  const out: number[] = [];
  for (let c = 0; c < 3; c++) {
    for (let r = 0; r < 3; r++) {
      out.push(
        a[r + 0] * b[c * 3 + 0] +
        a[r + 3] * b[c * 3 + 1] +
        a[r + 6] * b[c * 3 + 2],
      );
    }
  }
  return out as unknown as Rot3x3;
}

/** Extract 3×3 rotation from a Mat4 (column-major). */
function getRot3(m: Mat4): Rot3x3 {
  return [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
}

/** `to_world = space * v` interpreting space as (rot, pos). */
function positionFrom(v: Vec3, space: Mat4): Vec3 {
  return M.transformPoint(space, v);
}

/** `v_in_space_local = space^-1 * v_world`. Rotation-+-translation inverse. */
function positionTo(v: Vec3, space: Mat4): Vec3 {
  return M.inverseTransformPoint(space, v);
}

export class FABRIK {
  readonly chain: number[];
  private positions: Vec3[] = [];   // in chain-root-local space during solve
  private lengths: Float32Array;
  private root: Mat4 = M.create();   // chain[0]'s world transform at solve time

  constructor(public actor: Actor, source: number, target: number) {
    this.chain = buildChain(actor, source, target);
    this.lengths = new Float32Array(this.chain.length);
    for (let i = 0; i < this.chain.length; i++) this.positions.push([0, 0, 0]);
  }

  firstBoneIndex(): number { return this.chain[0]; }
  lastBoneIndex(): number { return this.chain[this.chain.length - 1]; }
  firstBoneWorldPosition(): Vec3 { return M.getPosition(this.actor.transforms[this.chain[0]]); }
  lastBoneWorldPosition(): Vec3 { return M.getPosition(this.actor.transforms[this.chain[this.chain.length - 1]]); }
  lastBoneWorldRotation(): Rot3x3 { return getRot3(this.actor.transforms[this.chain[this.chain.length - 1]]); }

  /** Solve the chain so that the last bone reaches `targetPos`.
   *  If `targetRot` is provided, the last bone's rotation is set to it;
   *  otherwise we keep its current world rotation. */
  solve(
    targetPos: Vec3,
    targetRot: Rot3x3 | null,
    maxIterations: number,
    threshold: number,
    poleTarget: Vec3 | null,
    poleWeight: number,
  ): void {
    this.prepare();

    const targetLocal = positionTo(targetPos, this.root);
    const poleLocal = poleTarget ? positionTo(poleTarget, this.root) : null;

    for (let iter = 0; iter < maxIterations; iter++) {
      this.backwardPass(targetLocal);
      this.forwardPass();
      if (poleLocal && poleWeight > 0) this.applyPoleConstraint(targetLocal, poleLocal, poleWeight);

      const dx = this.positions[this.chain.length - 1][0] - targetLocal[0];
      const dy = this.positions[this.chain.length - 1][1] - targetLocal[1];
      const dz = this.positions[this.chain.length - 1][2] - targetLocal[2];
      if (dx * dx + dy * dy + dz * dz < threshold * threshold) break;
    }

    this.assign(targetRot);
  }

  private prepare(): void {
    M.copy(this.root, this.actor.transforms[this.chain[0]]);
    for (let i = 0; i < this.chain.length; i++) {
      const boneIdx = this.chain[i];
      this.positions[i] = positionTo(M.getPosition(this.actor.transforms[boneIdx]), this.root);
      this.lengths[i] = this.actor.getCurrentLength(boneIdx);
    }
  }

  private backwardPass(targetLocal: Vec3): void {
    const n = this.chain.length;
    for (let i = n - 1; i >= 1; i--) {
      if (i === n - 1) {
        this.positions[i] = V.copy(targetLocal);
      } else {
        const dir = V.normalize(V.sub(this.positions[i], this.positions[i + 1]));
        const len = this.lengths[i + 1];
        this.positions[i] = V.add(this.positions[i + 1], V.scale(dir, len));
      }
    }
  }

  private forwardPass(): void {
    const n = this.chain.length;
    for (let i = 1; i < n; i++) {
      const dir = V.normalize(V.sub(this.positions[i], this.positions[i - 1]));
      const len = this.lengths[i];
      this.positions[i] = V.add(this.positions[i - 1], V.scale(dir, len));
    }
  }

  /** Rodrigues rotation of each middle joint around the chain axis to bring
   *  its projection onto the pole-target projection. `weight` in [0, 1]. */
  private applyPoleConstraint(targetLocal: Vec3, poleLocal: Vec3, weight: number): void {
    const rootPos = this.positions[0];
    const chainAxis = V.sub(targetLocal, rootPos);
    const chainLen = V.length(chainAxis);
    if (chainLen < 1e-6) return;
    const chainDir: Vec3 = [chainAxis[0] / chainLen, chainAxis[1] / chainLen, chainAxis[2] / chainLen];

    const poleRel = V.sub(poleLocal, rootPos);
    const poleDot = V.dot(poleRel, chainDir);
    const poleProj0: Vec3 = [
      poleRel[0] - poleDot * chainDir[0],
      poleRel[1] - poleDot * chainDir[1],
      poleRel[2] - poleDot * chainDir[2],
    ];
    const poleProjLen = V.length(poleProj0);
    if (poleProjLen < 1e-6) return;
    const poleProj: Vec3 = [poleProj0[0] / poleProjLen, poleProj0[1] / poleProjLen, poleProj0[2] / poleProjLen];

    for (let i = 1; i < this.chain.length - 1; i++) {
      const jointRel = V.sub(this.positions[i], rootPos);
      const jDot = V.dot(jointRel, chainDir);
      const jProj0: Vec3 = [
        jointRel[0] - jDot * chainDir[0],
        jointRel[1] - jDot * chainDir[1],
        jointRel[2] - jDot * chainDir[2],
      ];
      const jProjLen = V.length(jProj0);
      if (jProjLen < 1e-6) continue;
      const jProj: Vec3 = [jProj0[0] / jProjLen, jProj0[1] / jProjLen, jProj0[2] / jProjLen];

      // Python uses signed angle in DEGREES (Vector3.SignedAngle → Rad2Deg),
      // then Deg2Rad on the weighted angle. Net: sign * |θ| * weight in radians.
      const angle = V.signedAngle(jProj, poleProj, chainDir) * weight * Math.PI / 180;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);
      const cross = V.cross(chainDir, jointRel);
      const dProj = V.dot(jointRel, chainDir);
      const rotated: Vec3 = [
        cosA * jointRel[0] + sinA * cross[0] + (1 - cosA) * dProj * chainDir[0],
        cosA * jointRel[1] + sinA * cross[1] + (1 - cosA) * dProj * chainDir[1],
        cosA * jointRel[2] + sinA * cross[2] + (1 - cosA) * dProj * chainDir[2],
      ];
      this.positions[i] = V.add(rootPos, rotated);
    }
  }

  /** Write solved positions + derived rotations back to the actor (FK). */
  private assign(targetRot: Rot3x3 | null): void {
    const n = this.chain.length;
    const endRot = targetRot ?? this.lastBoneWorldRotation();

    for (let i = 0; i < n - 1; i++) {
      const boneIdx = this.chain[i];
      const childIdx = this.chain[i + 1];
      const newPos = positionFrom(this.positions[i], this.root);
      const curRot = getRot3(this.actor.transforms[boneIdx]);

      // `space` is the bone at its new position with its old rotation.
      const space: Mat4 = M.create();
      M.fromTR(space, newPos, curRot);

      // Where the child WOULD be if we kept its bind-local offset from `space`.
      const childBindLocalPos = M.getPosition(this.actor.zeroTransform[childIdx]);
      const fromWorld = M.transformPoint(space, childBindLocalPos);
      const toWorld = positionFrom(this.positions[i + 1], this.root);

      // Compute the world-space rotation that takes (space→from) to
      // (space→to), pre-multiplied onto the current rotation.
      const from = V.sub(fromWorld, newPos);
      const to = V.sub(toWorld, newPos);
      const q = quatFromTo(from, to);
      const delta = quatToMat3(q);
      const newRot = mat3Multiply(delta, curRot);

      // Build new world transform and apply with rigid FK.
      const newWorld = M.create();
      M.fromTR(newWorld, newPos, newRot);
      this.actor.setTransformFK(boneIdx, newWorld);
    }

    // Last bone in the chain — move to target with the target rotation.
    const lastIdx = this.chain[n - 1];
    const newPos = positionFrom(this.positions[n - 1], this.root);
    const newWorld = M.create();
    M.fromTR(newWorld, newPos, endRot);
    this.actor.setTransformFK(lastIdx, newWorld);
  }
}
