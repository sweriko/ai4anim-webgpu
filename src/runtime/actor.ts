/** Actor — minimal character state: per-bone world transforms + velocities +
 *  precomputed default bone lengths (for RestoreBoneLengths). Driven every frame
 *  by the inference pipeline; syncs to Three.js bones afterwards.
 */
import * as THREE from "three";
import * as M from "../math/mat4.js";
import * as Q from "../math/quat.js";
import * as V from "../math/vec3.js";
import type { Mat4 } from "../math/mat4.js";
import type { Vec3 } from "../math/vec3.js";

// Module-level scratch — one Matrix4 reused across every call to
// writeBoneMatrices (zero allocation on the hot path).
const _tmpOffset = new THREE.Matrix4();

export class Actor {
  /** One 4×4 world transform per bone. */
  transforms: Mat4[];
  /** World-space velocity per bone. */
  velocities: Vec3[];
  /** Character root (pelvis-level) transform, separate from the bones. */
  root: Mat4 = M.create();
  /** Distance from each bone to its parent at rest. */
  defaultLengths: Float32Array;
  /** Parent index per bone, -1 if root. */
  parents: Int16Array;
  /** Bind-pose LOCAL transform of each bone relative to its parent. Python's
   *  `Bone.ZeroTransform`, used by IK to know where a child sits when its
   *  parent's rotation is taken as the reference frame. */
  zeroTransform: Mat4[];
  /** Transitive set of descendant indices per bone (for rigid-FK propagation). */
  successors: number[][];
  /** Direct children indices per bone. */
  children: number[][] = [];

  // Sync targets
  readonly threeBones: THREE.Bone[];
  readonly threeRoot: THREE.Object3D;
  /** Bind-pose world matrix of each bone (at t=0, from Three.js skeleton). */
  bindWorld: Mat4[];

  constructor(threeBones: THREE.Bone[], threeRoot: THREE.Object3D) {
    this.threeBones = threeBones;
    this.threeRoot = threeRoot;
    const n = threeBones.length;

    // Snapshot the bind pose: force Three.js world matrices, then copy.
    threeRoot.updateMatrixWorld(true);

    // Build parent indices by looking up each bone's parent in our ordered list.
    const idxOf = new Map<THREE.Object3D, number>();
    threeBones.forEach((b, i) => idxOf.set(b, i));
    this.parents = new Int16Array(n);
    for (let i = 0; i < n; i++) {
      const p = threeBones[i].parent;
      this.parents[i] = (p && idxOf.has(p)) ? idxOf.get(p)! : -1;
    }

    // Snapshot bind world transforms.
    this.bindWorld = threeBones.map((b) => {
      const m = M.create();
      m.set(b.matrixWorld.elements);
      return m;
    });

    // Start transforms at the bind pose; velocities at zero.
    this.transforms = this.bindWorld.map(M.clone);
    this.velocities = Array.from({ length: n }, () => V.zero());

    // Default bone lengths (distance to parent at bind).
    this.defaultLengths = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const p = this.parents[i];
      if (p < 0) { this.defaultLengths[i] = 0; continue; }
      const myPos = M.getPosition(this.bindWorld[i]);
      const pPos = M.getPosition(this.bindWorld[p]);
      this.defaultLengths[i] = V.distance(myPos, pPos);
    }

    // Bind-pose LOCAL transforms relative to parent.
    //   bind_local = parent_bind_world^-1 * bind_world
    this.zeroTransform = this.bindWorld.map((bw, i) => {
      const p = this.parents[i];
      if (p < 0) return M.create();   // identity
      const inv = M.create();
      M.invertAffine(inv, this.bindWorld[p]);
      const out = M.create();
      M.multiply(out, inv, bw);
      return out;
    });

    // Transitive successors (all descendants) per bone, in parent-first order.
    this.successors = Array.from({ length: n }, () => [] as number[]);
    for (let i = 0; i < n; i++) {
      let p = this.parents[i];
      while (p >= 0) {
        this.successors[p].push(i);
        p = this.parents[p];
      }
    }

    // Direct child list per bone — used by RestoreBoneAlignments (bones with
    // exactly one child get their rotation realigned toward that child).
    this.children = Array.from({ length: n }, () => [] as number[]);
    for (let i = 0; i < n; i++) {
      const p = this.parents[i];
      if (p >= 0) this.children[p].push(i);
    }

    // Character root starts at the origin (the scene root transform).
    M.identity(this.root);
  }

  /** Current distance to parent (bind pose length can be read via defaultLengths). */
  getCurrentLength(i: number): number {
    const p = this.parents[i];
    if (p < 0) return 0;
    return V.distance(M.getPosition(this.transforms[i]), M.getPosition(this.transforms[p]));
  }

  /** Apply a new world transform to bone `i` and rigidly propagate the delta
   *  to every successor. Matches Python's SetTransform(..., FK=True). */
  setTransformFK(i: number, newWorld: Mat4): void {
    const oldWorld = this.transforms[i];
    // Capture each successor's transform in OLD bone's local space.
    const oldInv = M.create();
    M.invertAffine(oldInv, oldWorld);
    const succLocal: Mat4[] = this.successors[i].map((s) => {
      const l = M.create();
      M.multiply(l, oldInv, this.transforms[s]);
      return l;
    });
    // Update the bone.
    M.copy(this.transforms[i], newWorld);
    // Restore each successor in the NEW bone's local frame.
    for (let k = 0; k < this.successors[i].length; k++) {
      const s = this.successors[i][k];
      const out = M.create();
      M.multiply(out, newWorld, succLocal[k]);
      M.copy(this.transforms[s], out);
    }
  }

  get boneCount(): number { return this.transforms.length; }

  /** Skeleton-space index of each entry in {@link threeBones}. */
  private trackedSkelIdx: Int32Array = new Int32Array(0);

  /** Topologically ordered untracked-bone entries: skeleton index + parent's
   *  skeleton index (or -1 if the parent isn't in the skeleton). Shared
   *  across all agents (shape is template-defined). */
  private untrackedCascade: readonly { skelIdx: number; parentSkelIdx: number }[] = [];

  /** Bind-pose LOCAL matrices per skeleton bone — shared template data. */
  private bindLocalMatrices: readonly THREE.Matrix4[] = [];

  /** `boneInverses[skelIdx]` per skeleton bone — shared template data. */
  private boneInverses: readonly THREE.Matrix4[] = [];

  /** Per-agent scratch: one Matrix4 per skeleton bone for the per-frame
   *  world-matrix computation. Allocated once in {@link attachToRig}. */
  private scratchWorld: THREE.Matrix4[] = [];

  /** Write this agent's skinning matrices into the shared storage buffer
   *  at its slot's offset.
   *
   *    1. For each of our 23 tracked bones, seed the skeleton-slot-indexed
   *       world matrix from `actor.transforms[i]`.
   *    2. For each untracked bone in topological order, recompute its world
   *       as `parent_world × bindLocal` — the parent is either tracked
   *       (just set) or an earlier untracked bone (already recomputed).
   *    3. For every skeleton bone, fold in `boneInverses[k]` and write the
   *       16 floats into the shared array at `slotBase + k * 16`.
   *
   *  No matrixWorld writes on the THREE.Bone objects, no scene traversal,
   *  no `skeleton.update`. The result lands directly in the StorageBuffer
   *  the custom instanced-skinning shader reads. */
  writeBoneMatrices(sharedArray: Float32Array, slotBase: number): void {
    const world = this.scratchWorld;
    const trackedIdx = this.trackedSkelIdx;
    const cascade = this.untrackedCascade;
    const bindLocals = this.bindLocalMatrices;
    const boneInverses = this.boneInverses;

    // Phase 1: tracked bones — world = actor.transforms[i].
    for (let i = 0; i < this.boneCount; i++) {
      const mw = world[trackedIdx[i]].elements;
      const t = this.transforms[i];
      for (let k = 0; k < 16; k++) mw[k] = t[k];
    }

    // Phase 2: untracked bones — cascade world = parent_world × bindLocal.
    for (let j = 0; j < cascade.length; j++) {
      const entry = cascade[j];
      const dst = world[entry.skelIdx];
      if (entry.parentSkelIdx < 0) {
        dst.copy(bindLocals[entry.skelIdx]);
      } else {
        dst.multiplyMatrices(world[entry.parentSkelIdx], bindLocals[entry.skelIdx]);
      }
    }

    // Phase 3: fold in boneInverses and write into shared storage.
    const offset = _tmpOffset;
    for (let k = 0; k < world.length; k++) {
      offset.multiplyMatrices(world[k], boneInverses[k]);
      offset.toArray(sharedArray, slotBase + k * 16);
    }
  }

  /** Wire this Actor to the shared skinned rig. Called once per agent.
   *
   *    - Caches each tracked bone's skeleton-slot index (used to locate
   *      the right mat4 inside this agent's storage slice).
   *    - Keeps references to the rig's `untrackedCascade`, `bindLocalMatrices`,
   *      and `boneInverses` (all template-derived, shared across agents).
   *    - Allocates `scratchWorld[totalBones]` — per-agent scratch for the
   *      cascading world-matrix compute in writeBoneMatrices. */
  attachToRig(rig: {
    totalBones: number;
    boneNameToIndex: Map<string, number>;
    untrackedCascade: readonly { skelIdx: number; parentSkelIdx: number }[];
    bindLocalMatrices: readonly THREE.Matrix4[];
    boneInverses: readonly THREE.Matrix4[];
  }): void {
    this.trackedSkelIdx = new Int32Array(this.threeBones.length);
    for (let i = 0; i < this.threeBones.length; i++) {
      const idx = rig.boneNameToIndex.get(this.threeBones[i].name);
      if (idx === undefined) {
        throw new Error(
          `Actor.attachToRig: tracked bone '${this.threeBones[i].name}' missing from shared rig skeleton`);
      }
      this.trackedSkelIdx[i] = idx;
    }
    this.untrackedCascade = rig.untrackedCascade;
    this.bindLocalMatrices = rig.bindLocalMatrices;
    this.boneInverses = rig.boneInverses;
    this.scratchWorld = Array.from({ length: rig.totalBones }, () => new THREE.Matrix4());
  }

  /** Restore each bone's distance from its parent to the rest-pose length.
   *  Called after inference to counteract predicted-position drift. */
  restoreBoneLengths(): void {
    for (let i = 0; i < this.boneCount; i++) {
      const p = this.parents[i];
      if (p < 0) continue;
      const parentPos = M.getPosition(this.transforms[p]);
      const pos = M.getPosition(this.transforms[i]);
      const dir = V.sub(pos, parentPos);
      const len = V.length(dir);
      if (len < 1e-6) continue;
      const scale = this.defaultLengths[i] / len;
      const newPos: Vec3 = [
        parentPos[0] + dir[0] * scale,
        parentPos[1] + dir[1] * scale,
        parentPos[2] + dir[2] * scale,
      ];
      M.setPosition(this.transforms[i], newPos);
    }
  }

  /** For every bone with exactly one child, re-align the bone's rotation so
   *  the child's bind-relative direction matches where the child actually is.
   *  Matches `Actor.RestoreBoneAlignments` / `Bone.RestoreAlignment`.
   *
   *  Note: the parent's rotation is updated but its children are NOT moved —
   *  matches Python's SetRotation without FK=True. */
  restoreBoneAlignments(): void {
    for (let i = 0; i < this.boneCount; i++) {
      if (this.children[i].length !== 1) continue;
      const childIdx = this.children[i][0];
      const parentPos = M.getPosition(this.transforms[i]);
      const childCurrent = M.getPosition(this.transforms[childIdx]);
      // Where the child would be given the parent's current transform and the
      // child's bind-local offset.
      const childBindLocal = M.getPosition(this.zeroTransform[childIdx]);
      const childFromBind = M.transformPoint(this.transforms[i], childBindLocal);

      const from: Vec3 = [
        childFromBind[0] - parentPos[0],
        childFromBind[1] - parentPos[1],
        childFromBind[2] - parentPos[2],
      ];
      const to: Vec3 = [
        childCurrent[0] - parentPos[0],
        childCurrent[1] - parentPos[1],
        childCurrent[2] - parentPos[2],
      ];
      const fl = Math.hypot(from[0], from[1], from[2]);
      const tl = Math.hypot(to[0], to[1], to[2]);
      if (fl < 1e-6 || tl < 1e-6) continue;
      const u: Vec3 = [from[0] / fl, from[1] / fl, from[2] / fl];
      const v: Vec3 = [to[0] / tl, to[1] / tl, to[2] / tl];
      const d = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
      if (d > 0.9999999) continue;    // already aligned
      // Quaternion from u → v, then rotation matrix, then pre-multiply.
      let qx: number, qy: number, qz: number, qw: number;
      if (d < -0.999999) {
        const axis = Math.abs(u[0]) < 0.9 ? [u[1] * 0 - u[2], u[2] - u[0] * 0, u[0] - u[1]] as Vec3 : [0, -u[2], u[1]] as Vec3;
        const al = Math.hypot(axis[0], axis[1], axis[2]);
        qx = axis[0] / al; qy = axis[1] / al; qz = axis[2] / al; qw = 0;
      } else {
        const cx = u[1] * v[2] - u[2] * v[1];
        const cy = u[2] * v[0] - u[0] * v[2];
        const cz = u[0] * v[1] - u[1] * v[0];
        qw = Math.sqrt((d + 1) * 0.5);
        const inv = 1 / (2 * qw);
        qx = cx * inv; qy = cy * inv; qz = cz * inv;
      }
      // Delta rotation matrix (column-major 3×3).
      const dr: [number, number, number, number, number, number, number, number, number] = [
        1 - 2 * (qy * qy + qz * qz),
        2 * (qx * qy + qz * qw),
        2 * (qx * qz - qy * qw),
        2 * (qx * qy - qz * qw),
        1 - 2 * (qx * qx + qz * qz),
        2 * (qy * qz + qx * qw),
        2 * (qx * qz + qy * qw),
        2 * (qy * qz - qx * qw),
        1 - 2 * (qx * qx + qy * qy),
      ];
      // Pre-multiply: new = dr * currentRot. Column-major matmul on the 3×3
      // portion in-place.
      const m = this.transforms[i];
      const c0 = [m[0], m[1], m[2]];
      const c1 = [m[4], m[5], m[6]];
      const c2 = [m[8], m[9], m[10]];
      m[0] = dr[0] * c0[0] + dr[3] * c0[1] + dr[6] * c0[2];
      m[1] = dr[1] * c0[0] + dr[4] * c0[1] + dr[7] * c0[2];
      m[2] = dr[2] * c0[0] + dr[5] * c0[1] + dr[8] * c0[2];
      m[4] = dr[0] * c1[0] + dr[3] * c1[1] + dr[6] * c1[2];
      m[5] = dr[1] * c1[0] + dr[4] * c1[1] + dr[7] * c1[2];
      m[6] = dr[2] * c1[0] + dr[5] * c1[1] + dr[8] * c1[2];
      m[8] = dr[0] * c2[0] + dr[3] * c2[1] + dr[6] * c2[2];
      m[9] = dr[1] * c2[0] + dr[4] * c2[1] + dr[7] * c2[2];
      m[10] = dr[2] * c2[0] + dr[5] * c2[1] + dr[8] * c2[2];
    }
  }

  setTransformFromTRQuat(i: number, pos: Vec3, q: Q.Quat): void {
    const r: [number, number, number, number, number, number, number, number, number] = [
      1 - 2 * (q[1] * q[1] + q[2] * q[2]),
      2 * (q[0] * q[1] + q[2] * q[3]),
      2 * (q[0] * q[2] - q[1] * q[3]),
      2 * (q[0] * q[1] - q[2] * q[3]),
      1 - 2 * (q[0] * q[0] + q[2] * q[2]),
      2 * (q[1] * q[2] + q[0] * q[3]),
      2 * (q[0] * q[2] + q[1] * q[3]),
      2 * (q[1] * q[2] - q[0] * q[3]),
      1 - 2 * (q[0] * q[0] + q[1] * q[1]),
    ];
    M.fromTR(this.transforms[i], pos, r);
  }
}
