/** SingleChainIK — contact-gated single FABRIK chain.
 *
 *  Port of `Demos/Locomotion/Quadruped/LegIK.py`. Unlike the biped's dual-chain
 *  LegIK, this wraps ONE FABRIK chain (e.g. forearm→hand site, or knee→foot
 *  site). As contact strengthens the end-effector's target is lerped back
 *  toward its previous target position — giving a ground-stick effect without
 *  any explicit Y-lock or ankle-ball length constraint.
 *
 *  Rotation uses the current chain's end-bone rotation every solve (no
 *  interpolation) — matches the Python reference.
 */
import * as V from "../math/vec3.js";
import type { Vec3 } from "../math/vec3.js";
import { FABRIK } from "./fabrik.js";

type Rot3x3 = [number, number, number, number, number, number, number, number, number];

export class SingleChainIK {
  /** End-effector Y at bind pose — kept so callers can query it for grounding UI. */
  readonly eeBaseline: number;

  private targetPos: Vec3;

  constructor(public readonly ik: FABRIK) {
    const eePos = ik.lastBoneWorldPosition();
    this.eeBaseline = eePos[1];
    this.targetPos = V.copy(eePos);
  }

  /** Solve the chain, blending predicted end-effector position back toward
   *  the previous target by `contact ∈ [0, 1]`. */
  solve(contact: number, maxIterations: number, threshold: number): void {
    const predicted = this.ik.lastBoneWorldPosition();
    this.targetPos = [
      predicted[0] + (this.targetPos[0] - predicted[0]) * contact,
      predicted[1] + (this.targetPos[1] - predicted[1]) * contact,
      predicted[2] + (this.targetPos[2] - predicted[2]) * contact,
    ];
    const targetRot: Rot3x3 = this.ik.lastBoneWorldRotation();

    this.ik.solve(this.targetPos, targetRot, maxIterations, threshold, null, 0);
  }
}
