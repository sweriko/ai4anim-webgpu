/** Unified IK wrapper — same interface over biped (dual-chain LegIK × 2) and
 *  quadruped (single-chain FABRIK × 4). Built from a bundle's metadata so
 *  NMMAgent doesn't branch on model.kind.
 */
import type { Actor } from "../runtime/actor.js";
import type { Mat4 } from "../math/mat4.js";
import * as M from "../math/mat4.js";
import type { BundleMeta } from "../model/bundle.js";
import { isBiped, isQuadruped } from "../model/bundle.js";
import { FABRIK } from "./fabrik.js";
import { LegIK } from "./leg_ik.js";
import { SingleChainIK } from "./single_chain_ik.js";

const IK_EPS = 1e-3;

export interface CharacterIK {
  /** Run every IK pass; `contacts` has 4 values in model-defined order. */
  solve(contacts: Float32Array, actor: Actor, iterations: number): void;
}

class BipedCharacterIK implements CharacterIK {
  private readonly leftLeg: LegIK;
  private readonly rightLeg: LegIK;
  private readonly leftKnee: number;
  private readonly rightKnee: number;

  constructor(actor: Actor, meta: BundleMeta) {
    if (!isBiped(meta)) throw new Error("BipedCharacterIK expects biped bundle");
    const legs = meta.skeleton.legs;
    this.leftLeg = new LegIK(
      new FABRIK(actor, legs.left_hip, legs.left_ankle),
      new FABRIK(actor, legs.left_ankle, legs.left_ball),
    );
    this.rightLeg = new LegIK(
      new FABRIK(actor, legs.right_hip, legs.right_ankle),
      new FABRIK(actor, legs.right_ankle, legs.right_ball),
    );
    this.leftKnee = legs.left_knee;
    this.rightKnee = legs.right_knee;
  }

  solve(contacts: Float32Array, actor: Actor, iterations: number): void {
    const leftPole = this.poleFromKnee(actor.transforms[this.leftKnee]);
    const rightPole = this.poleFromKnee(actor.transforms[this.rightKnee]);
    this.leftLeg.solve(contacts[0], contacts[1], iterations, IK_EPS, leftPole, 1.0);
    this.rightLeg.solve(contacts[2], contacts[3], iterations, IK_EPS, rightPole, 1.0);
  }

  private poleFromKnee(kneeWorld: Mat4) {
    // Pole target = the knee's +Z axis projected 1 unit forward (matches Python
    // Program.py Vector3.PositionFrom(Vector3(0,0,1), knee.Transform)).
    return M.transformPoint(kneeWorld, [0, 0, 1]);
  }
}

class QuadrupedCharacterIK implements CharacterIK {
  private readonly chains: SingleChainIK[];
  private readonly contactIdx: number[];

  constructor(actor: Actor, meta: BundleMeta) {
    if (!isQuadruped(meta)) throw new Error("QuadrupedCharacterIK expects quadruped bundle");
    this.chains = meta.skeleton.ik_chains.map(
      (c) => new SingleChainIK(new FABRIK(actor, c.source, c.target)));
    this.contactIdx = meta.skeleton.ik_chains.map((c) => c.contact_index);
  }

  solve(contacts: Float32Array, _actor: Actor, iterations: number): void {
    for (let i = 0; i < this.chains.length; i++) {
      this.chains[i].solve(contacts[this.contactIdx[i]], iterations, IK_EPS);
    }
  }
}

export function createCharacterIK(actor: Actor, meta: BundleMeta): CharacterIK {
  if (isQuadruped(meta)) return new QuadrupedCharacterIK(actor, meta);
  if (isBiped(meta)) return new BipedCharacterIK(actor, meta);
  throw new Error(`createCharacterIK: unknown model kind '${meta.model.kind}'`);
}
