/** LegIK — two-stage FABRIK for a biped leg.
 *
 *  Port of `Demos/Locomotion/Biped/LegIK.py`. For each leg we maintain two
 *  FABRIK chains: hip→ankle and ankle→ball. Each is solved with a contact-
 *  weighted target that clamps the foot's Y to a baseline (ground height) as
 *  the predicted contact strengthens.
 */
import * as V from "../math/vec3.js";
import type { Vec3 } from "../math/vec3.js";
import { FABRIK } from "./fabrik.js";

type Rot3x3 = [number, number, number, number, number, number, number, number, number];

/** Interpolate two rotation matrices. Matches Python's Rotation.Interpolate
 *  (lerp matrix entries → Rotation.Normalize → Rotation.Look(axisZ, axisY)).
 *  Deliberately does NOT Gram-Schmidt orthogonalize — Python's Look leaves
 *  X = cross(y, z) un-normalized. */
function interpolateRot3(a: Rot3x3, b: Rot3x3, t: number): Rot3x3 {
  const m: number[] = [];
  for (let i = 0; i < 9; i++) m.push(a[i] + (b[i] - a[i]) * t);
  const zRaw: Vec3 = [m[6], m[7], m[8]];
  const yRaw: Vec3 = [m[3], m[4], m[5]];
  const z = V.normalize(zRaw);
  const y = V.normalize(yRaw);
  const x = V.cross(y, z);
  return [x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]];
}

export class LegIK {
  /** Initial ankle Y (ground baseline). */
  readonly ankleBaseline: number;
  /** Initial ball Y (ground baseline). */
  readonly ballBaseline: number;
  /** Rest distance between ankle and ball joints. */
  readonly ankleBallDistance: number;

  private ankleTargetPos: Vec3;
  private ankleTargetRot: Rot3x3;
  private ballTargetPos: Vec3;
  private ballTargetRot: Rot3x3;

  constructor(
    public readonly ankleIK: FABRIK,
    public readonly ballIK: FABRIK,
  ) {
    const anklePos = ankleIK.lastBoneWorldPosition();
    const ballPos = ballIK.lastBoneWorldPosition();
    this.ankleBaseline = anklePos[1];
    this.ballBaseline = ballPos[1];
    this.ankleBallDistance = V.distance(anklePos, ballPos);
    this.ankleTargetPos = V.copy(anklePos);
    this.ballTargetPos = V.copy(ballPos);
    this.ankleTargetRot = ankleIK.lastBoneWorldRotation();
    this.ballTargetRot = ballIK.lastBoneWorldRotation();
  }

  solve(
    ankleContact: number,
    ballContact: number,
    maxIterations: number,
    threshold: number,
    poleTarget: Vec3 | null,
    poleWeight: number,
  ): void {
    this.solveAnkle(ankleContact, maxIterations, threshold, poleTarget, poleWeight);
    this.solveBall(ballContact, maxIterations, threshold);
  }

  private solveAnkle(
    contact: number,
    maxIterations: number,
    threshold: number,
    poleTarget: Vec3 | null,
    poleWeight: number,
  ): void {
    const weight = contact;

    // Y-lock: pull the previous target's Y toward the baseline, clamped
    // above ground.
    const locked: Vec3 = [this.ankleTargetPos[0], this.ankleTargetPos[1], this.ankleTargetPos[2]];
    const liftedY = locked[1] + (this.ankleBaseline - locked[1]) * weight;
    locked[1] = Math.max(liftedY, this.ankleBaseline);

    // Blend predicted ankle pos with locked pos by contact.
    const predicted = this.ankleIK.lastBoneWorldPosition();
    this.ankleTargetPos = [
      predicted[0] + (locked[0] - predicted[0]) * weight,
      predicted[1] + (locked[1] - predicted[1]) * weight,
      predicted[2] + (locked[2] - predicted[2]) * weight,
    ];

    // Rotation: half-lerp back toward previous target as contact strengthens.
    this.ankleTargetRot = interpolateRot3(
      this.ankleIK.lastBoneWorldRotation(),
      this.ankleTargetRot,
      0.5 * weight,
    );

    this.ankleIK.solve(
      this.ankleTargetPos,
      this.ankleTargetRot,
      maxIterations,
      threshold,
      poleTarget,
      poleWeight,
    );
  }

  private solveBall(
    contact: number,
    maxIterations: number,
    threshold: number,
  ): void {
    const weight = contact;

    const locked: Vec3 = [this.ballTargetPos[0], this.ballTargetPos[1], this.ballTargetPos[2]];
    const liftedY = locked[1] + (this.ballBaseline - locked[1]) * weight;
    locked[1] = Math.max(liftedY, this.ballBaseline);

    const predicted = this.ballIK.lastBoneWorldPosition();
    let t: Vec3 = [
      predicted[0] + (locked[0] - predicted[0]) * weight,
      predicted[1] + (locked[1] - predicted[1]) * weight,
      predicted[2] + (locked[2] - predicted[2]) * weight,
    ];

    // Enforce the ankle-ball length horizontally so the grounded height
    // above can't be pulled out of reach. Python LegIK.py lines 80-95.
    const groundedHeight = t[1];
    let dir: Vec3 = V.sub(t, this.ankleTargetPos);
    dir[1] = 0;
    const hd = V.length(dir);
    dir = hd > 1e-6 ? V.normalize(dir) : [1, 0, 0];
    const verticalOffset = groundedHeight - this.ankleTargetPos[1];
    const reachSq = Math.max(this.ankleBallDistance ** 2 - verticalOffset ** 2, 0);
    const reach = Math.sqrt(reachSq);
    t = [
      this.ankleTargetPos[0] + reach * dir[0],
      groundedHeight,
      this.ankleTargetPos[2] + reach * dir[2],
    ];
    this.ballTargetPos = t;

    this.ballTargetRot = interpolateRot3(
      this.ballIK.lastBoneWorldRotation(),
      this.ballTargetRot,
      0.5 * weight,
    );

    this.ballIK.solve(
      this.ballTargetPos,
      this.ballTargetRot,
      maxIterations,
      threshold,
      null, 0,
    );
  }
}
