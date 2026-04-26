/** NMMAgent — one neural-motion-matched character instance.
 *
 *  Owned by an `NMMEngine`. The engine feeds its input slot into the batched
 *  inference each prediction tick, then delivers the output slice back to the
 *  agent via `onPrediction()`. Every render frame the engine calls the
 *  agent's `updateControl()` and `animate()` to advance its kinematic state.
 *
 *  Game code talks to the agent via:
 *    - setGoal(velocity, facing): the motion you want the character to pursue
 *    - setStyle(name): one of bundle.meta.guidances (biped only — quadruped
 *                      picks guidance from the speed+action state machine)
 *    - setAction(name | null): quadruped "Sit"/"Stand"/"Lie" posture toggle
 *    - getPosition() / getTransform(): read the current world state
 *
 *  All coordinate frames are world space.
 */

import * as THREE from "three";
import * as M from "../math/mat4.js";
import * as Q from "../math/quat.js";
import * as R from "../math/rotation.js";
import * as V from "../math/vec3.js";
import type { Mat4 } from "../math/mat4.js";
import type { Vec3 } from "../math/vec3.js";
import { clamp, interpolateDt, smoothStep } from "../math/index.js";

import { Actor } from "../runtime/actor.js";
import { RootSeries } from "../runtime/root_series.js";
import { Sequence } from "../runtime/sequence.js";
import { TimeSeries } from "../runtime/timeseries.js";
import { IO } from "../runtime/io.js";
import { QuadrupedControl } from "../runtime/quadruped_control.js";
import type { QuadrupedAction } from "../runtime/quadruped_control.js";
import { createCharacterIK } from "../skeleton/character_ik.js";
import type { CharacterIK } from "../skeleton/character_ik.js";

import type { Bundle } from "../model/bundle.js";
import { isQuadruped } from "../model/bundle.js";
import type { SharedSkinnedRig } from "./SharedSkinnedMesh.js";

export interface NMMAgentOptions {
  position?: Vec3;
  facing?: Vec3;
  style?: string;
  predictionFps?: number;
  ikIterations?: number;
}

export class NMMAgent {
  readonly actor: Actor;

  private readonly bundle: Bundle;
  private readonly rig: SharedSkinnedRig;
  private readonly io: IO;
  private readonly ctrl: TimeSeries;
  private readonly simulation: RootSeries;
  private readonly rootControl: RootSeries;
  private readonly characterIK: CharacterIK;
  private readonly quadrupedControl: QuadrupedControl | null;
  private readonly seqLen: number;
  private readonly seqWindow: number;
  private readonly boneCount: number;
  private readonly contactCount: number;
  private readonly bones: THREE.Bone[];
  private readonly ikIterations: number;

  // Desired goal — updated by game code. Internal smoothed by RootSeries.
  private desiredVelocity: Vec3 = V.zero();
  private desiredFacing: Vec3 = [0, 0, 1];

  // Style + guidance.
  private styleName: string;
  private selectedGuidance: Vec3[];
  private currentGuidanceState: string;

  // Prediction state.
  private previous: Sequence | null = null;
  private current: Sequence | null = null;
  private timescale = 1.0;
  private synchronization = 0.0;
  private lastPrevT = 0;
  private lastCurT = 0;
  private lastBlend = 0;

  // Scratch buffers.
  private readonly scratchPrevPos: Vec3[] = [];
  private readonly scratchCurPos: Vec3[] = [];
  private readonly scratchPrevQ: Q.Quat[] = [];
  private readonly scratchCurQ: Q.Quat[] = [];
  private readonly scratchPrevVel: Vec3[] = [];
  private readonly scratchCurVel: Vec3[] = [];
  private readonly scratchPrevContacts: Float32Array;
  private readonly scratchCurContacts: Float32Array;
  private readonly scratchContacts: Float32Array;
  private readonly scratchPrevRoot = M.create();
  private readonly scratchCurRoot = M.create();
  private readonly scratchBlendedRoot = M.create();
  private readonly scratchLerpPos: Vec3 = [0, 0, 0];
  private readonly scratchSlerpQ: Q.Quat = [0, 0, 0, 1];
  private readonly scratchLerpVel: Vec3 = [0, 0, 0];

  // Engine-assigned input buffer slot.
  slot = -1;

  constructor(bundle: Bundle, rig: SharedSkinnedRig, opts: NMMAgentOptions) {
    this.bundle = bundle;
    this.rig = rig;
    this.ikIterations = opts.ikIterations ?? 1;
    this.io = new IO(bundle.meta);

    this.bones = rig.trackedBones;
    this.actor = new Actor(this.bones, rig.templateRoot);
    this.boneCount = this.actor.boneCount;
    this.contactCount = 4;   // fixed in the model output layout for both kinds

    for (let i = 0; i < this.boneCount; i++) {
      this.scratchPrevPos.push([0, 0, 0]);
      this.scratchCurPos.push([0, 0, 0]);
      this.scratchPrevQ.push([0, 0, 0, 1]);
      this.scratchCurQ.push([0, 0, 0, 1]);
      this.scratchPrevVel.push([0, 0, 0]);
      this.scratchCurVel.push([0, 0, 0]);
    }
    this.scratchPrevContacts = new Float32Array(this.contactCount);
    this.scratchCurContacts = new Float32Array(this.contactCount);
    this.scratchContacts = new Float32Array(this.contactCount);

    this.actor.attachToRig(rig);

    this.characterIK = createCharacterIK(this.actor, bundle.meta);

    // Control / root series.
    const control = bundle.meta.control;
    this.seqLen = control.sequence_length;
    this.seqWindow = control.sequence_window;
    this.ctrl = new TimeSeries(0, this.seqWindow, this.seqLen);
    this.simulation = new RootSeries(this.ctrl);
    this.rootControl = new RootSeries(this.ctrl);

    // Initial position / facing.
    const position: Vec3 = opts.position ?? [0, 0, 0];
    const facing: Vec3 = opts.facing ?? [0, 0, 1];
    this.setInitialPose(position, facing);

    // Style + guidance.
    const styles = bundle.meta.guidances;
    this.styleName = opts.style ?? styles[0];
    this.selectedGuidance = this.loadGuidance(this.styleName);
    this.currentGuidanceState = this.styleName;

    // Quadruped-only control state machine.
    if (isQuadruped(bundle.meta)) {
      this.quadrupedControl = new QuadrupedControl(bundle.meta.control);
      this.quadrupedControl.setDefaultFacing(facing);
    } else {
      this.quadrupedControl = null;
    }
  }

  setGoal(velocity: Vec3, facing: Vec3): void {
    this.desiredVelocity = [velocity[0], 0, velocity[2]];
    const len = Math.hypot(facing[0], facing[2]);
    this.desiredFacing = len > 1e-6
      ? [facing[0] / len, 0, facing[2] / len]
      : [0, 0, 1];
    if (this.quadrupedControl) {
      this.quadrupedControl.setMove(this.desiredVelocity, this.desiredFacing);
    }
  }

  /** Quadruped action posture (Sit / Stand / Lie). No-op on biped. */
  setAction(action: QuadrupedAction): void {
    if (this.quadrupedControl) this.quadrupedControl.setAction(action);
  }

  setStyle(name: string): void {
    if (!this.bundle.meta.guidances.includes(name)) return;
    this.styleName = name;
    // On quadruped the guidance state is speed-driven, so setStyle doesn't
    // immediately change what the network sees — the next update() cycle
    // recomputes based on the picked state.
    if (!this.quadrupedControl) {
      this.selectedGuidance = this.loadGuidance(name);
      this.currentGuidanceState = name;
    }
  }

  get style(): string { return this.styleName; }
  get guidanceState(): string { return this.currentGuidanceState; }

  getPosition(): Vec3 { return M.getPosition(this.actor.root); }
  getTransform(): Mat4 { return M.clone(this.actor.root); }
  getVelocity(): Vec3 { return V.copy(this.simulation.velocities[0]); }

  resetTo(position: Vec3, facing: Vec3): void {
    this.setInitialPose(position, facing);
    this.previous = null;
    this.current = null;
    this.timescale = 1.0;
    this.synchronization = 0.0;
    if (this.quadrupedControl) this.quadrupedControl.setDefaultFacing(facing);
  }

  // ------------------------------------------------------------------
  //  Engine-internal API — invoked by NMMEngine.update()
  // ------------------------------------------------------------------

  updateControl(dt: number): void {
    const ctl = this.bundle.meta.control;

    let vel: Vec3;
    let direction: Vec3;
    let guidanceState: string;

    if (this.quadrupedControl) {
      const currentSpeed = this.current ? this.current.getLength() / this.seqWindow : 0;
      const currentFacing = M.getAxisZ(this.actor.root);
      const out = this.quadrupedControl.update(dt, currentSpeed, currentFacing);
      vel = out.velocity;
      direction = out.direction;
      guidanceState = out.guidanceState;
    } else {
      vel = V.clampMagnitude(this.desiredVelocity, 3.0);
      direction = this.desiredFacing;
      if (V.lengthSq(direction) < 1e-4) {
        if (V.lengthSq(vel) > 1e-4) direction = V.normalize(vel);
        else direction = this.simulation.getDirection(0);
      }
      // Biped: style picker chooses guidance, Idle swap when near-stationary.
      const speed = V.length(vel);
      guidanceState = speed < 0.1 ? "Idle" : this.styleName;
    }

    this.currentGuidanceState = guidanceState;
    this.selectedGuidance = this.loadGuidance(
      this.bundle.meta.guidances.includes(guidanceState)
        ? guidanceState
        : this.styleName,
    );

    // Pivot the simulation at the agent's current world position — blended
    // toward the already-predicted trajectory by `synchronization`.
    const position = V.lerp(
      this.simulation.getPosition(0),
      M.getPosition(this.actor.root),
      this.synchronization,
    );
    this.simulation.control(position, direction, vel, dt);

    // Compose the rootControl series as a blend between simulation and the
    // root predicted by the last inference (if any).
    if (this.current) {
      const tc = ctl.trajectory_correction;
      for (let i = 0; i < this.rootControl.sampleCount; i++) {
        const blended = M.interpolate(
          this.simulation.transforms[i], this.current.rootTransforms[i], tc);
        M.copy(this.rootControl.transforms[i], blended);
      }
      for (let i = 0; i < this.rootControl.sampleCount; i++) {
        const target = M.getPosition(this.rootControl.transforms[i]);
        const curActor = M.getPosition(this.actor.root);
        const residual = [
          target[0] - curActor[0], target[1] - curActor[1], target[2] - curActor[2],
        ];
        const time = this.rootControl.series.timestamps[i] - this.rootControl.series.timestamps[0];
        const safeTime = time < 1e-3 ? 1e-3 : time;
        this.rootControl.velocities[i] = [residual[0] / safeTime, 0, residual[2] / safeTime];
      }
      for (let i = 0; i < this.rootControl.sampleCount; i++) {
        this.rootControl.velocities[i] = V.lerp(
          this.rootControl.velocities[i], this.current.rootVelocities[i], tc);
      }
    } else {
      for (let i = 0; i < this.rootControl.sampleCount; i++) {
        M.copy(this.rootControl.transforms[i], this.simulation.transforms[i]);
        this.rootControl.velocities[i] = V.copy(this.simulation.velocities[i]);
      }
    }
  }

  writeInputTo(dst: Float32Array, stride: number): void {
    if (this.slot < 0) throw new Error("NMMAgent.writeInputTo called before engine registration");
    const inputVec = this.io.feed(this.actor, this.rootControl, this.selectedGuidance);
    dst.set(inputVec, this.slot * stride);
  }

  onPrediction(outputSlice: Float32Array, creationTime: number): void {
    const seq = this.buildSequence(outputSlice);
    seq.creationTime = creationTime;
    this.previous = this.current ?? seq;
    this.current = seq;
  }

  animate(dt: number, totalTime: number): void {
    if (!this.current || !this.previous) return;

    const ctl = this.bundle.meta.control;
    const seqWindow = this.seqWindow;

    const requiredSpeed = (V.distance(
      M.getPosition(this.actor.root), this.simulation.getPosition(0))
      + this.simulation.getLength()) / seqWindow;
    const predictedSpeed = this.current.getLength() / seqWindow;
    let ts = 1.0, sync = 0.0;
    if (requiredSpeed > 0.1 && predictedSpeed > 0.1) {
      ts = requiredSpeed / predictedSpeed;
      sync = 1.0;
    }
    this.timescale = interpolateDt(this.timescale, ts, dt, ctl.timescale_sensitivity);
    this.timescale = clamp(this.timescale, ctl.min_timescale, ctl.max_timescale);
    this.synchronization = interpolateDt(
      this.synchronization, sync, dt, ctl.synchronization_sensitivity);

    const sdt = dt * this.timescale;
    const prevT = (totalTime - this.previous.creationTime) * this.timescale;
    const curT = (totalTime - this.current.creationTime) * this.timescale;
    const blend = clamp((totalTime - this.current.creationTime) * ctl.prediction_fps, 0, 1);
    this.lastPrevT = prevT;
    this.lastCurT = curT;
    this.lastBlend = blend;

    const boneCount = this.boneCount;
    this.previous.sampleRoot(prevT, this.scratchPrevRoot);
    this.current.sampleRoot(curT, this.scratchCurRoot);
    M.interpolateInto(this.scratchBlendedRoot, this.scratchPrevRoot, this.scratchCurRoot, blend);
    this.previous.sampleBonePositionsInto(prevT, this.scratchPrevPos);
    this.current.sampleBonePositionsInto(curT, this.scratchCurPos);
    this.previous.sampleBoneQuaternionsInto(prevT, this.scratchPrevQ);
    this.current.sampleBoneQuaternionsInto(curT, this.scratchCurQ);
    this.previous.sampleBoneVelocitiesInto(prevT, this.scratchPrevVel);
    this.current.sampleBoneVelocitiesInto(curT, this.scratchCurVel);
    this.previous.sampleContactsInto(prevT, this.scratchPrevContacts);
    this.current.sampleContactsInto(curT, this.scratchCurContacts);
    const contacts = this.scratchContacts;
    for (let i = 0; i < contacts.length; i++) {
      contacts[i] = this.scratchPrevContacts[i]
        + (this.scratchCurContacts[i] - this.scratchPrevContacts[i]) * blend;
    }

    const lock = this.current.getRootLock();
    M.interpolateIntoFirst(this.actor.root, this.scratchBlendedRoot, this.actor.root, lock);

    for (let i = 0; i < boneCount; i++) {
      V.lerpInto(this.scratchLerpPos, this.scratchPrevPos[i], this.scratchCurPos[i], blend);
      Q.slerpInto(this.scratchSlerpQ, this.scratchPrevQ[i], this.scratchCurQ[i], blend);
      V.lerpInto(this.scratchLerpVel, this.scratchPrevVel[i], this.scratchCurVel[i], blend);

      const cur = this.actor.transforms[i];
      const projX = cur[12] + this.scratchLerpVel[0] * sdt;
      const projY = cur[13] + this.scratchLerpVel[1] * sdt;
      const projZ = cur[14] + this.scratchLerpVel[2] * sdt;
      this.scratchLerpPos[0] = projX + (this.scratchLerpPos[0] - projX) * 0.5;
      this.scratchLerpPos[1] = projY + (this.scratchLerpPos[1] - projY) * 0.5;
      this.scratchLerpPos[2] = projZ + (this.scratchLerpPos[2] - projZ) * 0.5;

      this.actor.setTransformFromTRQuat(i, this.scratchLerpPos, this.scratchSlerpQ);
      V.assign(this.actor.velocities[i], this.scratchLerpVel);
    }

    this.actor.restoreBoneLengths();
    this.actor.restoreBoneAlignments();

    this.characterIK.solve(contacts, this.actor, this.ikIterations);
  }

  writeBoneMatricesToRig(sharedArray: Float32Array): void {
    if (this.slot < 0) return;
    this.actor.writeBoneMatrices(sharedArray, this.slot * this.rig.agentStride);
  }

  get debugState() {
    return {
      simulation: this.simulation,
      rootControl: this.rootControl,
      previous: this.previous,
      current: this.current,
      guidance: this.selectedGuidance,
      guidanceState: this.currentGuidanceState,
      timescale: this.timescale,
      synchronization: this.synchronization,
      prevT: this.lastPrevT,
      curT: this.lastCurT,
      blend: this.lastBlend,
      contacts: this.scratchContacts,
    };
  }

  // ------------------------------------------------------------------
  //  Internals
  // ------------------------------------------------------------------

  private loadGuidance(name: string): Vec3[] {
    const flat = this.bundle.guidance(name);
    const out: Vec3[] = [];
    for (let i = 0; i < this.bundle.meta.skeleton.bone_count; i++) {
      out.push([flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]]);
    }
    return out;
  }

  private setInitialPose(position: Vec3, facing: Vec3): void {
    const len = Math.hypot(facing[0], facing[2]);
    const f: Vec3 = len > 1e-6 ? [facing[0] / len, 0, facing[2] / len] : [0, 0, 1];
    const right: Vec3 = [f[2], 0, -f[0]];
    const r: R.Rot3x3 = [
      right[0], right[1], right[2],
      0,        1,        0,
      f[0],     f[1],     f[2],
    ];
    M.fromTR(this.actor.root, position, r);
    for (let i = 0; i < this.seqLen; i++) {
      M.copy(this.simulation.transforms[i], this.actor.root);
      this.simulation.velocities[i] = V.zero();
      M.copy(this.rootControl.transforms[i], this.actor.root);
      this.rootControl.velocities[i] = V.zero();
    }
    for (let i = 0; i < this.boneCount; i++) {
      M.copy(this.actor.transforms[i], this.actor.bindWorld[i]);
    }
  }

  private buildSequence(out: Float32Array): Sequence {
    const ctl = this.bundle.meta.control;
    const seq = new Sequence();
    seq.timestamps = new Float32Array(this.seqLen);
    for (let i = 0; i < this.seqLen; i++) {
      seq.timestamps[i] = (i / (this.seqLen - 1)) * this.seqWindow;
    }

    const res = this.io.read(out, this.actor.root);
    seq.rootTransforms = res.frames.map((f) => f.rootTransform);
    seq.rootVelocities = res.frames.map((f) => f.rootVelocity);
    seq.contacts = res.frames.map((f) => {
      const outc = new Float32Array(this.contactCount);
      for (let i = 0; i < this.contactCount; i++) {
        outc[i] = smoothStep(f.contacts[i], ctl.contact_threshold, ctl.contact_power);
      }
      return outc;
    });
    seq.bonePositions = res.frames.map((f) => f.bonePositions);
    seq.boneQuaternions = res.frames.map((f) => {
      const qs: Q.Quat[] = [];
      for (let i = 0; i < f.bonePositions.length; i++) {
        const rot = R.look(f.boneAxesZ[i], f.boneAxesY[i]);
        qs.push(mat3ToQuat(rot));
      }
      return qs;
    });
    seq.boneVelocities = res.frames.map((f) => f.boneVelocities);
    seq.guidances = res.frames.map((f) => f.guidances);
    return seq;
  }
}

function mat3ToQuat(r: R.Rot3x3): Q.Quat {
  const m00 = r[0], m10 = r[1], m20 = r[2];
  const m01 = r[3], m11 = r[4], m21 = r[5];
  const m02 = r[6], m12 = r[7], m22 = r[8];
  const trace = m00 + m11 + m22;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    return [(m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s, 0.25 / s];
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    return [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s];
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    return [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s];
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    return [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s];
  }
}
