/** Quadruped control state machine — derived from
 *  `Demos/Locomotion/Quadruped/Program.py`.
 *
 *  Ownership: one `QuadrupedControl` per agent. Game code calls
 *    - setMove(velocity, facing) each frame   → target speed from |velocity|
 *    - setAction("sit" | "stand" | "lie" | null)  → overrides movement if at rest
 *
 *  Each frame the agent runs `update(dt, currentSpeed)` to advance the PID
 *  and pick the guidance state (walk/pace/trot/canter/Idle/Sit/Stand/Lie).
 */
import type { Vec3 } from "../math/vec3.js";
import * as V from "../math/vec3.js";
import { PID } from "./pid.js";
import type { QuadrupedControlMeta } from "../model/bundle.js";

export type QuadrupedAction = "sit" | "stand" | "lie" | null;

export interface QuadrupedControlOutput {
  /** Blended velocity to feed to the simulation. */
  velocity: Vec3;
  /** Facing direction (xz), unit length. */
  direction: Vec3;
  /** Guidance state name to swap in (must exist in bundle.meta.guidances). */
  guidanceState: string;
  /** Speed that PID settled on this frame. Used by debug overlays. */
  speed: number;
}

export class QuadrupedControl {
  private requestedVelocity: Vec3 = V.zero();
  private requestedFacing: Vec3 = [0, 0, 1];
  private requestedAction: QuadrupedAction = null;
  private defaultFacing: Vec3 = [0, 0, 1];

  private readonly pid: PID;
  private readonly modes: QuadrupedControlMeta["locomotion_modes"];
  private readonly trigger: number;
  private readonly deadzone: number;

  /** Track the last non-zero facing so standing still doesn't collapse the
   *  direction vector (the sim wants a valid heading every frame). */
  private lastFacing: Vec3 = [0, 0, 1];

  constructor(meta: QuadrupedControlMeta) {
    this.modes = meta.locomotion_modes;
    this.trigger = meta.action_trigger_speed_max;
    this.deadzone = meta.input_deadzone;
    this.pid = new PID(meta.pid.kp, meta.pid.ki, meta.pid.kd);
  }

  setMove(velocity: Vec3, facing: Vec3): void {
    this.requestedVelocity = [velocity[0], 0, velocity[2]];
    const len = Math.hypot(facing[0], facing[2]);
    if (len > 1e-5) {
      this.requestedFacing = [facing[0] / len, 0, facing[2] / len];
      this.lastFacing = this.requestedFacing;
    } else {
      this.requestedFacing = V.zero();
    }
  }

  setAction(action: QuadrupedAction): void {
    this.requestedAction = action;
  }

  setDefaultFacing(facing: Vec3): void {
    const len = Math.hypot(facing[0], facing[2]);
    if (len > 1e-5) {
      this.defaultFacing = [facing[0] / len, 0, facing[2] / len];
      this.lastFacing = this.defaultFacing;
    }
  }

  update(dt: number, currentSpeed: number, currentFacing: Vec3): QuadrupedControlOutput {
    const moveMag = Math.hypot(this.requestedVelocity[0], this.requestedVelocity[2]);
    const requestedSpeed = moveMag > this.deadzone ? moveMag : 0;

    const canTriggerAction = currentSpeed < this.trigger;
    const action = canTriggerAction ? this.requestedAction : null;
    const targetSpeed = action !== null ? 0 : requestedSpeed;

    const correction = this.pid.update(currentSpeed, dt, targetSpeed);
    const speed = Math.max(currentSpeed + correction, 0);

    // Compose movement vector: when we have a move intent, aim along the
    // requested direction; when we don't, freeze but keep the current facing.
    let velocity: Vec3;
    let direction: Vec3;
    const restingFacing = this.chooseRestingFacing(currentFacing);
    if (action !== null) {
      velocity = V.zero();
      direction = restingFacing;
    } else if (moveMag > this.deadzone) {
      const inv = 1 / moveMag;
      const moveDir: Vec3 = [
        this.requestedVelocity[0] * inv, 0, this.requestedVelocity[2] * inv];
      velocity = [moveDir[0] * speed, 0, moveDir[2] * speed];
      direction = moveDir;
      this.lastFacing = moveDir;
    } else {
      velocity = V.zero();
      direction = restingFacing;
    }

    const guidanceState = this.pickGuidance(action, speed);
    return { velocity, direction, guidanceState, speed };
  }

  private chooseRestingFacing(currentFacing: Vec3): Vec3 {
    const cur = Math.hypot(currentFacing[0], currentFacing[2]);
    if (cur > 1e-5) return [currentFacing[0] / cur, 0, currentFacing[2] / cur];
    const last = Math.hypot(this.lastFacing[0], this.lastFacing[2]);
    if (last > 1e-5) return this.lastFacing;
    return this.defaultFacing;
  }

  private pickGuidance(action: QuadrupedAction, speed: number): string {
    if (action === "sit") return "Sit";
    if (action === "stand") return "Stand";
    if (action === "lie") return "Lie";
    if (speed < 0.1) return "Idle";
    if (speed < this.modes.pace) return "Walk";
    if (speed < this.modes.trot) return "Pace";
    if (speed < this.modes.canter) return "Trot";
    return "Canter";
  }
}
