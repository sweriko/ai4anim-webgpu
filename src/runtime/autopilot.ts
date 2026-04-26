/** Autopilot brains — drive a non-player agent via setGoal / setStyle /
 *  setAction so it wanders around on its own. Two flavors, both sharing the
 *  same per-agent state machine shape:
 *
 *    idle  →  pick a new goal, start walking/running
 *    walk/run  →  head toward goal; on arrival or timer expiry, re-decide
 *    (quadruped extra)  occasionally trigger Sit/Stand/Lie at rest
 *
 *  Each brain owns a pseudo-RNG so its "personality" is seed-stable across
 *  sessions — nice for perf comparisons and screenshots. Feed a different
 *  `seed` per brain so the crowd doesn't all decide in lock-step.
 */
import type { NMMAgent } from "../engine/NMMAgent.js";
import type { Vec3 } from "../math/vec3.js";
import type { QuadrupedAction } from "./quadruped_control.js";

/** Lehmer LCG — deterministic, tiny, zero state overhead. */
function lcg(seed: number): () => number {
  let a = seed >>> 0;
  if (a === 0) a = 1;
  return () => {
    a = (a * 48271) % 0x7fffffff;
    return a / 0x7fffffff;
  };
}

export interface Brain {
  update(dt: number, totalTime: number): void;
}

export interface BipedBrainOptions {
  agent: NMMAgent;
  seed: number;
  /** Movement styles the brain cycles between (excludes "Idle"). */
  movementStyles: string[];
  /** Radius within which the brain picks new wander targets. */
  playfieldRadius: number;
}

export class BipedBrain implements Brain {
  private readonly agent: NMMAgent;
  private readonly movementStyles: string[];
  private readonly playfield: number;
  private readonly rand: () => number;

  private state: "walk" | "run" | "idle" = "idle";
  private timer: number;
  private goal: Vec3 = [0, 0, 0];
  private readonly baseSpeed: number;
  private readonly runSpeed: number;
  private readonly idleBias: number;
  private readonly runBias: number;
  private nextStyleSwap: number;

  constructor(opts: BipedBrainOptions) {
    this.agent = opts.agent;
    this.movementStyles = opts.movementStyles;
    this.playfield = opts.playfieldRadius;
    this.rand = lcg(opts.seed);

    this.timer = this.rand() * 2;
    this.goal = this.pickGoal();
    this.baseSpeed = 0.7 + this.rand() * 0.7;
    this.runSpeed = 1.6 + this.rand() * 0.8;
    this.idleBias = 0.10 + this.rand() * 0.20;
    this.runBias = 0.20 + this.rand() * 0.40;
    this.nextStyleSwap = 8 + this.rand() * 20;
  }

  update(dt: number, totalTime: number): void {
    // Swap style periodically so the crowd doesn't freeze on one preset.
    if (totalTime >= this.nextStyleSwap && this.movementStyles.length > 0) {
      const next = this.movementStyles[Math.floor(this.rand() * this.movementStyles.length)];
      this.agent.setStyle(next);
      this.nextStyleSwap = totalTime + 12 + this.rand() * 25;
    }

    this.timer -= dt;
    const pos = this.agent.getPosition();
    const dx = this.goal[0] - pos[0];
    const dz = this.goal[2] - pos[2];
    const dist = Math.hypot(dx, dz);

    if (this.timer <= 0 || (this.state !== "idle" && dist < 0.6)) {
      const r = this.rand();
      if (r < this.idleBias) {
        this.state = "idle";
        this.timer = 1.0 + this.rand() * 2.5;
      } else {
        this.state = r < this.idleBias + this.runBias ? "run" : "walk";
        this.timer = 4 + this.rand() * 6;
        this.goal = this.pickGoal();
      }
    }

    let vel: Vec3, facing: Vec3;
    if (this.state === "idle") {
      vel = [0, 0, 0];
      const cv = this.agent.getVelocity();
      const cvLen = Math.hypot(cv[0], cv[2]);
      facing = cvLen > 0.01 ? [cv[0] / cvLen, 0, cv[2] / cvLen] : [0, 0, 1];
    } else {
      const speed = this.state === "run" ? this.runSpeed : this.baseSpeed;
      const inv = dist > 1e-3 ? 1 / dist : 0;
      vel = [dx * inv * speed, 0, dz * inv * speed];
      facing = [dx * inv, 0, dz * inv];
    }
    this.agent.setGoal(vel, facing);
  }

  private pickGoal(): Vec3 {
    const a = this.rand() * Math.PI * 2;
    const r = Math.sqrt(this.rand()) * this.playfield;
    return [Math.cos(a) * r, 0, Math.sin(a) * r];
  }
}

export interface QuadrupedBrainOptions {
  agent: NMMAgent;
  seed: number;
  /** Speed map from the quadruped bundle (walk/pace/trot/canter). */
  speeds: { walk: number; pace: number; trot: number; canter: number };
  playfieldRadius: number;
}

export class QuadrupedBrain implements Brain {
  private readonly agent: NMMAgent;
  private readonly speeds: QuadrupedBrainOptions["speeds"];
  private readonly playfield: number;
  private readonly rand: () => number;

  private state: "wander" | "idle" | "action" = "idle";
  private gait: "walk" | "pace" | "trot" | "canter" = "walk";
  private action: QuadrupedAction = null;
  private timer: number;
  private goal: Vec3 = [0, 0, 0];
  private readonly idleBias: number;
  private readonly actionBias: number;
  /** Per-agent gait preferences — some dogs like to canter, some plod. */
  private readonly gaitWeights: Record<"walk" | "pace" | "trot" | "canter", number>;

  constructor(opts: QuadrupedBrainOptions) {
    this.agent = opts.agent;
    this.speeds = opts.speeds;
    this.playfield = opts.playfieldRadius;
    this.rand = lcg(opts.seed);

    this.timer = this.rand() * 2;
    this.goal = this.pickGoal();
    this.idleBias = 0.15 + this.rand() * 0.20;
    this.actionBias = 0.10 + this.rand() * 0.15;
    // Random gait preferences — each agent ends up favoring different gaits.
    this.gaitWeights = {
      walk:   0.25 + this.rand() * 0.5,
      pace:   0.25 + this.rand() * 0.5,
      trot:   0.25 + this.rand() * 0.5,
      canter: 0.15 + this.rand() * 0.4,
    };
  }

  update(dt: number, _totalTime: number): void {
    this.timer -= dt;
    const pos = this.agent.getPosition();
    const dx = this.goal[0] - pos[0];
    const dz = this.goal[2] - pos[2];
    const dist = Math.hypot(dx, dz);

    if (this.timer <= 0 || (this.state === "wander" && dist < 0.8)) {
      const r = this.rand();
      if (r < this.actionBias) {
        // Pick an action pose — use current speed to decide triggerable.
        const choices: QuadrupedAction[] = ["sit", "stand", "lie"];
        this.action = choices[Math.floor(this.rand() * choices.length)];
        this.state = "action";
        this.timer = 3 + this.rand() * 4;    // hold for 3-7s
      } else if (r < this.actionBias + this.idleBias) {
        this.action = null;
        this.state = "idle";
        this.timer = 1.5 + this.rand() * 2;
      } else {
        this.action = null;
        this.state = "wander";
        this.gait = this.pickGait();
        this.timer = 4 + this.rand() * 6;
        this.goal = this.pickGoal();
      }
    }

    let vel: Vec3, facing: Vec3;
    if (this.state === "wander") {
      const speed = this.speeds[this.gait];
      const inv = dist > 1e-3 ? 1 / dist : 0;
      vel = [dx * inv * speed, 0, dz * inv * speed];
      facing = [dx * inv, 0, dz * inv];
    } else {
      vel = [0, 0, 0];
      const cv = this.agent.getVelocity();
      const cvLen = Math.hypot(cv[0], cv[2]);
      facing = cvLen > 0.01 ? [cv[0] / cvLen, 0, cv[2] / cvLen] : [0, 0, 1];
    }
    this.agent.setGoal(vel, facing);
    this.agent.setAction(this.action);
  }

  private pickGoal(): Vec3 {
    const a = this.rand() * Math.PI * 2;
    const r = Math.sqrt(this.rand()) * this.playfield;
    return [Math.cos(a) * r, 0, Math.sin(a) * r];
  }

  private pickGait(): "walk" | "pace" | "trot" | "canter" {
    const w = this.gaitWeights;
    const total = w.walk + w.pace + w.trot + w.canter;
    let p = this.rand() * total;
    if ((p -= w.walk) < 0) return "walk";
    if ((p -= w.pace) < 0) return "pace";
    if ((p -= w.trot) < 0) return "trot";
    return "canter";
  }
}
