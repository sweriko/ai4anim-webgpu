/** Keyboard + mouse input state.
 *
 *  Supports both biped control (WASD + Shift sprint + LMB-drag facing +
 *  Q/E style cycle) and quadruped control (WASD + Alt/Ctrl/Shift gait +
 *  R/T/V action posture, no facing input — quadruped faces its velocity).
 *
 *  When a `Touch` source is supplied and its left stick is active, the
 *  movement / sprint / quadruped-gait queries derive from stick magnitude
 *  instead of keys. Likewise the right stick stands in for LMB-drag facing.
 */
import type { Touch } from "./touch.js";

export type QuadrupedGait = "walk" | "pace" | "trot" | "canter";
export type QuadrupedAction = "sit" | "stand" | "lie" | null;

export class Input {
  keys = new Set<string>();
  facingMouseDown = false;
  mouseX = 0;
  mouseY = 0;
  directionMouseStart: [number, number] | null = null;
  stylePrev = false;
  styleNext = false;
  private touch?: Touch;

  constructor(target: HTMLElement, touch?: Touch) {
    this.touch = touch;
    // Keys go on window so WASD works without first clicking the canvas —
    // overlay UIs (joystick, tweakpane) eat clicks otherwise. Skip events
    // whose target is a form input so users can still type into Tweakpane.
    window.addEventListener("keydown", (e) => {
      if (Input.isFormTarget(e.target)) return;
      const k = e.key.toLowerCase();
      this.keys.add(k);
      if (k === "q") this.stylePrev = true;
      if (k === "e") this.styleNext = true;
    });
    window.addEventListener("keyup", (e) => {
      if (Input.isFormTarget(e.target)) return;
      this.keys.delete(e.key.toLowerCase());
    });
    // Pointer events filtered to real mouse — mobile pointerType="touch"
    // synthesizes mouse events on iOS Safari which would otherwise spuriously
    // trigger facing-drag when the user just taps the canvas.
    target.addEventListener("pointerdown", (e) => {
      if (e.pointerType !== "mouse") return;
      if (e.button === 0) this.facingMouseDown = true;
    });
    target.addEventListener("pointerup", (e) => {
      if (e.pointerType !== "mouse") return;
      if (e.button === 0) { this.facingMouseDown = false; this.directionMouseStart = null; }
    });
    target.addEventListener("pointermove", (e) => {
      if (e.pointerType !== "mouse") return;
      this.mouseX = e.clientX; this.mouseY = e.clientY;
    });
    target.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  private static isFormTarget(t: EventTarget | null): boolean {
    if (!(t instanceof HTMLElement)) return false;
    return t.tagName === "INPUT" || t.tagName === "TEXTAREA"
      || t.tagName === "SELECT" || t.isContentEditable;
  }

  /** Return [x, z] in [-1,1] from WASD (x: A/D, z: W/S → forward/back), or
   *  the left touch stick when active. Stick y is screen-down so it flips. */
  getMovementVector(): [number, number] {
    if (this.touch?.leftActive) {
      const [sx, sy] = this.touch.leftStick;
      return [sx, -sy];
    }
    const k = this.keys;
    let x = 0, z = 0;
    if (k.has("w")) z += 1;
    if (k.has("s")) z -= 1;
    if (k.has("a")) x -= 1;
    if (k.has("d")) x += 1;
    return [x, z];
  }

  /** Biped sprint — Shift, or any active touch (stick magnitude scales speed
   *  continuously when sprint is on, so partial deflection still walks). */
  isSprint(): boolean {
    if (this.touch?.leftActive) return true;
    return this.keys.has("shift");
  }

  /** LMB-drag facing offset, momentum-smoothed. Biped only. The right
   *  touch stick stands in when active. */
  getFacingDelta(): [number, number] {
    if (this.touch?.rightActive) {
      const [sx, sy] = this.touch.rightStick;
      return [sx, -sy];
    }
    if (!this.facingMouseDown) return [0, 0];
    const pos: [number, number] = [this.mouseX, this.mouseY];
    if (!this.directionMouseStart) {
      this.directionMouseStart = pos;
      return [0, 0];
    }
    const momentum = 0.01;
    this.directionMouseStart[0] = this.directionMouseStart[0] * (1 - momentum) + pos[0] * momentum;
    this.directionMouseStart[1] = this.directionMouseStart[1] * (1 - momentum) + pos[1] * momentum;
    return [pos[0] - this.directionMouseStart[0], this.directionMouseStart[1] - pos[1]];
  }

  consumeStylePrev(): boolean { const v = this.stylePrev; this.stylePrev = false; return v; }
  consumeStyleNext(): boolean { const v = this.styleNext; this.styleNext = false; return v; }

  // --- Quadruped-specific ------------------------------------------------

  /** Alt → walk, Ctrl → trot, Shift → canter, default → pace. On touch,
   *  the gait is selected from left-stick magnitude — capped at trot since
   *  canter is uncontrollably fast on a phone-sized viewport. */
  getQuadrupedGait(): QuadrupedGait {
    if (this.touch?.leftActive) {
      const m = this.touch.leftMagnitude;
      if (m < 0.35) return "walk";
      if (m < 0.70) return "pace";
      return "trot";
    }
    const k = this.keys;
    if (k.has("alt")) return "walk";
    if (k.has("control")) return "trot";
    if (k.has("shift")) return "canter";
    return "pace";
  }

  /** Action posture — R (Sit), T (Stand), V (Lie). None when no key held. */
  getQuadrupedAction(): QuadrupedAction {
    if (this.keys.has("r")) return "sit";
    if (this.keys.has("t")) return "stand";
    if (this.keys.has("v")) return "lie";
    return null;
  }
}
