/** On-screen virtual joysticks.
 *
 *  Fixed-position bases — always visible. Pressing inside a base captures
 *  its pointer (LMB on desktop, single touch on mobile) and the thumb
 *  tracks the cursor's offset from the base center, capped at RADIUS.
 *  Outputs are normalized [-1, 1] (y is screen-down positive — callers
 *  flip as needed for world-space forward).
 *
 *  Touch-primary devices show both sticks (left = movement, right =
 *  facing) plus a "Switch Character" button. Mouse-primary desktops only
 *  show the right (facing) stick — WASD remains the movement input.
 */
export class Touch {
  leftStick: [number, number] = [0, 0];
  rightStick: [number, number] = [0, 0];
  leftActive = false;
  rightActive = false;

  private static readonly RADIUS = 60;
  private rightBase: HTMLDivElement | null = null;
  private rightThumb: HTMLDivElement | null = null;

  constructor(opts?: { onSwitch?: () => void }) {
    const isTouch = Touch.isTouchDevice();
    if (isTouch) {
      this.makeStick("left");
      this.makeStick("right");
      if (opts?.onSwitch) this.makeSwitchButton(opts.onSwitch);
    } else {
      // Desktop: only the facing stick — WASD stays the movement input.
      this.makeStick("right");
    }
    document.documentElement.classList.add("has-joysticks");
  }

  static isTouchDevice(): boolean {
    return typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches;
  }

  get leftMagnitude(): number {
    return Math.hypot(this.leftStick[0], this.leftStick[1]);
  }

  private makeSwitchButton(onTap: () => void): void {
    const btn = document.createElement("button");
    btn.className = "switch-btn";
    btn.type = "button";
    btn.textContent = "Switch Character";
    btn.addEventListener("click", (e) => { e.preventDefault(); onTap(); });
    document.body.appendChild(btn);
  }

  /** Drive the right stick's visual state externally. Used by the
   *  LMB-drag-on-canvas path so the on-screen stick reflects facing input
   *  even when the user isn't touching it. Inputs are screen-pixel deltas
   *  (dy is screen-down positive); the method clamps and normalizes to the
   *  joystick radius. No-op while the right stick is being touched directly
   *  — that path drives the thumb itself. */
  visualizeRight(dx: number, dy: number, active: boolean): void {
    if (this.rightActive) return;
    if (!this.rightBase || !this.rightThumb) return;
    const mag = Math.hypot(dx, dy);
    const cap = Math.min(mag, Touch.RADIUS);
    const tx = mag > 1e-6 ? (dx / mag) * cap : 0;
    const ty = mag > 1e-6 ? (dy / mag) * cap : 0;
    this.rightBase.classList.toggle("active", active);
    this.rightThumb.style.transform =
      `translate(calc(-50% + ${tx}px), calc(-50% + ${ty}px))`;
  }

  private makeStick(side: "left" | "right"): void {
    const base = document.createElement("div");
    base.className = `joystick-base joystick-${side}`;
    const thumb = document.createElement("div");
    thumb.className = "joystick-thumb";
    base.appendChild(thumb);
    document.body.appendChild(base);
    if (side === "right") { this.rightBase = base; this.rightThumb = thumb; }

    let pointerId: number | null = null;

    const set = (clientX: number, clientY: number) => {
      const r = base.getBoundingClientRect();
      const dx = clientX - (r.left + r.width / 2);
      const dy = clientY - (r.top + r.height / 2);
      const mag = Math.hypot(dx, dy);
      const cap = Math.min(mag, Touch.RADIUS);
      const tx = mag > 1e-6 ? (dx / mag) * cap : 0;
      const ty = mag > 1e-6 ? (dy / mag) * cap : 0;
      thumb.style.transform = `translate(calc(-50% + ${tx}px), calc(-50% + ${ty}px))`;
      const nx = tx / Touch.RADIUS, ny = ty / Touch.RADIUS;
      if (side === "left") this.leftStick = [nx, ny];
      else this.rightStick = [nx, ny];
    };

    const reset = () => {
      thumb.style.transform = "translate(-50%, -50%)";
      pointerId = null;
      base.classList.remove("active");
      if (side === "left") { this.leftStick = [0, 0]; this.leftActive = false; }
      else { this.rightStick = [0, 0]; this.rightActive = false; }
    };

    base.addEventListener("pointerdown", (e) => {
      // Mouse: only LMB drags the stick. RMB on the base would otherwise
      // pop the browser context menu — contextmenu listener below prevents.
      if (e.pointerType === "mouse" && e.button !== 0) return;
      if (pointerId !== null) return;
      pointerId = e.pointerId;
      base.classList.add("active");
      base.setPointerCapture(e.pointerId);
      if (side === "left") this.leftActive = true; else this.rightActive = true;
      set(e.clientX, e.clientY);
      e.preventDefault();
    });

    base.addEventListener("pointermove", (e) => {
      if (e.pointerId !== pointerId) return;
      set(e.clientX, e.clientY);
    });

    const onEnd = (e: PointerEvent) => {
      if (e.pointerId !== pointerId) return;
      reset();
    };
    base.addEventListener("pointerup", onEnd);
    base.addEventListener("pointercancel", onEnd);
    base.addEventListener("lostpointercapture", onEnd);
    base.addEventListener("contextmenu", (e) => e.preventDefault());
  }
}
