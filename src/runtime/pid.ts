/** Scalar PID controller — port of ai4animation/PID.py sized down to scalars
 *  (the quadruped demo uses it for speed tracking, 1-D). */
export class PID {
  kp: number;
  ki: number;
  kd: number;
  private setpoint: number;
  private integral = 0;
  private previousMeasurement: number | null = null;

  constructor(kp: number, ki: number, kd: number, setpoint = 0) {
    this.kp = kp; this.ki = ki; this.kd = kd;
    this.setpoint = setpoint;
  }

  reset(): void {
    this.integral = 0;
    this.previousMeasurement = null;
  }

  /** Returns the PID output (additive correction). Matches Python signature
   *  `pid(measurement, dt, setpoint=..., feedforward=0)`. */
  update(measurement: number, dt: number, setpoint?: number): number {
    if (dt <= 0) return measurement;
    if (setpoint !== undefined) this.setpoint = setpoint;
    const error = this.setpoint - measurement;
    this.integral += error * dt;
    const derivative = this.previousMeasurement === null
      ? 0 : -(measurement - this.previousMeasurement) / dt;
    this.previousMeasurement = measurement;
    return this.kp * error + this.ki * this.integral + this.kd * derivative;
  }
}
