/** Port of RootModule.Series — future trajectory state (position, direction, velocity).
 *
 *  Each sample stores a 4×4 transform (pos + rotation with Z = facing direction)
 *  and a velocity vector. `Control()` updates the pivot (index 0 = now) from user
 *  input and propagates forward with exponential-decay smoothing.
 */
import * as Mat from "../math/mat4.js";
import * as R from "../math/rotation.js";
import * as V from "../math/vec3.js";
import type { Mat4 } from "../math/mat4.js";
import type { Vec3 } from "../math/vec3.js";
import { TimeSeries } from "./timeseries.js";

export class RootSeries {
  transforms: Mat4[];   // [sampleCount]
  velocities: Vec3[];   // [sampleCount]

  constructor(public readonly series: TimeSeries) {
    this.transforms = Array.from({ length: series.sampleCount }, () => Mat.create());
    this.velocities = Array.from({ length: series.sampleCount }, () => V.zero());
  }

  get sampleCount(): number { return this.series.sampleCount; }
  get deltaTime(): number { return this.series.deltaTime; }

  getPosition(i: number): Vec3 { return Mat.getPosition(this.transforms[i]); }
  setPosition(i: number, v: Vec3): void { Mat.setPosition(this.transforms[i], v); }
  getDirection(i: number): Vec3 { return Mat.getAxisZ(this.transforms[i]); }
  setDirection(i: number, v: Vec3): void {
    Mat.setRotation3x3(this.transforms[i], R.lookPlanar(v));
  }
  getVelocity(i: number): Vec3 { return V.copy(this.velocities[i]); }
  setVelocity(i: number, v: Vec3): void { this.velocities[i] = V.copy(v); }

  /** Port of RootModule.Series.Control(position, direction, velocity, dt). */
  control(
    position: Vec3,
    direction: Vec3,
    velocity: Vec3,
    deltaTime: number,
    moveSensitivity = 10.0,
    turnSensitivity = 10.0,
  ): void {
    let dir = V.normalize(direction);
    if (V.lengthSq(dir) === 0) {
      if (V.lengthSq(velocity) !== 0) dir = V.normalize(velocity);
      else dir = this.getDirection(0);
    }

    // Pivot = current (index 0)
    const curVel = this.getVelocity(0);
    const newVel = V.lerpDt(curVel, velocity, deltaTime, moveSensitivity);
    this.setVelocity(0, newVel);
    this.setPosition(0, V.add(position, V.scale(newVel, deltaTime)));
    this.setDirection(0, V.slerpDt(this.getDirection(0), dir, deltaTime, turnSensitivity));

    // Forward-integrate subsequent samples.
    for (let i = 1; i < this.sampleCount; i++) {
      const ratio = (i - 0) / (this.sampleCount - 1);
      const prevVel = this.getVelocity(i - 1);
      const blendedVel = V.lerpDt(prevVel, velocity, this.deltaTime, ratio * moveSensitivity);
      this.setVelocity(i, blendedVel);
      this.setPosition(i, V.add(this.getPosition(i - 1), V.scale(blendedVel, this.deltaTime)));
      this.setDirection(i, V.slerp(this.getDirection(0), dir, ratio));
    }
  }

  /** Arc length of the trajectory. */
  getLength(): number {
    let len = 0;
    for (let i = 1; i < this.sampleCount; i++) {
      len += V.distance(this.getPosition(i - 1), this.getPosition(i));
    }
    return len;
  }

  /** Replace all samples with given data (for sequence-driven override). */
  setAll(transforms: Mat4[], velocities: Vec3[]): void {
    for (let i = 0; i < this.sampleCount; i++) {
      Mat.copy(this.transforms[i], transforms[i]);
      this.velocities[i] = V.copy(velocities[i]);
    }
  }
}
