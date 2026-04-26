/** TimeSeries — a linearly-spaced grid of timestamps on [start, end]. */
export class TimeSeries {
  readonly timestamps: Float32Array;
  constructor(
    readonly start: number,
    readonly end: number,
    readonly sampleCount: number,
  ) {
    this.timestamps = new Float32Array(sampleCount);
    for (let i = 0; i < sampleCount; i++) {
      const r = sampleCount > 1 ? i / (sampleCount - 1) : 0;
      this.timestamps[i] = start + (end - start) * r;
    }
  }
  get window(): number { return this.end - this.start; }
  get deltaTime(): number { return this.window / (this.sampleCount - 1); }
}
