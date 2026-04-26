/** Debug visualization overlays — port of `Program.py` Draw() + GUI() bars.
 *
 *  Individual layers:
 *    - simulation: control-target trajectory (always on; matches Python's
 *      unconditional `self.SimulationObject.Draw()`)
 *    - rootControl: the simulation-vs-prediction blended trajectory
 *    - prevSeq / curSeq: 16-frame ghost skeletons of each predicted sequence
 *    - guidance: magenta target joint positions
 *
 *  Each uses a pair of Three.js objects (line + direction-arrow segments) for
 *  trajectories, or a `LineSegments` pair for ghost skeletons, or `Points` for
 *  guidance. Geometry buffers are allocated once, updated in-place each frame.
 */
import * as THREE from "three";
import type { RootSeries } from "../runtime/root_series.js";
import type { Sequence } from "../runtime/sequence.js";
import type { Actor } from "../runtime/actor.js";
import * as M from "../math/mat4.js";

const SEQ = 16;

/** One trajectory visualization — a polyline of positions plus per-sample
 *  direction arrows (Z-axis out 0.5 m) and velocity arrows. */
class TrajectoryViz {
  readonly group = new THREE.Group();
  private posGeom = new THREE.BufferGeometry();
  private dirGeom = new THREE.BufferGeometry();
  private velGeom = new THREE.BufferGeometry();
  private posBuf = new Float32Array(SEQ * 3);
  private dirBuf = new Float32Array(SEQ * 2 * 3);   // 2 endpoints per arrow
  private velBuf = new Float32Array(SEQ * 2 * 3);

  constructor(color: number, dirColor: number, velColor: number) {
    this.posGeom.setAttribute("position", new THREE.BufferAttribute(this.posBuf, 3));
    this.dirGeom.setAttribute("position", new THREE.BufferAttribute(this.dirBuf, 3));
    this.velGeom.setAttribute("position", new THREE.BufferAttribute(this.velBuf, 3));

    this.group.add(new THREE.Line(this.posGeom,
      new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.9 })));
    this.group.add(new THREE.LineSegments(this.dirGeom,
      new THREE.LineBasicMaterial({ color: dirColor, depthTest: false, transparent: true, opacity: 0.9 })));
    this.group.add(new THREE.LineSegments(this.velGeom,
      new THREE.LineBasicMaterial({ color: velColor, depthTest: false, transparent: true, opacity: 0.7 })));
    this.group.renderOrder = 10;
  }

  update(rs: RootSeries): void {
    for (let i = 0; i < SEQ; i++) {
      const p = M.getPosition(rs.transforms[i]);
      const z = M.getAxisZ(rs.transforms[i]);
      const v = rs.velocities[i];
      this.posBuf[i * 3] = p[0];
      this.posBuf[i * 3 + 1] = p[1] + 0.01;
      this.posBuf[i * 3 + 2] = p[2];

      const di = i * 6;
      this.dirBuf[di] = p[0];
      this.dirBuf[di + 1] = p[1] + 0.01;
      this.dirBuf[di + 2] = p[2];
      this.dirBuf[di + 3] = p[0] + z[0] * 0.5;
      this.dirBuf[di + 4] = p[1] + 0.01 + z[1] * 0.5;
      this.dirBuf[di + 5] = p[2] + z[2] * 0.5;

      const vi = i * 6;
      this.velBuf[vi] = p[0];
      this.velBuf[vi + 1] = p[1] + 0.01;
      this.velBuf[vi + 2] = p[2];
      this.velBuf[vi + 3] = p[0] + v[0] * 0.2;
      this.velBuf[vi + 4] = p[1] + 0.01 + v[1] * 0.2;
      this.velBuf[vi + 5] = p[2] + v[2] * 0.2;
    }
    this.posGeom.attributes.position.needsUpdate = true;
    this.dirGeom.attributes.position.needsUpdate = true;
    this.velGeom.attributes.position.needsUpdate = true;
  }
}

/** 16 ghost skeletons — one per predicted frame — drawn as parent→child line
 *  segments sharing a single buffer. */
class SequenceGhost {
  readonly group = new THREE.Group();
  private geom = new THREE.BufferGeometry();
  private buf: Float32Array;
  private readonly segsTotal: number;

  constructor(private boneCount: number, parents: Int16Array, color: number) {
    // One segment per bone-with-parent, per frame.
    let n = 0;
    for (let i = 0; i < boneCount; i++) if (parents[i] >= 0) n++;
    this.segsTotal = n * SEQ;
    this.buf = new Float32Array(this.segsTotal * 2 * 3);
    this.geom.setAttribute("position", new THREE.BufferAttribute(this.buf, 3));

    this.group.add(new THREE.LineSegments(this.geom,
      new THREE.LineBasicMaterial({
        color, transparent: true, opacity: 0.35,
        depthTest: false, depthWrite: false,
      })));
    this.group.renderOrder = 9;
  }

  update(seq: Sequence, parents: Int16Array): void {
    let w = 0;
    for (let f = 0; f < SEQ; f++) {
      const pos = seq.bonePositions[f];
      for (let i = 0; i < this.boneCount; i++) {
        const p = parents[i];
        if (p < 0) continue;
        const a = pos[p];
        const b = pos[i];
        this.buf[w++] = a[0]; this.buf[w++] = a[1]; this.buf[w++] = a[2];
        this.buf[w++] = b[0]; this.buf[w++] = b[1]; this.buf[w++] = b[2];
      }
    }
    this.geom.attributes.position.needsUpdate = true;
  }
}

/** Guidance skeleton — magenta joint points + parent→child lines, matches
 *  Python's Guidance.Draw which renders the full skeleton in magenta at the
 *  target joint positions. */
class GuidanceViz {
  readonly group = new THREE.Group();
  private pointsBuf: Float32Array;
  private linesBuf: Float32Array;
  private pointsGeom = new THREE.BufferGeometry();
  private linesGeom = new THREE.BufferGeometry();
  private readonly boneCount: number;
  private readonly parents: Int16Array;

  constructor(boneCount: number, parents: Int16Array) {
    this.boneCount = boneCount;
    this.parents = parents;
    this.pointsBuf = new Float32Array(boneCount * 3);
    let segs = 0;
    for (let i = 0; i < boneCount; i++) if (parents[i] >= 0) segs++;
    this.linesBuf = new Float32Array(segs * 2 * 3);

    this.pointsGeom.setAttribute("position", new THREE.BufferAttribute(this.pointsBuf, 3));
    this.linesGeom.setAttribute("position", new THREE.BufferAttribute(this.linesBuf, 3));

    const points = new THREE.Points(this.pointsGeom,
      new THREE.PointsMaterial({ color: 0xff00ff, size: 0.05, sizeAttenuation: true,
        depthTest: false, transparent: true, opacity: 0.95 }));
    const lines = new THREE.LineSegments(this.linesGeom,
      new THREE.LineBasicMaterial({ color: 0xff00ff, transparent: true, opacity: 0.6,
        depthTest: false }));
    points.renderOrder = 11;
    lines.renderOrder = 11;
    this.group.add(points);
    this.group.add(lines);
  }

  update(rootLocal: [number, number, number][], actorRoot: M.Mat4): void {
    const world: [number, number, number][] = [];
    for (let i = 0; i < this.boneCount; i++) {
      const w = M.transformPoint(actorRoot, rootLocal[i]);
      world.push(w);
      this.pointsBuf[i * 3] = w[0];
      this.pointsBuf[i * 3 + 1] = w[1];
      this.pointsBuf[i * 3 + 2] = w[2];
    }
    let k = 0;
    for (let i = 0; i < this.boneCount; i++) {
      const p = this.parents[i];
      if (p < 0) continue;
      const a = world[p], b = world[i];
      this.linesBuf[k++] = a[0]; this.linesBuf[k++] = a[1]; this.linesBuf[k++] = a[2];
      this.linesBuf[k++] = b[0]; this.linesBuf[k++] = b[1]; this.linesBuf[k++] = b[2];
    }
    this.pointsGeom.attributes.position.needsUpdate = true;
    this.linesGeom.attributes.position.needsUpdate = true;
  }
}

export class Debug {
  readonly simulation: TrajectoryViz;
  readonly rootControl: TrajectoryViz;
  readonly prevSeq: SequenceGhost;
  readonly curSeq: SequenceGhost;
  readonly guidance: GuidanceViz;

  /** Per-layer visibility (toggled by the user). SimulationObject is on by
   *  default to match Python's unconditional draw. */
  private toggles = {
    simulation: true,
    rootControl: false,
    prevSeq: false,
    curSeq: false,
    guidance: false,
  };

  private readonly scene: THREE.Scene;

  constructor(scene: THREE.Scene, private actor: Actor) {
    this.scene = scene;
    // Colors follow Python: position=black, direction=orange, velocity=green.
    this.simulation = new TrajectoryViz(0x000000, 0xff9000, 0x22cc44);
    // RootControl gets a distinct blue-ish to tell it apart from simulation.
    this.rootControl = new TrajectoryViz(0x0080ff, 0xff9000, 0x22cc44);
    this.prevSeq = new SequenceGhost(actor.boneCount, actor.parents, 0xff3030);
    this.curSeq = new SequenceGhost(actor.boneCount, actor.parents, 0x30ff60);
    this.guidance = new GuidanceViz(actor.boneCount, actor.parents);

    scene.add(this.simulation.group);
    scene.add(this.rootControl.group);
    scene.add(this.prevSeq.group);
    scene.add(this.curSeq.group);
    scene.add(this.guidance.group);

    this.applyVisibility();
  }

  /** Remove every debug overlay from the scene. Call on session teardown. */
  dispose(): void {
    this.scene.remove(this.simulation.group);
    this.scene.remove(this.rootControl.group);
    this.scene.remove(this.prevSeq.group);
    this.scene.remove(this.curSeq.group);
    this.scene.remove(this.guidance.group);
  }

  toggle(key: keyof Debug["toggles"]): boolean {
    this.toggles[key] = !this.toggles[key];
    this.applyVisibility();
    return this.toggles[key];
  }

  getStates() { return { ...this.toggles }; }

  private applyVisibility(): void {
    this.simulation.group.visible = this.toggles.simulation;
    this.rootControl.group.visible = this.toggles.rootControl;
    this.prevSeq.group.visible = this.toggles.prevSeq;
    this.curSeq.group.visible = this.toggles.curSeq;
    this.guidance.group.visible = this.toggles.guidance;
  }

  updateSimulation(rs: RootSeries): void {
    if (this.toggles.simulation) this.simulation.update(rs);
  }
  updateRootControl(rs: RootSeries): void {
    if (this.toggles.rootControl) this.rootControl.update(rs);
  }
  updatePrevSeq(seq: Sequence): void {
    if (this.toggles.prevSeq) this.prevSeq.update(seq, this.actor.parents);
  }
  updateCurSeq(seq: Sequence): void {
    if (this.toggles.curSeq) this.curSeq.update(seq, this.actor.parents);
  }
  updateGuidance(rootLocal: [number, number, number][]): void {
    if (this.toggles.guidance) this.guidance.update(rootLocal, this.actor.root);
  }
}
