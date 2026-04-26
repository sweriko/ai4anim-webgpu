/** Tweakpane-driven UI + stats-gl profiler panels.
 *
 *  Exposes buttons for spawning / clearing characters, a "Steer" dropdown for
 *  picking which agent is WASD-driven, and live overlays (contacts, timing,
 *  debug-layer toggles). The main loop updates `state.*` every frame and calls
 *  `pane.refresh()` at a throttled rate.
 */

import Stats from "stats-gl";
import { Pane } from "tweakpane";
import type { BladeApi, ButtonApi, BindingApi, FolderApi } from "@tweakpane/core";
import type { WebGPURenderer } from "three/webgpu";
import type { Debug } from "./debug.js";
import type { ModelKind } from "../model/bundle.js";
import type { StudioBulbParams, StudioTransform } from "./studiolights.js";

export interface DirectionalLightParams {
  /** Hex color string. */
  color: string;
  intensity: number;
  /** Offset from the followed target (determines sun angle). */
  offsetX: number; offsetY: number; offsetZ: number;
  castShadow: boolean;
}

export interface UIState {
  steer: string;
  steerKind: ModelKind | null;
  gaitState: string;
  /** User-picked style for the biped dropdown — distinct from gaitState, which
   *  flips to "Idle" automatically when the biped is stationary. Quadruped
   *  doesn't use this (its guidance is speed-driven). */
  playerStyle: string;
  speed: number;
  simulation: boolean;
  rootControl: boolean;
  prevSeq: boolean;
  curSeq: boolean;
  guidance: boolean;
  timescale: number;
  synchronization: number;
  blend: number;
  prevT: number;
  curT: number;
  contacts: [number, number, number, number];
}

export interface UIOptions {
  renderer: WebGPURenderer;
  onSpawnBiped: () => void;
  onSpawnDog: () => void;
  onClearAll: () => void;
  onSteerChange: (id: string) => void;
  onStyleChange: (style: string) => void;
  onResetCamera: () => void;
  /** Studio-light authoring hooks. Both fire on any control change so the
   *  caller can re-derive instance / light matrices in one place. */
  studio?: {
    bulb: StudioBulbParams;
    transforms: [StudioTransform, StudioTransform];
    onBulbChange: (p: StudioBulbParams) => void;
    onTransformChange: (idx: 0 | 1, t: StudioTransform) => void;
  };
  /** Directional key-light controls (color, intensity, sun angle, shadows). */
  directional?: {
    params: DirectionalLightParams;
    onChange: (p: DirectionalLightParams) => void;
  };
}

export class UI {
  readonly state: UIState;
  readonly pane: Pane;
  readonly stats: Stats;
  /** True when the primary pointer is coarse (touch). Stats panel and
   *  Tweakpane are hidden via CSS in this case, so per-frame refresh /
   *  panel updates are skipped to avoid useless work. */
  readonly isTouch: boolean;

  private triPanel: ReturnType<Stats["addPanel"]>;
  private callPanel: ReturnType<Stats["addPanel"]>;
  private inferPanel: ReturnType<Stats["addPanel"]>;
  private maxTris = 1;
  private maxCalls = 1;
  private maxInferMs = 1;
  private lastInferMs = 0;

  private renderer: WebGPURenderer;
  private debug!: Debug;

  // Dynamic bindings we rebuild as state changes.
  private controlFolder: FolderApi;
  private steerBinding: BindingApi<string> | null = null;
  private contactsFolder: FolderApi;
  private runFolder: FolderApi;
  private gaitBlade: { dispose(): void } | null = null;

  private readonly onSteerChange: (id: string) => void;
  private readonly onStyleChange: (style: string) => void;

  constructor(opts: UIOptions) {
    this.renderer = opts.renderer;
    this.onSteerChange = opts.onSteerChange;
    this.onStyleChange = opts.onStyleChange;
    this.isTouch = typeof matchMedia === "function"
      && matchMedia("(pointer: coarse)").matches;

    this.state = {
      steer: "none",
      steerKind: null,
      gaitState: "Idle",
      playerStyle: "Neutral",
      speed: 0,
      simulation: true,
      rootControl: false,
      prevSeq: false,
      curSeq: false,
      guidance: false,
      timescale: 1.0,
      synchronization: 0.0,
      blend: 0.0,
      prevT: 0.0,
      curT: 0.0,
      contacts: [0, 0, 0, 0],
    };

    // --- stats-gl ---
    this.stats = new Stats({ trackGPU: true, horizontal: false });
    this.stats.init(this.renderer);
    document.body.appendChild(this.stats.dom);
    this.stats.dom.classList.add("stats-gl");

    this.triPanel = this.stats.addPanel(new Stats.Panel("TRIS", "#0ff", "#022"));
    this.callPanel = this.stats.addPanel(new Stats.Panel("CALLS", "#f80", "#220"));
    this.inferPanel = this.stats.addPanel(new Stats.Panel("INFER", "#ff6", "#221"));

    // --- Tweakpane ---
    // Pane is hidden via CSS on touch devices, so expansion only matters
    // on desktop — open by default there for discoverability.
    this.pane = new Pane({ title: "AI4Anim WebGPU", expanded: !this.isTouch }) as unknown as Pane;

    // Spawn controls — always first so users find them quickly.
    const spawnFolder = this.pane.addFolder({ title: "Spawn", expanded: true });
    (spawnFolder.addButton({ title: "Spawn Biped" }) as ButtonApi)
      .on("click", () => opts.onSpawnBiped());
    (spawnFolder.addButton({ title: "Spawn Dog" }) as ButtonApi)
      .on("click", () => opts.onSpawnDog());
    (spawnFolder.addButton({ title: "Clear All" }) as ButtonApi)
      .on("click", () => opts.onClearAll());

    // Control — steer dropdown lives here, rebuilt whenever the roster changes.
    this.controlFolder = this.pane.addFolder({ title: "Control", expanded: true });
    this.rebuildSteerDropdown([{ id: "none", label: "(no agents)" }]);

    // Debug layer toggles.
    const layerFolder = this.pane.addFolder({ title: "Debug Layers", expanded: true });
    type LayerKey = "simulation" | "rootControl" | "prevSeq" | "curSeq" | "guidance";
    const addLayerToggle = (stateKey: LayerKey, label: string) => {
      (layerFolder.addBinding(this.state, stateKey, { label }) as BindingApi<boolean>)
        .on("change", (ev) => {
          if (!this.debug) return;
          const cur = this.debug.getStates()[stateKey];
          if (cur !== ev.value) this.debug.toggle(stateKey);
        });
    };
    addLayerToggle("simulation", "Simulation  [1]");
    addLayerToggle("rootControl", "Root Ctrl   [2]");
    addLayerToggle("prevSeq", "Previous    [3]");
    addLayerToggle("curSeq", "Current     [4]");
    addLayerToggle("guidance", "Guidance    [5]");

    // Runtime stats — live tracked on whoever the player is. Gait/style at
    // index 0 is rebuilt by `rebuildGaitControl` per player kind.
    this.runFolder = this.pane.addFolder({ title: "Runtime (Player)", expanded: true });
    this.rebuildGaitControl(null, []);
    (this.runFolder.addBinding(this.state, "speed", {
      readonly: true, view: "graph", min: 0, max: 4, label: "speed",
    }) as unknown as BladeApi);
    (this.runFolder.addBinding(this.state, "timescale", {
      readonly: true, view: "graph", min: 1.0, max: 1.5,
    }) as unknown as BladeApi);
    (this.runFolder.addBinding(this.state, "synchronization", {
      label: "sync", readonly: true, view: "graph", min: 0, max: 1,
    }) as unknown as BladeApi);
    (this.runFolder.addBinding(this.state, "blend", {
      readonly: true, view: "graph", min: 0, max: 1,
    }) as unknown as BladeApi);

    // Contacts — rebuilt when the player's kind changes.
    this.contactsFolder = this.pane.addFolder({ title: "Contacts (Player)", expanded: true });
    this.rebuildContactsFolder(["c0", "c1", "c2", "c3"]);

    // Camera helpers.
    const camFolder = this.pane.addFolder({ title: "Camera" });
    (camFolder.addButton({ title: "Reset Camera" }) as ButtonApi)
      .on("click", () => opts.onResetCamera());

    if (opts.directional) this.buildDirectionalFolder(opts.directional);
    if (opts.studio) this.buildStudioFolder(opts.studio);
  }

  private buildDirectionalFolder(d: NonNullable<UIOptions["directional"]>): void {
    const folder = this.pane.addFolder({ title: "Directional Light", expanded: false });
    const fire = () => d.onChange(d.params);
    (folder.addBinding(d.params, "color") as BindingApi).on("change", fire);
    (folder.addBinding(d.params, "intensity",
      { min: 0, max: 10, step: 0.05 }) as BindingApi).on("change", fire);
    (folder.addBinding(d.params, "castShadow",
      { label: "cast shadow" }) as BindingApi).on("change", fire);
    const off = folder.addFolder({ title: "Sun offset (from player)", expanded: false });
    for (const k of ["offsetX", "offsetY", "offsetZ"] as const) {
      (off.addBinding(d.params, k,
        { min: -30, max: 30, step: 0.1 }) as BindingApi).on("change", fire);
    }
  }

  /** Studio-light authoring controls. Bulb params shared between both
   *  fixtures (modelling one physical bulb spec); transforms per-fixture. */
  private buildStudioFolder(s: NonNullable<UIOptions["studio"]>): void {
    const folder = this.pane.addFolder({ title: "Studio Lights", expanded: false });

    const bulb = folder.addFolder({ title: "Bulb (shared)", expanded: true });
    const fireBulb = () => s.onBulbChange(s.bulb);
    (bulb.addBinding(s.bulb, "color") as BindingApi).on("change", fireBulb);
    (bulb.addBinding(s.bulb, "intensity",
      { min: 0, max: 50, step: 0.1 }) as BindingApi).on("change", fireBulb);
    (bulb.addBinding(s.bulb, "width",
      { min: 0.05, max: 5, step: 0.01 }) as BindingApi).on("change", fireBulb);
    (bulb.addBinding(s.bulb, "height",
      { min: 0.05, max: 5, step: 0.01 }) as BindingApi).on("change", fireBulb);
    const offset = bulb.addFolder({ title: "Bulb offset (model space)", expanded: false });
    for (const k of ["offsetX", "offsetY", "offsetZ"] as const) {
      (offset.addBinding(s.bulb, k,
        { min: -3, max: 3, step: 0.01 }) as BindingApi).on("change", fireBulb);
    }
    const rot = bulb.addFolder({ title: "Bulb rotation (rad)", expanded: false });
    // rotX gets headroom below -π so the default (~-π, facing down) isn't
    // pinned to the slider's minimum.
    (rot.addBinding(s.bulb, "rotX",
      { min: -2 * Math.PI, max: Math.PI, step: 0.01 }) as BindingApi).on("change", fireBulb);
    for (const k of ["rotY", "rotZ"] as const) {
      (rot.addBinding(s.bulb, k,
        { min: -Math.PI, max: Math.PI, step: 0.01 }) as BindingApi).on("change", fireBulb);
    }
    (bulb.addBinding(s.bulb, "showHelper",
      { label: "show helper" }) as BindingApi).on("change", fireBulb);

    const buildXform = (idx: 0 | 1) => {
      const t = s.transforms[idx];
      const f = folder.addFolder({ title: `Light #${idx + 1}`, expanded: false });
      const fire = () => s.onTransformChange(idx, t);
      (f.addBinding(t, "x", { min: -20, max: 20, step: 0.01 }) as BindingApi).on("change", fire);
      (f.addBinding(t, "y", { min: -2,  max: 10, step: 0.01 }) as BindingApi).on("change", fire);
      (f.addBinding(t, "z", { min: -20, max: 20, step: 0.01 }) as BindingApi).on("change", fire);
      (f.addBinding(t, "rotY",
        { label: "rot Y", min: -Math.PI, max: Math.PI, step: 0.01 }) as BindingApi)
        .on("change", fire);
      (f.addBinding(t, "scale",
        { min: 0.05, max: 5, step: 0.01 }) as BindingApi).on("change", fire);
    };
    buildXform(0);
    buildXform(1);
  }

  /** Replace the current Steer dropdown with one reflecting the given roster.
   *  `select` is the id that should be the currently-displayed choice. */
  rebuildSteerDropdown(options: { id: string; label: string }[], select?: string): void {
    if (this.steerBinding) {
      this.steerBinding.dispose();
      this.steerBinding = null;
    }
    if (options.length === 0) options = [{ id: "none", label: "(no agents)" }];
    const map: Record<string, string> = {};
    for (const o of options) map[o.label] = o.id;
    if (select && options.some((o) => o.id === select)) {
      this.state.steer = select;
    } else if (!options.some((o) => o.id === this.state.steer)) {
      this.state.steer = options[0].id;
    }
    this.steerBinding = this.controlFolder.addBinding(this.state, "steer", {
      options: map, label: "Steer",
    }) as BindingApi<string>;
    this.steerBinding.on("change", (ev) => this.onSteerChange(String(ev.value)));
  }

  /** Rebuild the gait/style blade at the top of the Runtime folder.
   *
   *    biped:     dropdown of guidance names. Selection calls onStyleChange,
   *               which routes to NMMAgent.setStyle on the player.
   *    quadruped: readonly text. The quadruped's guidance is speed-driven
   *               inside NMMAgent.updateControl, so a picker would be a
   *               no-op — show what the network is actually using instead.
   *    null:      readonly em-dash placeholder.
   */
  rebuildGaitControl(
    kind: ModelKind | null, styles: readonly string[], current?: string,
  ): void {
    if (this.gaitBlade) {
      this.gaitBlade.dispose();
      this.gaitBlade = null;
    }
    if (kind === "biped" && styles.length > 0) {
      const map: Record<string, string> = {};
      for (const s of styles) map[s] = s;
      this.state.playerStyle = current && styles.includes(current)
        ? current
        : (styles.includes("Neutral") ? "Neutral" : styles[0]);
      const binding = this.runFolder.addBinding(this.state, "playerStyle", {
        label: "style", options: map, index: 0,
      }) as BindingApi<string>;
      binding.on("change", (ev) => this.onStyleChange(String(ev.value)));
      this.gaitBlade = binding;
    } else {
      // Quadruped (speed-driven) and "no player" both render readonly text.
      this.gaitBlade = this.runFolder.addBinding(this.state, "gaitState", {
        readonly: true, label: kind === "quadruped" ? "gait" : "gait/style",
        index: 0,
      }) as unknown as { dispose(): void };
    }
  }

  /** Rebuild the 4 contact bars with labels that match the player's kind. */
  rebuildContactsFolder(labels: [string, string, string, string] | string[]): void {
    // Remove every binding in the folder.
    const children = [...(this.contactsFolder as unknown as { children: unknown[] }).children];
    for (const c of children) (c as { dispose: () => void }).dispose();
    for (let i = 0; i < 4; i++) {
      (this.contactsFolder.addBinding(this.state.contacts, `${i}` as "0" | "1" | "2" | "3", {
        label: labels[i] ?? `c${i}`, readonly: true, view: "graph", min: 0, max: 1,
      }) as unknown as BladeApi);
    }
  }

  linkDebug(debug: Debug): void {
    this.debug = debug;
    const s = debug.getStates();
    this.state.simulation = s.simulation;
    this.state.rootControl = s.rootControl;
    this.state.prevSeq = s.prevSeq;
    this.state.curSeq = s.curSeq;
    this.state.guidance = s.guidance;
    this.pane.refresh();
  }

  syncDebugStates(): void {
    if (!this.debug) return;
    const s = this.debug.getStates();
    this.state.simulation = s.simulation;
    this.state.rootControl = s.rootControl;
    this.state.prevSeq = s.prevSeq;
    this.state.curSeq = s.curSeq;
    this.state.guidance = s.guidance;
    this.pane.refresh();
  }

  tickStats(): void {
    if (this.isTouch) return;
    // info.render.calls is cumulative since app start; drawCalls is the
    // per-frame value (reset by info.reset() inside renderer.render()).
    const info = this.renderer.info.render;
    this.maxTris = Math.max(this.maxTris, info.triangles);
    this.maxCalls = Math.max(this.maxCalls, info.drawCalls);

    this.triPanel.update(info.triangles, this.maxTris * 1.2, 0);
    this.triPanel.updateGraph(info.triangles, this.maxTris * 1.2);

    this.callPanel.update(info.drawCalls, this.maxCalls * 1.2, 0);
    this.callPanel.updateGraph(info.drawCalls, this.maxCalls * 1.2);

    const inferCap = Math.max(this.maxInferMs * 1.2, 5);
    this.inferPanel.update(this.lastInferMs, inferCap, 2);
    this.inferPanel.updateGraph(this.lastInferMs, inferCap);

    this.stats.update();
  }

  tickInference(ms: number): void {
    this.lastInferMs = ms;
    this.maxInferMs = Math.max(this.maxInferMs, ms);
  }

  dispose(): void {
    this.pane.dispose();
    this.stats.dom.remove();
  }
}
