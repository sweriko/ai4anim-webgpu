/** Demo driver for the NMMEngine — mixed biped + quadruped scene.
 *
 *  Two engines run in parallel, each with its own instanced SkinnedMesh.
 *  One agent at a time is the "player": WASD-driven, camera follows. Other
 *  agents run on autopilot brains. Tab (or the on-screen Switch button)
 *  cycles which agent is the player.
 *
 *  Controls:
 *    Biped player     — WASD move, Shift sprint, LMB-drag facing, Q/E style
 *    Quadruped player — WASD move, Alt/Ctrl/Shift gait, R/T/V sit/stand/lie
 *    Debug layers     — 1..5 toggles
 *    Tab              — cycle player through alive agents
 */
import * as THREE from "three";
import { NMMEngine } from "./engine/index.js";
import type { NMMAgent } from "./engine/index.js";
import { makeScene } from "./render/scene.js";
import { Input } from "./input.js";
import { Touch } from "./touch.js";
import { Debug } from "./render/debug.js";
import { UI } from "./render/ui.js";
import type { DirectionalLightParams } from "./render/ui.js";
import { StudioLights, DEFAULT_BULB, DEFAULT_TRANSFORMS } from "./render/studiolights.js";
import * as V from "./math/vec3.js";
import { isQuadruped } from "./model/bundle.js";
import type { ModelKind } from "./model/bundle.js";
import type { Vec3 } from "./math/vec3.js";
import { BipedBrain, QuadrupedBrain } from "./runtime/autopilot.js";
import type { Brain } from "./runtime/autopilot.js";

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const bootEl = document.getElementById("boot") as HTMLDivElement;
const setBoot = (m: string) => { if (bootEl) { bootEl.textContent = m; bootEl.style.display = ""; } };
const hideBoot = () => { if (bootEl) bootEl.style.display = "none"; };

const MAX_AGENTS_PER_KIND = 32;
const PLAYFIELD_RADIUS = 12;
const LOCOMOTION_SPEED = {
  walk: 0.7, pace: 1.2, trot: 2.0, canter: 4.0,
} as const;

interface AgentRecord {
  id: string;              // stable — used as the steer-dropdown key
  label: string;           // "Biped #1", "Dog #3"
  kind: ModelKind;
  agent: NMMAgent;
  brain: Brain | null;     // null iff this agent is the player
}

async function boot() {
  setBoot("initializing WebGPU…");
  const { renderer, scene, camera, controls, key: shadowLight } = await makeScene(canvas);

  const directionalParams: DirectionalLightParams = {
    color: "#ffffff",
    intensity: 2.5,
    offsetX: 6, offsetY: 10, offsetZ: 4,
    castShadow: true,
  };
  const applyDirectional = (p: DirectionalLightParams) => {
    shadowLight.color.set(p.color);
    shadowLight.intensity = p.intensity;
    shadowLight.castShadow = p.castShadow;
  };
  applyDirectional(directionalParams);

  const params = new URLSearchParams(location.search);
  const precision = (params.get("precision") === "fp32" ? "fp32" : "fp16") as "fp32" | "fp16";

  setBoot("loading biped + quadruped bundles + meshes…");
  const [bipedEngine, quadrupedEngine] = await Promise.all([
    NMMEngine.load({
      renderer, bundleBaseUrl: "/", characterGlbUrl: "/assets/geno.glb",
      maxAgents: MAX_AGENTS_PER_KIND, precision, bundleKind: "biped",
    }),
    NMMEngine.load({
      renderer, bundleBaseUrl: "/", characterGlbUrl: "/assets/dog.glb",
      maxAgents: MAX_AGENTS_PER_KIND, precision, bundleKind: "quadruped",
    }),
  ]);

  scene.add(bipedEngine.mesh);
  scene.add(quadrupedEngine.mesh);

  // Studio lights — two GLB instances drawn as one InstancedMesh per source
  // mesh, each fitted with its own RectAreaLight whose pose is derived from
  // the studiolight transform + a shared bulb-relative offset/rotation.
  const studioLights = await StudioLights.load("/assets/studiolight.glb");
  scene.add(studioLights.group);
  const studioState = {
    bulb: { ...DEFAULT_BULB },
    transforms: [
      { ...DEFAULT_TRANSFORMS[0] },
      { ...DEFAULT_TRANSFORMS[1] },
    ] as [typeof DEFAULT_TRANSFORMS[0], typeof DEFAULT_TRANSFORMS[1]],
  };
  studioLights.setBulbParams(studioState.bulb);
  studioLights.setStudioTransform(0, studioState.transforms[0]);
  studioLights.setStudioTransform(1, studioState.transforms[1]);

  const agents = new Map<string, AgentRecord>();
  const spawnCounters = { biped: 0, quadruped: 0 };
  let playerId: string | null = null;
  let debug: Debug | null = null;
  let lastPlayerKind: ModelKind | null = null;

  const touch = new Touch({ onSwitch: () => cyclePlayer() });
  const input = new Input(canvas, touch);

  // WASD keycap hint — clicking a key behaves like holding the matching
  // keyboard key (writes into input.keys). The canvas is a sibling, so
  // clicks don't bubble to its LMB-facing handler. setPointerCapture keeps
  // the press alive if the user drags off the key before releasing.
  const wasdKeys = ["w", "a", "s", "d"] as const;
  const wasdEls: Record<string, HTMLElement | null> = {};
  for (const k of wasdKeys) {
    const el = document.querySelector<HTMLElement>(`#wasd-hint [data-k='${k}']`);
    wasdEls[k] = el;
    if (!el) continue;
    el.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      input.keys.add(k);
      el.classList.add("pressed");
    });
    const release = () => { input.keys.delete(k); el.classList.remove("pressed"); };
    el.addEventListener("pointerup", release);
    el.addEventListener("pointercancel", release);
  }
  window.addEventListener("keydown", (e) => {
    wasdEls[e.key.toLowerCase()]?.classList.add("pressed");
  });
  window.addEventListener("keyup", (e) => {
    wasdEls[e.key.toLowerCase()]?.classList.remove("pressed");
  });

  // Credit button + popup. Toggle on click; close on outside click or Esc.
  const creditBtn = document.getElementById("credit-btn");
  const creditPopup = document.getElementById("credit-popup");
  if (creditBtn && creditPopup) {
    creditBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      creditPopup.classList.toggle("hidden");
    });
    document.addEventListener("click", (e) => {
      if (creditPopup.classList.contains("hidden")) return;
      const t = e.target;
      if (t instanceof Node && (creditPopup.contains(t) || creditBtn.contains(t))) return;
      creditPopup.classList.add("hidden");
    });
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") creditPopup.classList.add("hidden");
    });
  }

  // --- Camera follow ----------------------------------------------------
  const initialCameraPos = camera.position.clone();
  const initialTarget = controls.target.clone();
  function cameraFollow() {
    const player = playerId ? agents.get(playerId) : null;
    if (!player) return;
    const target = player.agent.getPosition();
    const headY = player.kind === "biped" ? 1.2 : 0.5;
    const desired = new THREE.Vector3(target[0], target[1] + headY, target[2]);
    const prev = controls.target;
    const dx = desired.x - prev.x, dy = desired.y - prev.y, dz = desired.z - prev.z;
    prev.x += dx; prev.y += dy; prev.z += dz;
    camera.position.x += dx; camera.position.y += dy; camera.position.z += dz;
    controls.update();
    // Re-anchor the directional rig so its tight shadow box stays around the
    // player. Offset is authored via the tweakpane directional folder.
    shadowLight.position.set(
      target[0] + directionalParams.offsetX,
      target[1] + directionalParams.offsetY,
      target[2] + directionalParams.offsetZ);
    shadowLight.target.position.set(target[0], target[1], target[2]);
    shadowLight.target.updateMatrixWorld();
  }

  function rosterOptions() {
    return Array.from(agents.values()).map((r) => ({ id: r.id, label: r.label }));
  }

  function contactLabelsFor(kind: ModelKind): [string, string, string, string] {
    // Bundle has skeleton.contact_labels; not in bundle types' common base so cast.
    const engine = kind === "biped" ? bipedEngine : quadrupedEngine;
    const labels = (engine.bundle.meta.skeleton as unknown as { contact_labels: string[] })
      .contact_labels;
    return [labels[0], labels[1], labels[2], labels[3]];
  }

  /** Canonical default style for a kind — what the player should spawn with
   *  so its motion is predictable, not randomized. */
  function defaultStyleFor(kind: ModelKind): string {
    const engine = kind === "biped" ? bipedEngine : quadrupedEngine;
    const styles = engine.styles;
    // Biped's "Neutral" is the plain locomotion style. Quadruped guidance is
    // speed-driven so any starting value is overwritten in updateControl —
    // pick the first guidance to keep things deterministic.
    if (kind === "biped" && styles.includes("Neutral")) return "Neutral";
    return styles[0];
  }

  // --- UI ---------------------------------------------------------------
  const ui = new UI({
    renderer,
    onSpawnBiped: () => spawn("biped"),
    onSpawnDog: () => spawn("quadruped"),
    onClearAll: () => clearAll(),
    onSteerChange: (id) => setPlayer(id === "none" ? null : id),
    onStyleChange: (style) => {
      const player = playerId ? agents.get(playerId) : null;
      if (player) player.agent.setStyle(style);
    },
    onResetCamera: () => {
      const player = playerId ? agents.get(playerId) : null;
      const base = player ? player.agent.getPosition() : [0, 0, 0];
      camera.position.set(
        base[0] + initialCameraPos.x,
        base[1] + initialCameraPos.y,
        base[2] + initialCameraPos.z);
      controls.target.set(
        base[0] + initialTarget.x,
        base[1] + initialTarget.y,
        base[2] + initialTarget.z);
      controls.update();
    },
    studio: {
      bulb: studioState.bulb,
      transforms: studioState.transforms,
      onBulbChange: (p) => studioLights.setBulbParams(p),
      onTransformChange: (idx, t) => studioLights.setStudioTransform(idx, t),
    },
    directional: {
      params: directionalParams,
      onChange: (p) => applyDirectional(p),
    },
  });

  // --- Debug-layer keyboard shortcuts ----------------------------------
  // Listen on window — canvas focus isn't reliable when overlay UIs eat clicks.
  window.addEventListener("keydown", (e) => {
    const t = e.target;
    if (t instanceof HTMLElement && (t.tagName === "INPUT" || t.tagName === "TEXTAREA"
      || t.tagName === "SELECT" || t.isContentEditable)) return;
    const k = e.key.toLowerCase();
    if (!debug) return;
    if (k === "1") { debug.toggle("simulation"); ui.syncDebugStates(); }
    else if (k === "2") { debug.toggle("rootControl"); ui.syncDebugStates(); }
    else if (k === "3") { debug.toggle("prevSeq"); ui.syncDebugStates(); }
    else if (k === "4") { debug.toggle("curSeq"); ui.syncDebugStates(); }
    else if (k === "5") { debug.toggle("guidance"); ui.syncDebugStates(); }
    else if (k === "tab") { e.preventDefault(); cyclePlayer(); }
  });

  // --- Spawn / clear ---------------------------------------------------
  function spawnPosition(): { pos: Vec3; facing: Vec3 } {
    // Bias the first few spawns close to origin, spread the rest farther out.
    const count = agents.size;
    const radius = Math.min(PLAYFIELD_RADIUS, 1.5 + count * 0.6);
    const a = Math.random() * Math.PI * 2;
    const r = Math.sqrt(Math.random()) * radius;
    return {
      pos: [Math.cos(a) * r, 0, Math.sin(a) * r],
      facing: [-Math.cos(a), 0, -Math.sin(a)],
    };
  }

  function spawn(kind: ModelKind): void {
    const engine = kind === "biped" ? bipedEngine : quadrupedEngine;
    if (engine.agentCount >= MAX_AGENTS_PER_KIND) {
      console.warn(`[spawn] ${kind} at cap (${MAX_AGENTS_PER_KIND}) — ignoring`);
      return;
    }
    const { pos, facing } = spawnPosition();

    // Pick a starting style. The first agent (the soon-to-be player) gets
    // the canonical default — "Neutral" for biped, otherwise the bundle's
    // first guidance — so the controlled character is in a known, stable
    // state. Subsequent NPC spawns randomize for crowd variety; the
    // autopilot brain re-rolls their style every 8–30s anyway.
    let style: string;
    const willBecomePlayer = playerId === null;
    if (kind === "biped") {
      if (willBecomePlayer) {
        style = defaultStyleFor(kind);
      } else {
        const styles = engine.styles.filter((s) => s !== "Idle");
        style = styles.length > 0
          ? styles[Math.floor(Math.random() * styles.length)]
          : engine.styles[0];
      }
    } else {
      style = defaultStyleFor(kind);
    }

    const agent = engine.createAgent({ position: pos, facing, style });

    const counter = kind === "biped"
      ? ++spawnCounters.biped : ++spawnCounters.quadruped;
    const id = `${kind}-${counter}`;
    const label = kind === "biped" ? `Biped #${counter}` : `Dog #${counter}`;

    // Autopilot brain by default — becomes the player only if nothing else is.
    const brain = makeBrain(kind, agent);

    const record: AgentRecord = { id, label, kind, agent, brain };
    agents.set(id, record);

    // Refresh UI dropdown.
    const opts = rosterOptions();
    ui.rebuildSteerDropdown(opts, playerId ?? id);

    if (playerId === null) setPlayer(id);
  }

  function makeBrain(kind: ModelKind, agent: NMMAgent): Brain {
    const seed = ((Math.random() * 2 ** 31) | 0) >>> 0;
    if (kind === "biped") {
      const styles = (agent as unknown as { bundle: { meta: { guidances: string[] } } })
        .bundle?.meta?.guidances ?? bipedEngine.styles;
      return new BipedBrain({
        agent, seed,
        movementStyles: styles.filter((s) => s !== "Idle"),
        playfieldRadius: PLAYFIELD_RADIUS,
      });
    }
    const meta = quadrupedEngine.bundle.meta;
    if (!isQuadruped(meta)) throw new Error("quadruped engine has non-quadruped bundle");
    return new QuadrupedBrain({
      agent, seed,
      speeds: { ...meta.control.locomotion_modes },
      playfieldRadius: PLAYFIELD_RADIUS,
    });
  }

  function clearAll(): void {
    for (const r of agents.values()) {
      const engine = r.kind === "biped" ? bipedEngine : quadrupedEngine;
      engine.removeAgent(r.agent);
    }
    agents.clear();
    setPlayer(null);
    ui.rebuildSteerDropdown([]);
  }

  function setPlayer(id: string | null): void {
    const prev = playerId ? agents.get(playerId) : null;
    const next = id ? agents.get(id) : null;
    if (prev && prev.id !== next?.id) prev.brain = makeBrain(prev.kind, prev.agent);
    if (next) next.brain = null;
    playerId = next?.id ?? null;

    // Retarget debug overlays to the new player's actor (or dispose if none).
    if (debug) { debug.dispose(); debug = null; }
    if (next) {
      debug = new Debug(scene, next.agent.actor);
      ui.linkDebug(debug);
    }

    // Update contact labels if kind changed.
    const newKind = next?.kind ?? null;
    if (newKind !== lastPlayerKind) {
      if (newKind) ui.rebuildContactsFolder(contactLabelsFor(newKind));
      else ui.rebuildContactsFolder(["c0", "c1", "c2", "c3"]);
      lastPlayerKind = newKind;
    }

    // Drives CSS that hides the right (facing) joystick when controlling a
    // quadruped, since it has no facing input.
    document.documentElement.classList.toggle("player-quadruped", newKind === "quadruped");

    // Rebuild the gait/style blade to suit the new player kind: dropdown for
    // biped, readonly text for quadruped (speed-driven), em-dash for none.
    const styles = next
      ? (next.kind === "biped" ? bipedEngine.styles : quadrupedEngine.styles)
      : [];
    ui.rebuildGaitControl(newKind, styles, next?.agent.style);

    ui.state.steer = playerId ?? "none";
    ui.state.steerKind = newKind;
    ui.pane.refresh();
  }

  function cyclePlayer(): void {
    const ids = Array.from(agents.keys());
    if (ids.length === 0) return;
    const cur = playerId ? ids.indexOf(playerId) : -1;
    const next = ids[(cur + 1) % ids.length];
    setPlayer(next);
  }

  // --- Input routing ---------------------------------------------------
  const _camFwd = new THREE.Vector3();
  const _camRight = new THREE.Vector3();
  const _UP = new THREE.Vector3(0, 1, 0);
  function cameraBasis(): { fwd: Vec3; right: Vec3 } {
    camera.getWorldDirection(_camFwd);
    _camFwd.y = 0;
    if (_camFwd.lengthSq() < 1e-6) _camFwd.set(0, 0, 1);
    _camFwd.normalize();
    _camRight.crossVectors(_camFwd, _UP).normalize();
    return { fwd: [_camFwd.x, 0, _camFwd.z], right: [_camRight.x, 0, _camRight.z] };
  }

  function drivePlayer() {
    // Read facing once and mirror it onto the right joystick — so the
    // stick animates along with LMB-drag on the canvas, not just direct
    // touches. fz uses up-positive screen axis; flip back to screen-down
    // for the joystick's pixel-space visualization.
    const [fx, fz] = input.getFacingDelta();
    touch.visualizeRight(fx, -fz, input.facingMouseDown);

    const player = playerId ? agents.get(playerId) : null;
    if (!player) return;
    const [mx, mz] = input.getMovementVector();
    const { fwd, right } = cameraBasis();

    if (player.kind === "biped") {
      const SPEED = input.isSprint() ? 2.0 : 1.0;
      const rawVel: Vec3 = [
        (mz * fwd[0] + mx * right[0]) * SPEED, 0,
        (mz * fwd[2] + mx * right[2]) * SPEED,
      ];
      const vel = V.clampMagnitude(rawVel, SPEED);
      const facing: Vec3 = [
        fz * fwd[0] + fx * right[0], 0,
        fz * fwd[2] + fx * right[2],
      ];
      player.agent.setGoal(vel, facing);
      if (input.consumeStylePrev()) cycleStyle(player, -1);
      if (input.consumeStyleNext()) cycleStyle(player, +1);
    } else {
      const moveLen = Math.hypot(mx, mz);
      const moveDir: Vec3 = moveLen > 1e-5
        ? [(mz * fwd[0] + mx * right[0]) / moveLen, 0, (mz * fwd[2] + mx * right[2]) / moveLen]
        : [0, 0, 0];
      const targetSpeed = moveLen < 0.05 ? 0 : LOCOMOTION_SPEED[input.getQuadrupedGait()];
      const vel: Vec3 = [moveDir[0] * targetSpeed, 0, moveDir[2] * targetSpeed];
      player.agent.setGoal(vel, moveDir);
      player.agent.setAction(input.getQuadrupedAction());
    }
  }

  function cycleStyle(player: AgentRecord, delta: number) {
    const engine = player.kind === "biped" ? bipedEngine : quadrupedEngine;
    const styles = engine.styles.slice().sort();
    if (styles.length === 0) return;
    const cur = styles.indexOf(player.agent.style);
    const idx = ((cur + delta) % styles.length + styles.length) % styles.length;
    player.agent.setStyle(styles[idx]);
  }

  // Default scene: one biped (auto-promoted to player by setPlayer inside
  // spawn) plus one quadruped wandering on autopilot. Mobile users tap
  // "Switch Character" to flip control between them; desktop has Tab + the
  // Steer dropdown.
  spawn("biped");
  spawn("quadruped");

  // --- Main loop --------------------------------------------------------
  hideBoot();
  const timer = new THREE.Timer();
  timer.connect(document);
  let totalTime = 0;
  let lastPaneRefresh = -1;
  let lastReadbackBiped = 0, lastReadbackQuad = 0;
  let lastTsResolve = 0;
  const PANE_REFRESH_DT = 1 / 15;

  function loop() {
    timer.update();
    const dt = Math.min(timer.getDelta(), 0.1);
    totalTime += dt;

    // Autopilot brains drive non-player agents; WASD drives the player.
    for (const r of agents.values()) {
      if (r.brain) r.brain.update(dt, totalTime);
    }
    drivePlayer();

    bipedEngine.update(dt);
    quadrupedEngine.update(dt);

    cameraFollow();
    renderer.render(scene, camera);
    ui.tickStats();

    // Infer latency — take whichever engine just finished.
    if (bipedEngine.lastReadbackMs !== lastReadbackBiped) {
      lastReadbackBiped = bipedEngine.lastReadbackMs;
      ui.tickInference(bipedEngine.lastComputeMs + bipedEngine.lastReadbackMs);
    }
    if (quadrupedEngine.lastReadbackMs !== lastReadbackQuad) {
      lastReadbackQuad = quadrupedEngine.lastReadbackMs;
      ui.tickInference(quadrupedEngine.lastComputeMs + quadrupedEngine.lastReadbackMs);
    }

    // Debug overlays follow the player.
    const player = playerId ? agents.get(playerId) : null;
    if (debug && player) {
      const d = player.agent.debugState;
      debug.updateSimulation(d.simulation);
      debug.updateRootControl(d.rootControl);
      if (d.current) debug.updateCurSeq(d.current);
      if (d.previous) debug.updatePrevSeq(d.previous);
      debug.updateGuidance(d.guidance);

      if (!ui.isTouch && totalTime - lastPaneRefresh >= PANE_REFRESH_DT) {
        lastPaneRefresh = totalTime;
        ui.state.timescale = d.timescale;
        ui.state.synchronization = d.synchronization;
        ui.state.blend = d.blend;
        ui.state.prevT = Math.min(Math.max(d.prevT, 0), 0.5);
        ui.state.curT = Math.min(Math.max(d.curT, 0), 0.5);
        ui.state.gaitState = d.guidanceState;
        // Mirror the agent's user-picked style into the dropdown — keeps the
        // panel in sync if Q/E cycled it. Quadruped doesn't bind to this.
        ui.state.playerStyle = player.agent.style;
        const vel = player.agent.getVelocity();
        ui.state.speed = Math.hypot(vel[0], vel[2]);
        for (let i = 0; i < 4; i++) ui.state.contacts[i] = d.contacts[i];
        ui.pane.refresh();
      }
    }

    if (totalTime - lastTsResolve > 1.0) {
      lastTsResolve = totalTime;
      renderer.resolveTimestampsAsync("render").catch(() => {});
      renderer.resolveTimestampsAsync("compute").catch(() => {});
    }
    requestAnimationFrame(loop);
  }
  loop();
}

boot().catch((err) => {
  console.error(err);
  setBoot(`error: ${err instanceof Error ? err.message : String(err)}`);
});
