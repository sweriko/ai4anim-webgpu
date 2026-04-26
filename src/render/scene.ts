import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export async function makeScene(canvas: HTMLCanvasElement) {
  const renderer = new WebGPURenderer({
    canvas,
    antialias: true,
    trackTimestamp: true,
    // The decoder FiLM+Linear kernel binds 10 storage buffers; the WebGPU
    // default is 8. Every modern adapter advertises ≥ 10, and three will
    // validate the request against `adapter.limits` during init().
    requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
  });
  await renderer.init();
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a1a);
  // Much wider fog — character can walk 100s of meters.
  scene.fog = new THREE.Fog(0x1a1a1a, 30, 200);

  const camera = new THREE.PerspectiveCamera(
    45, window.innerWidth / window.innerHeight, 0.1, 500);
  // Start behind the character (character faces world +Z at bind pose, so
  // "behind" is world -Z).
  camera.position.set(0, 2.5, -5);
  camera.lookAt(0, 1.2, 0);

  // Camera orbit on middle mouse only — LMB is taken by facing-drag and the
  // right joystick, so leaving LEFT/RIGHT null keeps OrbitControls out of
  // the way. Single-finger touch is also disabled (joysticks own it);
  // two-finger pinch dollies + rotates.
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.minDistance = 1.5;
  controls.maxDistance = 20;
  controls.maxPolarAngle = Math.PI * 0.49;   // don't clip through ground
  controls.mouseButtons = {
    LEFT: null as unknown as THREE.MOUSE,
    MIDDLE: THREE.MOUSE.ROTATE,
    RIGHT: null as unknown as THREE.MOUSE,
  };
  controls.touches = {
    ONE: null as unknown as THREE.TOUCH,
    TWO: THREE.TOUCH.DOLLY_ROTATE,
  };
  // Match the camera-follow logic's first-frame expectation (character head).
  controls.target.set(0, 1.2, 0);
  controls.update();

  // Directional key light — shadow caster + sun angle. Studio RectAreaLights
  // (added in main.ts) provide the soft fill on top. No hemisphere ambient.
  const key = new THREE.DirectionalLight(0xffffff, 2.5);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  // Tight local box — main.ts moves this rig to follow the player so shadow
  // resolution stays high without ballooning the orthographic camera.
  key.shadow.camera.left = -10;
  key.shadow.camera.right = 10;
  key.shadow.camera.top = 10;
  key.shadow.camera.bottom = -10;
  key.shadow.camera.near = 0.1;
  key.shadow.camera.far = 40;
  key.shadow.bias = -0.0002;
  key.position.set(6, 10, 4);
  key.target.position.set(0, 0, 0);
  scene.add(key);
  scene.add(key.target);

  // Ground — huge so it never runs out.
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(500, 500),
    new THREE.MeshStandardMaterial({ color: 0x2a2a2e, roughness: 0.95 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);
  const grid = new THREE.GridHelper(500, 500, 0x404040, 0x2f2f2f);
  scene.add(grid);

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight, false);
  });

  return { renderer, scene, camera, controls, key };
}
