/** Two studiolights authored from the same GLB, drawn with one InstancedMesh
 *  per source mesh (count = 2 → halves draw calls vs cloning the GLB twice).
 *  Each studiolight gets its own RectAreaLight whose pose is composed from
 *  the studiolight's world transform and a shared "bulb-relative-to-model"
 *  offset / rotation, so authoring the bulb once snaps both lights to the
 *  bulb of their respective fixture.
 *
 *  The rect-area-light LTC textures must be initialised once per renderer
 *  (`RectAreaLightNode.setLTC`); we do that on first construction. RAL on
 *  WebGPU does not cast shadows — the scene's directional key light keeps
 *  doing that.
 */

import * as THREE from "three";
import { RectAreaLightNode } from "three/webgpu";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RectAreaLightHelper } from "three/addons/helpers/RectAreaLightHelper.js";
import { RectAreaLightTexturesLib } from "three/addons/lights/RectAreaLightTexturesLib.js";

export interface StudioTransform {
  x: number; y: number; z: number;
  /** Yaw around world Y in radians. */
  rotY: number;
  /** Uniform scale applied to the loaded GLB. */
  scale: number;
}

export interface StudioBulbParams {
  /** Hex color string, e.g. "#ffe9c4". */
  color: string;
  intensity: number;
  /** Rect-area-light extents in metres. */
  width: number;
  height: number;
  /** Bulb position in the studiolight's local model space (pre-scale). */
  offsetX: number; offsetY: number; offsetZ: number;
  /** Bulb rotation in the studiolight's local model space (radians). */
  rotX: number; rotY: number; rotZ: number;
  /** Toggle the wireframe RectAreaLightHelper for both bulbs. */
  showHelper: boolean;
}

export const DEFAULT_BULB: StudioBulbParams = {
  color: "#ffe9c4",
  intensity: 50,
  width: 0.85,
  height: 0.79,
  offsetX: 0, offsetY: 1.77, offsetZ: 0.40,
  rotX: -3.15, rotY: 0, rotZ: 0,
  showHelper: true,
};

export const DEFAULT_TRANSFORMS: [StudioTransform, StudioTransform] = [
  { x: -3, y: 0, z:  3, rotY:  Math.PI * 0.75, scale: 1 },
  { x:  3, y: 0, z:  3, rotY: -Math.PI * 0.75, scale: 1 },
];

/** One InstancedMesh per Mesh found in the GLB, plus its baked
 *  gltf-root-relative transform so we can re-compose per studiolight. */
interface PerInstanced {
  mesh: THREE.InstancedMesh;
  /** Mesh's matrix relative to the GLB scene root, baked from the gltf
   *  hierarchy at load time. */
  meshLocal: THREE.Matrix4;
}

export class StudioLights {
  readonly group = new THREE.Group();
  private readonly instanced: PerInstanced[] = [];
  /** One rect light per studiolight (count = 2). Each is parented to the
   *  scene group; its world matrix is set in `refresh`. */
  private readonly lights: THREE.RectAreaLight[] = [];
  private readonly helpers: RectAreaLightHelper[] = [];

  private readonly transforms: [StudioTransform, StudioTransform] = [
    { ...DEFAULT_TRANSFORMS[0] },
    { ...DEFAULT_TRANSFORMS[1] },
  ];
  private bulb: StudioBulbParams = { ...DEFAULT_BULB };

  /** LTC init is process-global on the WebGPU light node; guard it. */
  private static ltcReady = false;

  private constructor() {
    this.group.name = "StudioLights";
  }

  static async load(url: string): Promise<StudioLights> {
    if (!StudioLights.ltcReady) {
      RectAreaLightNode.setLTC(RectAreaLightTexturesLib.init());
      StudioLights.ltcReady = true;
    }

    const gltf = await new GLTFLoader().loadAsync(url);
    // Bake nested transforms so each Mesh's world (relative to scene root)
    // includes its parent chain. We use these as the "geometry-anchored
    // local frame" inside one studiolight.
    gltf.scene.updateMatrixWorld(true);

    const out = new StudioLights();

    gltf.scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;
      const inst = new THREE.InstancedMesh(mesh.geometry, mesh.material, 2);
      inst.castShadow = true;
      inst.receiveShadow = true;
      // Matrix updates are pushed by hand from `refresh`; let three skip the
      // per-frame auto-update.
      inst.matrixAutoUpdate = false;
      inst.frustumCulled = false;   // 2 instances may straddle the bbox
      out.group.add(inst);
      out.instanced.push({ mesh: inst, meshLocal: mesh.matrixWorld.clone() });
    });

    if (out.instanced.length === 0) {
      throw new Error(`StudioLights: GLB at ${url} contains no meshes`);
    }

    // Two RectAreaLights, one per studiolight. They live as children of the
    // group so adding/removing the StudioLights handle pulls everything.
    for (let i = 0; i < 2; i++) {
      const L = new THREE.RectAreaLight(0xffffff, 1, 1, 1);
      L.matrixAutoUpdate = false;
      out.group.add(L);
      out.lights.push(L);

      const helper = new RectAreaLightHelper(L);
      helper.visible = false;
      L.add(helper);
      out.helpers.push(helper);
    }

    out.refresh();
    return out;
  }

  setStudioTransform(idx: 0 | 1, t: StudioTransform): void {
    this.transforms[idx] = { ...t };
    this.refresh();
  }

  setBulbParams(p: StudioBulbParams): void {
    this.bulb = { ...p };
    this.refresh();
  }

  /** Recompose every InstancedMesh's per-instance matrix and every rect
   *  light's world matrix from the current transform / bulb state. Cheap;
   *  call freely from UI bindings. */
  private refresh(): void {
    const studio = new THREE.Matrix4();
    const local = new THREE.Matrix4();
    const final = new THREE.Matrix4();
    const _q = new THREE.Quaternion();
    const _e = new THREE.Euler();
    const _p = new THREE.Vector3();
    const _s = new THREE.Vector3();

    for (let i = 0; i < 2; i++) {
      const t = this.transforms[i];
      _e.set(0, t.rotY, 0);
      _q.setFromEuler(_e);
      _p.set(t.x, t.y, t.z);
      _s.setScalar(t.scale);
      studio.compose(_p, _q, _s);

      // 1. Push instance matrix for each InstancedMesh.
      for (const im of this.instanced) {
        final.multiplyMatrices(studio, im.meshLocal);
        im.mesh.setMatrixAt(i, final);
      }

      // 2. Compose the rect light's world matrix: studio · bulbLocal.
      const b = this.bulb;
      _e.set(b.rotX, b.rotY, b.rotZ);
      _q.setFromEuler(_e);
      _p.set(b.offsetX, b.offsetY, b.offsetZ);
      _s.setScalar(1);   // light extents are sized via .width/.height
      local.compose(_p, _q, _s);
      final.multiplyMatrices(studio, local);

      const L = this.lights[i];
      L.matrix.copy(final);
      L.matrixWorld.copy(final);
      // Decompose into position / quaternion so internals that read those
      // (e.g. helper bbox, LTC pose) stay in sync.
      final.decompose(L.position, L.quaternion, L.scale);
    }

    // 3. Bulb shared params — color / intensity / extents / helper toggle.
    for (const L of this.lights) {
      L.color.set(this.bulb.color);
      L.intensity = this.bulb.intensity;
      L.width = this.bulb.width;
      L.height = this.bulb.height;
    }
    for (const h of this.helpers) {
      h.visible = this.bulb.showHelper;
    }

    for (const im of this.instanced) {
      im.mesh.instanceMatrix.needsUpdate = true;
      im.mesh.computeBoundingSphere();
    }
  }
}
