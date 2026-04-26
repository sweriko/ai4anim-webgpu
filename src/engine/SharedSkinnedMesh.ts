/** Shared instanced skinning — one SkinnedMesh renders every active agent
 *  in a single draw call, with per-instance bone matrices read from a
 *  shared storage buffer.
 *
 *  three.js's standard `isInstancedMesh` on a SkinnedMesh only supports
 *  *identical* poses across instances (all read the SAME
 *  `skeleton.boneMatrices`). Our agents have unique predicted poses each
 *  frame, so we override the skinning:
 *
 *     boneMatrix[k] = sharedStorage[instanceIndex × totalBones + skinIndex[k]]
 *
 *  Shared storage is sized `maxAgents × totalBones × mat4`. Each agent
 *  writes 23 tracked world matrices × boneInverses + cascaded untracked
 *  bone matrices directly into its slice per frame.
 *
 *  Compared to the per-agent SkinnedMesh pattern this saves N−1 draw calls,
 *  N−1 scene-graph walks, and N−1 per-material uniform uploads per frame.
 */

import * as THREE from "three";
// TSL's .d.ts types are weaker than its runtime node chaining — we freely
// call `.mul` / `.add` / `.xyz` on whatever the math helpers return, which
// matches how three's own examples (BitonicSort, TiledLights, …) write TSL.
// Cast imported primitives to a permissive shape to avoid noisy casts
// throughout the kernel body.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
import * as TSL from "three/tsl";
import { StorageBufferAttribute, MeshStandardNodeMaterial } from "three/webgpu";

const {
  Fn, attribute, instanceIndex, normalLocal, storage, tangentLocal, uint,
  uniform, vec4,
} = TSL as unknown as Record<string, (...args: unknown[]) => unknown>;

/** Permissive Node surface — matches how every TSL example written in JS
 *  uses these node objects: chain-friendly with `.mul`, `.add`, `.xyz`, … */
type N = {
  mul(x: unknown): N;
  add(x: unknown): N;
  element(x: unknown): N;
  transformDirection(x: unknown): N;
  x: N; y: N; z: N; w: N; xyz: N;
};

export interface SharedSkinnedRig {
  /** The one mesh added to the scene; `count` tracks active agent count. */
  mesh: THREE.SkinnedMesh;
  /** Flat [maxAgents × totalBones × 16] bone-matrix storage (CPU writable). */
  boneMatricesAttr: StorageBufferAttribute;
  /** CPU view — agents write into `boneMatricesArray.subarray(slot*stride, …)`. */
  boneMatricesArray: Float32Array;
  /** Floats per agent's slice (= totalBones × 16). */
  agentStride: number;
  /** Total bones in the skeleton (tracked + untracked). */
  totalBones: number;
  /** Bone name → index within skeleton.bones. Agents use this to locate each
   *  tracked bone's slot inside their stride. */
  boneNameToIndex: Map<string, number>;
  /** Bind-pose local matrices per bone — for CPU-side computation of
   *  untracked bone world matrices via `world = parent.world × bindLocal`. */
  bindLocalMatrices: THREE.Matrix4[];
  /** Inverse bind matrices per bone (three's skeleton convention). */
  boneInverses: THREE.Matrix4[];
  /** For each untracked bone, its own skeleton index + its parent's index
   *  (−1 if the parent isn't in the skeleton). Topologically ordered so a
   *  parent's world matrix is already fresh by the time a child is written. */
  untrackedCascade: { skelIdx: number; parentSkelIdx: number }[];
  /** Template scene-graph root — used by Actor to force matrixWorld updates
   *  before snapshotting bindWorld. Shared by every agent. */
  templateRoot: THREE.Object3D;
  /** The 23 tracked bones from the template skeleton in `bone_names` order.
   *  Shared by every agent — they're read-only (kinematic math writes into
   *  the shared storage buffer instead of touching these bones). */
  trackedBones: THREE.Bone[];
}

/** Build the shared instanced SkinnedMesh from a template GLB mesh.
 *  Geometry is shared (attributes live once on the GPU); material is a
 *  custom MeshStandardNodeMaterial with overridden skinning nodes.
 *
 *  `trackedParentNames` is optional — when provided, any `trackedBoneNames[i]`
 *  that's missing from the template's scene graph will be synthesized at
 *  identity-local to its declared parent. Used for models (e.g. the Quadruped
 *  dog) whose training data includes leaf markers that weren't exported in
 *  the shipped GLB. */
export function createSharedSkinnedRig(
  template: THREE.SkinnedMesh,
  maxAgents: number,
  trackedBoneNames: readonly string[],
  trackedParentNames?: readonly (string | null)[],
): SharedSkinnedRig {
  // Walk the template scene graph so we can find non-skinning "leaf marker"
  // bones too. Some rigs (Dog.glb) have sites (e.g. HeadSite) that the
  // network expects but that aren't in `skeleton.bones` — we still need their
  // world-space transforms for inference features.
  const sceneBones = new Map<string, THREE.Object3D>();
  template.updateWorldMatrix(true, true);
  let templateRootFind: THREE.Object3D = template;
  while (templateRootFind.parent !== null) templateRootFind = templateRootFind.parent;
  templateRootFind.traverse((o) => {
    if (o.name && !sceneBones.has(o.name)) sceneBones.set(o.name, o);
  });

  // Synthesize truly-missing tracked bones (absent from both skeleton and
  // scene graph) by creating a new Bone parented under its declared parent.
  if (trackedParentNames) {
    for (let i = 0; i < trackedBoneNames.length; i++) {
      const name = trackedBoneNames[i];
      if (sceneBones.has(name)) continue;
      const parentName = trackedParentNames[i];
      if (parentName === null) continue;
      const parent = sceneBones.get(parentName);
      if (!parent) {
        throw new Error(
          `createSharedSkinnedRig: cannot synthesize '${name}' — parent '${parentName}' also missing`);
      }
      const synth = new THREE.Bone();
      synth.name = name;
      parent.add(synth);
      sceneBones.set(name, synth);
    }
  }

  // Build a fresh skeleton that includes any synthesized bones. We cannot
  // mutate `template.skeleton.bones` in place because `Skeleton.boneMatrices`
  // is a Float32Array sized once at construction — the shadow pass (a stock
  // three skinning path) reads from that array via a uniform buffer binding
  // and would overflow if the bone count grew. Creating a new Skeleton
  // reallocates boneMatrices to match.
  const templateSkeleton = template.skeleton;
  const existingNames = new Set(templateSkeleton.bones.map((b) => b.name));
  const synthesizedBones: THREE.Bone[] = [];
  for (const name of trackedBoneNames) {
    if (existingNames.has(name)) continue;
    const obj = sceneBones.get(name);
    if (obj && obj instanceof THREE.Bone) synthesizedBones.push(obj);
  }
  const extendedBones = [...templateSkeleton.bones, ...synthesizedBones];
  const extendedInverses = [
    ...templateSkeleton.boneInverses.map((m) => m.clone()),
    ...synthesizedBones.map(() => new THREE.Matrix4()),
  ];
  const skeleton = synthesizedBones.length > 0
    ? new THREE.Skeleton(extendedBones, extendedInverses)
    : templateSkeleton;
  template.updateWorldMatrix(true, true);
  const totalBones = skeleton.bones.length;

  // Snapshot template data.
  const boneNameToIndex = new Map<string, number>();
  const bindLocalMatrices: THREE.Matrix4[] = [];
  for (let i = 0; i < skeleton.bones.length; i++) {
    const b = skeleton.bones[i];
    boneNameToIndex.set(b.name, i);
    bindLocalMatrices.push(b.matrix.clone());
  }
  const boneInverses = skeleton.boneInverses.map((m) => m.clone());

  const trackedSet = new Set(trackedBoneNames);
  const untrackedCascade: { skelIdx: number; parentSkelIdx: number }[] = [];
  for (let i = 0; i < skeleton.bones.length; i++) {
    const b = skeleton.bones[i];
    if (trackedSet.has(b.name)) continue;
    const parent = b.parent;
    const parentSkelIdx = parent && boneNameToIndex.has(parent.name)
      ? boneNameToIndex.get(parent.name)! : -1;
    untrackedCascade.push({ skelIdx: i, parentSkelIdx });
  }

  // Shared bone-matrix storage: one mat4 per (agent, bone) pair.
  // itemSize=16 → `storage(attr, 'mat4', count)` emits `array<mat4<f32>>`.
  const boneMatricesAttr = new StorageBufferAttribute(
    maxAgents * totalBones, 16,
  );
  const boneMatricesArray = boneMatricesAttr.array as Float32Array;
  const boneStorage = storage(
    boneMatricesAttr as unknown,
    "mat4" as unknown,
    maxAgents * totalBones as unknown,
  ) as unknown as N;

  // Identity-initialise every slot so warmup frames don't render at origin.
  for (let i = 0; i < maxAgents * totalBones; i++) {
    boneMatricesArray[i * 16 + 0]  = 1;
    boneMatricesArray[i * 16 + 5]  = 1;
    boneMatricesArray[i * 16 + 10] = 1;
    boneMatricesArray[i * 16 + 15] = 1;
  }
  boneMatricesAttr.needsUpdate = true;

  // -- Custom skinning nodes ---------------------------------------------
  const bindU = uniform(template.bindMatrix as unknown, "mat4" as unknown) as unknown as N;
  const bindInvU = uniform(template.bindMatrixInverse as unknown, "mat4" as unknown) as unknown as N;
  const totalBonesU = uint(totalBones as unknown) as unknown as N;
  const skinIdxA = attribute("skinIndex" as unknown, "uvec4" as unknown) as unknown as N;
  const skinWA   = attribute("skinWeight" as unknown, "vec4" as unknown) as unknown as N;

  // Four bone-matrix fetches for this vertex, offset into the agent's slice.
  const fetchBones = () => {
    const base = (instanceIndex as unknown as N).mul(totalBonesU);
    return {
      bX: boneStorage.element(base.add(skinIdxA.x)),
      bY: boneStorage.element(base.add(skinIdxA.y)),
      bZ: boneStorage.element(base.add(skinIdxA.z)),
      bW: boneStorage.element(base.add(skinIdxA.w)),
    };
  };

  // Combined position + normal skinning. `material.normalNode` expects a
  // VIEW-space value (it becomes `normalView` downstream), whereas our
  // skinning math produces a local-space normal. To avoid re-transforming
  // and matching three's own SkinningNode flow, we mutate `normalLocal` as
  // a side effect inside the positionNode Fn — three's default
  // `normalLocal → transformNormalToView → normalView` pipeline then does
  // the view-space conversion for free. Returning the local-space skinned
  // position satisfies material.positionNode.
  const positionNode = (Fn as unknown as (fn: () => unknown) => unknown)(() => {
    const { bX, bY, bZ, bW } = fetchBones();

    // --- position ----
    const pLocal = attribute("position" as unknown, "vec3" as unknown) as unknown as N;
    const bindPos = bindU.mul((vec4 as unknown as (p: N, w: number) => N)(pLocal, 1));
    const skinnedPos = bX.mul(skinWA.x).mul(bindPos)
      .add(bY.mul(skinWA.y).mul(bindPos))
      .add(bZ.mul(skinWA.z).mul(bindPos))
      .add(bW.mul(skinWA.w).mul(bindPos));
    const localPos = bindInvU.mul(skinnedPos).xyz;

    // --- normal + tangent (mutate normalLocal/tangentLocal as side-effects
    //     so three's default normalLocal → normalView → lighting path gets
    //     the correct skinned-world-space normal to view-space-transform). ----
    const skinMat = bX.mul(skinWA.x)
      .add(bY.mul(skinWA.y))
      .add(bZ.mul(skinWA.z))
      .add(bW.mul(skinWA.w));
    const fullNormalMat = bindInvU.mul(skinMat).mul(bindU);
    const nLocalAttr = attribute("normal" as unknown, "vec3" as unknown) as unknown as N;
    const skinnedNormal = fullNormalMat.transformDirection(nLocalAttr).xyz;
    (normalLocal as unknown as { assign(x: unknown): void }).assign(skinnedNormal);

    // Tangent skinning — only meaningful if the geometry has a tangent
    // attribute (normal maps). If absent, tangentLocal exists but the
    // assignment is harmless dead code in the emitted shader.
    if (template.geometry.getAttribute("tangent") !== undefined) {
      const tLocalAttr = attribute("tangent" as unknown, "vec3" as unknown) as unknown as N;
      const skinnedTangent = fullNormalMat.transformDirection(tLocalAttr).xyz;
      (tangentLocal as unknown as { assign(x: unknown): void }).assign(skinnedTangent);
    }

    return localPos;
  });
  const positionCall = (positionNode as () => unknown)();

  // -- Material -----------------------------------------------------------
  // Copy the template's MeshStandardMaterial data (color, maps, …) into a
  // MeshStandardNodeMaterial, then override position + normal with our
  // instanced-skinning nodes. All other node-material fields (envmap,
  // shadows, tone mapping) keep their defaults.
  const material = new MeshStandardNodeMaterial();
  // Copy scalar material data from the template but avoid a full `.copy()`
  // — that pulls in undefined NodeMaterial-specific fields from a plain
  // MeshStandardMaterial template and trips an `isOutputStructNode` check
  // inside three's default fragment flow.
  {
    const src = template.material as THREE.MeshStandardMaterial;
    if (src.color) material.color.copy(src.color);
    if (src.map) material.map = src.map;
    if (src.normalMap) material.normalMap = src.normalMap;
    if (src.aoMap) material.aoMap = src.aoMap;
    if (src.emissive) material.emissive.copy(src.emissive);
    if (src.emissiveMap) material.emissiveMap = src.emissiveMap;
    if (src.roughnessMap) material.roughnessMap = src.roughnessMap;
    if (src.metalnessMap) material.metalnessMap = src.metalnessMap;
    material.roughness = src.roughness ?? 1;
    material.metalness = src.metalness ?? 0;
    material.transparent = src.transparent;
    material.opacity = src.opacity;
    material.side = src.side;
  }
  (material as unknown as { positionNode: unknown }).positionNode = positionCall;
  // normalNode stays null — the skinning Fn above mutates `normalLocal` as a
  // side effect, so three's default `normalLocal → normalView` path produces
  // a correctly-transformed view-space normal downstream.
  (material as unknown as { normalNode: unknown }).normalNode = null;
  // Defensive: ensure the lighting branch runs in NodeMaterial.setup().
  (material as unknown as { fragmentNode: unknown }).fragmentNode = null;

  // -- Mesh ---------------------------------------------------------------
  const mesh = new THREE.SkinnedMesh(template.geometry, material);
  mesh.frustumCulled = false;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.bind(skeleton, template.bindMatrix);

  // Promote to instanced. `instanceMatrix` is identity per instance — agents
  // bake world pose into their bone matrices, so no additional per-instance
  // transform is needed.
  const identity = new THREE.Matrix4();
  const instArr = new Float32Array(maxAgents * 16);
  for (let i = 0; i < maxAgents; i++) identity.toArray(instArr, i * 16);
  const meshX = mesh as unknown as {
    isInstancedMesh: boolean;
    instanceMatrix: THREE.InstancedBufferAttribute;
    count: number;
  };
  meshX.isInstancedMesh = true;
  meshX.instanceMatrix = new THREE.InstancedBufferAttribute(instArr, 16);
  meshX.count = 0;

  // Resolve the 23 tracked bones in bone_names order — Agents snapshot
  // bindWorld from these once, then never write to them.
  const trackedBones: THREE.Bone[] = trackedBoneNames.map((name) => {
    const i = boneNameToIndex.get(name);
    if (i === undefined) {
      throw new Error(`createSharedSkinnedRig: bone '${name}' not in template skeleton`);
    }
    return skeleton.bones[i];
  });

  // Locate the template's scene-graph root. Actor needs this to trigger
  // updateMatrixWorld(true) before snapshotting bindWorld. Walk up from the
  // template mesh to the topmost Object3D in its current tree.
  let templateRoot: THREE.Object3D = template;
  while (templateRoot.parent !== null) templateRoot = templateRoot.parent;

  return {
    mesh,
    boneMatricesAttr,
    boneMatricesArray,
    agentStride: totalBones * 16,
    totalBones,
    boneNameToIndex,
    bindLocalMatrices,
    boneInverses,
    untrackedCascade,
    templateRoot,
    trackedBones,
  };
}
