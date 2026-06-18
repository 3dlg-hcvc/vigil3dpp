import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

const MANIFEST_URL = new URL("../viewer/eval_descriptions_pred.json", import.meta.url);
const MESH_BASE = new URL("../viewer/meshes/", import.meta.url);

const GT_COLOR = 0x22c55e;
const PRED_CORRECT_COLOR = 0x3b82f6;
const PRED_WRONG_COLOR = 0xef4444;
const IOU_CORRECT_THRESHOLD = 0.25;
const BOX_LINE_WIDTH = 3;

const MODELS = [
  { name: "3D-VisTA", file: "3d-vista.json" },
  { name: "3D-GRAND", file: "3d-grand.json" },
  { name: "GPS", file: "gps.json" },
  { name: "ZSVG3D", file: "zsvg3d.json" },
  { name: "V3DM", file: "v3dm.json" },
];

function groupByScene(groundingList) {
  const map = new Map();
  for (const g of groundingList) {
    const sid = g.scene_id;
    if (!map.has(sid)) map.set(sid, []);
    map.get(sid).push(g);
  }
  return map;
}

function makeOrientedBoxWireframe(box, color) {
  const hx = box.half_dims[0] * 2;
  const hy = box.half_dims[1] * 2;
  const hz = box.half_dims[2] * 2;
  const geom = new THREE.BoxGeometry(hx, hy, hz);
  const edges = new THREE.EdgesGeometry(geom);
  const lineGeom = new LineSegmentsGeometry();
  lineGeom.setPositions(edges.getAttribute("position").array);
  const mat = new LineMaterial({
    color,
    linewidth: BOX_LINE_WIDTH,
    worldUnits: false,
  });
  const lines = new LineSegments2(lineGeom, mat);
  lines.position.set(box.center[0], box.center[1], box.center[2]);
  const q = box.rotation || [0, 0, 0, 1];
  lines.quaternion.set(q[0], q[1], q[2], q[3]);
  geom.dispose();
  edges.dispose();
  return lines;
}

function makeAxisAlignedBoxWireframe(center, extent, color) {
  const geom = new THREE.BoxGeometry(extent[0], extent[1], extent[2]);
  const edges = new THREE.EdgesGeometry(geom);
  const lineGeom = new LineSegmentsGeometry();
  lineGeom.setPositions(edges.getAttribute("position").array);
  const mat = new LineMaterial({
    color,
    linewidth: BOX_LINE_WIDTH,
    worldUnits: false,
  });
  const lines = new LineSegments2(lineGeom, mat);
  lines.position.set(center[0], center[1], center[2]);
  geom.dispose();
  edges.dispose();
  return lines;
}

function clearGroup(group) {
  while (group.children.length) {
    const ch = group.children[0];
    group.remove(ch);
    if (ch.geometry) ch.geometry.dispose();
    if (ch.material) ch.material.dispose();
  }
}

function detachChildren(group) {
  while (group.children.length) {
    group.remove(group.children[0]);
  }
}

function attributeBackingArray(attr) {
  if (!attr) return null;
  if (attr.array) return attr.array;
  if (attr.data && attr.data.array) return attr.data.array;
  return null;
}

function prepareVertexColors(geometry) {
  const pos = geometry.getAttribute("position");
  const col = geometry.getAttribute("color");
  if (!pos || !col || col.itemSize < 3) return false;
  if (col.count !== pos.count) return false;

  const arr = attributeBackingArray(col);
  if (!arr || arr.length === 0) return false;
  if (col.normalized) return true;

  const stride = col.itemSize;
  const n = col.count;
  let maxCh = 0;
  for (let i = 0; i < n; i++) {
    const j = i * stride;
    maxCh = Math.max(maxCh, arr[j], arr[j + 1], arr[j + 2]);
  }
  if (maxCh <= 1.0) return true;

  const out = new Float32Array(n * 3);
  const divisor = maxCh > 255 ? maxCh : 255;
  for (let i = 0; i < n; i++) {
    const j = i * stride;
    const k = i * 3;
    out[k] = arr[j] / divisor;
    out[k + 1] = arr[j + 1] / divisor;
    out[k + 2] = arr[j + 2] / divisor;
  }
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(out, 3));
  return true;
}

function fitCameraToObject(camera, controls, object, margin = 1.35) {
  const box = new THREE.Box3().setFromObject(object);
  const sphere = new THREE.Sphere();
  box.getBoundingSphere(sphere);
  const dist = Math.max(sphere.radius * margin, 0.5);
  const dir = new THREE.Vector3(0.55, 0.35, 1).normalize();
  camera.near = Math.max(dist / 500, 0.01);
  camera.far = Math.max(dist * 50, 100);
  camera.updateProjectionMatrix();
  camera.position.copy(sphere.center).addScaledVector(dir, dist);
  controls.target.copy(sphere.center);
  controls.update();
}

function orientedBoxToAabb(box) {
  const center = new THREE.Vector3(box.center[0], box.center[1], box.center[2]);
  const hx = box.half_dims[0];
  const hy = box.half_dims[1];
  const hz = box.half_dims[2];
  const quat = new THREE.Quaternion(
    (box.rotation && box.rotation[0]) || 0,
    (box.rotation && box.rotation[1]) || 0,
    (box.rotation && box.rotation[2]) || 0,
    (box.rotation && box.rotation[3]) || 1
  );
  const corners = [
    new THREE.Vector3(-hx, -hy, -hz),
    new THREE.Vector3(-hx, -hy, hz),
    new THREE.Vector3(-hx, hy, -hz),
    new THREE.Vector3(-hx, hy, hz),
    new THREE.Vector3(hx, -hy, -hz),
    new THREE.Vector3(hx, -hy, hz),
    new THREE.Vector3(hx, hy, -hz),
    new THREE.Vector3(hx, hy, hz),
  ];
  const aabb = new THREE.Box3();
  aabb.makeEmpty();
  for (const c of corners) {
    c.applyQuaternion(quat).add(center);
    aabb.expandByPoint(c);
  }
  return aabb;
}

function predToAabb(predBox) {
  const center = predBox[0];
  const extent = predBox[1];
  const half = [extent[0] / 2, extent[1] / 2, extent[2] / 2];
  return new THREE.Box3(
    new THREE.Vector3(center[0] - half[0], center[1] - half[1], center[2] - half[2]),
    new THREE.Vector3(center[0] + half[0], center[1] + half[1], center[2] + half[2])
  );
}

function boxIou(a, b) {
  const overlap = a.clone().intersect(b);
  if (overlap.isEmpty()) return 0;
  const ovSize = new THREE.Vector3();
  overlap.getSize(ovSize);
  const interVol = ovSize.x * ovSize.y * ovSize.z;

  const aSize = new THREE.Vector3();
  const bSize = new THREE.Vector3();
  a.getSize(aSize);
  b.getSize(bSize);
  const aVol = aSize.x * aSize.y * aSize.z;
  const bVol = bSize.x * bSize.y * bSize.z;
  const denom = aVol + bVol - interVol;
  if (denom <= 0) return 0;
  return interVol / denom;
}

async function main() {
  const wrap = document.getElementById("vigil3d-pred-viewer-canvas-wrap");
  const sceneSelect = document.getElementById("vigil3d-pred-scene-select");
  const descEl = document.getElementById("vigil3d-pred-desc-text");
  const idxEl = document.getElementById("vigil3d-pred-desc-index");
  const btnPrev = document.getElementById("vigil3d-pred-btn-prev");
  const btnNext = document.getElementById("vigil3d-pred-btn-next");
  const loadingEl = document.getElementById("vigil3d-pred-viewer-loading");
  const modelButtons = Array.from(document.querySelectorAll(".vigil3d-model-option"));

  if (!wrap || !sceneSelect || !descEl) return;

  const setLoading = (msg) => {
    if (loadingEl) {
      loadingEl.hidden = !msg;
      loadingEl.textContent = msg || "";
    }
  };

  const setNavEnabled = (on) => {
    btnPrev.disabled = !on;
    btnNext.disabled = !on;
    sceneSelect.disabled = !on;
  };

  let manifest;
  try {
    setLoading("Loading examples…");
    const res = await fetch(MANIFEST_URL);
    manifest = await res.json();
  } catch (e) {
    console.error(e);
    setLoading("Could not load viewer data.");
    return;
  }

  const byScene = groupByScene(manifest.grounding || []);
  const sceneIds = Array.from(byScene.keys()).sort();
  if (sceneIds.length === 0) {
    setLoading("No examples in manifest.");
    return;
  }

  sceneSelect.innerHTML = "";
  for (const sid of sceneIds) {
    const opt = document.createElement("option");
    opt.value = sid;
    opt.textContent = sid.replace(/_/g, " ");
    sceneSelect.appendChild(opt);
  }

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a1a);

  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 500);
  camera.up.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  if (THREE.ColorManagement) {
    THREE.ColorManagement.enabled = true;
  }
  wrap.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.DOLLY,
    RIGHT: THREE.MOUSE.PAN,
  };
  controls.touches = {
    ONE: THREE.TOUCH.ROTATE,
    TWO: THREE.TOUCH.DOLLY_PAN,
  };
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  const amb = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(amb);

  const pcRoot = new THREE.Group();
  const gtRoot = new THREE.Group();
  const predRoot = new THREE.Group();
  scene.add(pcRoot);
  scene.add(gtRoot);
  scene.add(predRoot);

  const plyCache = new Map();
  const loader = new PLYLoader();
  const modelRawCache = new Map();
  const modelRawLoadPromises = new Map();
  const scenePredictionCache = new Map();
  const scenePredictionLoadPromises = new Map();
  let lineResolutionW = 1;
  let lineResolutionH = 1;

  let currentSceneId = sceneIds[0];
  let descIndex = 0;
  let activeModel = null;

  function setActiveModelUI(nameOrNull) {
    for (const btn of modelButtons) {
      const active = btn.dataset.model === nameOrNull;
      btn.classList.toggle("is-active", active);
    }
  }

  function resize() {
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    if (w < 2 || h < 2) return;
    lineResolutionW = w;
    lineResolutionH = h;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    for (const root of [gtRoot, predRoot]) {
      for (const ch of root.children) {
        if (ch.material && ch.material.resolution) {
          ch.material.resolution.set(lineResolutionW, lineResolutionH);
        }
      }
    }
  }

  const ro = new ResizeObserver(() => resize());
  ro.observe(wrap);
  resize();

  function currentGrounding() {
    const list = byScene.get(currentSceneId) || [];
    return list[descIndex] || null;
  }

  function renderGroundTruthBoxes(grounding) {
    clearGroup(gtRoot);
    const entities = grounding?.entities || [];
    for (const ent of entities) {
      if (!ent.is_target || !ent.box) continue;
      const gtLines = makeOrientedBoxWireframe(ent.box, GT_COLOR);
      if (gtLines.material && gtLines.material.resolution) {
        gtLines.material.resolution.set(lineResolutionW, lineResolutionH);
      }
      gtRoot.add(gtLines);
    }
  }

  function getGroundTruthAabbs(grounding) {
    const entities = grounding?.entities || [];
    const out = [];
    for (const ent of entities) {
      if (!ent.is_target || !ent.box) continue;
      out.push(orientedBoxToAabb(ent.box));
    }
    return out;
  }

  function renderPredictionsForCurrent() {
    clearGroup(predRoot);
    if (!activeModel) return;
    const grounding = currentGrounding();
    if (!grounding || !grounding.id) return;
    const sceneModels = scenePredictionCache.get(currentSceneId);
    if (!sceneModels) return;
    const byId = sceneModels.get(activeModel);
    if (!byId) return;
    const predEntry = byId.get(grounding.id);
    if (!predEntry || !Array.isArray(predEntry.prediction)) return;
    console.log("[viewer_predictions] predEntry", predEntry.prediction);

    const gtAabbs = getGroundTruthAabbs(grounding);
    for (const predBox of predEntry.prediction) {
      if (!Array.isArray(predBox) || predBox.length < 2) continue;
      const center = predBox[0];
      const extent = predBox[1];
      if (!Array.isArray(center) || !Array.isArray(extent)) continue;
      if (center.length < 3 || extent.length < 3) continue;

      const predAabb = predToAabb(predBox);
      let bestIou = 0;
      for (const gtAabb of gtAabbs) {
        bestIou = Math.max(bestIou, boxIou(predAabb, gtAabb));
      }
      const color = bestIou >= IOU_CORRECT_THRESHOLD ? PRED_CORRECT_COLOR : PRED_WRONG_COLOR;
      const predLines = makeAxisAlignedBoxWireframe(center, extent, color);
      if (predLines.material && predLines.material.resolution) {
        predLines.material.resolution.set(lineResolutionW, lineResolutionH);
      }
      predRoot.add(predLines);
      console.log("[viewer_predictions] rendered prediction box", {
        model: activeModel,
        scene_id: currentSceneId,
        example_id: grounding.id,
        center: [center[0], center[1], center[2]],
        extent: [extent[0], extent[1], extent[2]],
      });
    }
  }

  function updateDescriptionUI() {
    const list = byScene.get(currentSceneId) || [];
    const g = list[descIndex];
    if (!g) return;
    descEl.textContent = g.text || "";
    if (idxEl) idxEl.textContent = `${descIndex + 1} / ${list.length}`;
    renderGroundTruthBoxes(g);
    renderPredictionsForCurrent();
  }

  async function ensurePointCloud(sceneId) {
    if (plyCache.has(sceneId)) {
      detachChildren(pcRoot);
      pcRoot.add(plyCache.get(sceneId));
      fitCameraToObject(camera, controls, pcRoot);
      setLoading("");
      setNavEnabled(true);
      return;
    }
    const url = new URL(`${sceneId}.ply`, MESH_BASE);
    setLoading("Loading scene…");
    setNavEnabled(false);
    return new Promise((resolve, reject) => {
      loader.load(
        url.href,
        (geometry) => {
          const hasVertexColors = prepareVertexColors(geometry);
          let root;
          if (geometry.index && geometry.index.count > 0) {
            geometry.computeVertexNormals();
            const mat = hasVertexColors
              ? new THREE.MeshBasicMaterial({
                  vertexColors: true,
                  side: THREE.DoubleSide,
                })
              : new THREE.MeshStandardMaterial({
                  vertexColors: false,
                  roughness: 1,
                  metalness: 0,
                  color: 0xb4b4be,
                });
            root = new THREE.Mesh(geometry, mat);
          } else {
            geometry.computeVertexNormals();
            const mat = new THREE.PointsMaterial({
              size: 0.02,
              vertexColors: hasVertexColors,
              sizeAttenuation: true,
            });
            if (!hasVertexColors) mat.color.setHex(0x888888);
            root = new THREE.Points(geometry, mat);
          }
          root.name = `pc_${sceneId}`;
          plyCache.set(sceneId, root);
          detachChildren(pcRoot);
          pcRoot.add(root);
          fitCameraToObject(camera, controls, pcRoot);
          setLoading("");
          setNavEnabled(true);
          resolve();
        },
        undefined,
        (err) => {
          console.error(err);
          setLoading("Failed to load scene mesh.");
          setNavEnabled(true);
          reject(err);
        }
      );
    });
  }

  async function ensureModelRawLoaded(modelInfo) {
    if (modelRawCache.has(modelInfo.name)) return modelRawCache.get(modelInfo.name);
    if (modelRawLoadPromises.has(modelInfo.name)) return modelRawLoadPromises.get(modelInfo.name);

    const promise = (async () => {
      const url = new URL(`../viewer/predictions/${modelInfo.file}`, import.meta.url);
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      modelRawCache.set(modelInfo.name, json || []);
      return modelRawCache.get(modelInfo.name);
    })();

    modelRawLoadPromises.set(modelInfo.name, promise);
    try {
      return await promise;
    } finally {
      modelRawLoadPromises.delete(modelInfo.name);
    }
  }

  async function ensureScenePredictionsLoaded(sceneId) {
    if (scenePredictionCache.has(sceneId)) return true;
    if (scenePredictionLoadPromises.has(sceneId)) return scenePredictionLoadPromises.get(sceneId);

    const promise = (async () => {
      const sceneGroundings = byScene.get(sceneId) || [];
      const sceneIds = new Set(sceneGroundings.map((g) => g.id).filter(Boolean));
      const modelMapByName = new Map();

      await Promise.all(
        MODELS.map(async (modelInfo) => {
          try {
            const raw = await ensureModelRawLoaded(modelInfo);
            const byId = new Map();
            for (const entry of raw || []) {
              if (entry && entry.id && sceneIds.has(entry.id)) {
                byId.set(entry.id, entry);
              }
            }
            modelMapByName.set(modelInfo.name, byId);
          } catch (err) {
            console.error(`Failed to load predictions for ${modelInfo.name}`, err);
            modelMapByName.set(modelInfo.name, new Map());
          }
        })
      );

      scenePredictionCache.set(sceneId, modelMapByName);
      return true;
    })();

    scenePredictionLoadPromises.set(sceneId, promise);
    try {
      return await promise;
    } finally {
      scenePredictionLoadPromises.delete(sceneId);
    }
  }

  async function applySceneChange() {
    currentSceneId = sceneSelect.value;
    descIndex = 0;
    setNavEnabled(false);
    try {
      await ensurePointCloud(currentSceneId);
      setNavEnabled(false);
      setLoading("Loading predictions…");
      await ensureScenePredictionsLoaded(currentSceneId);
      setLoading("");
      setNavEnabled(true);
      updateDescriptionUI();
    } catch {
      /* handled in loader */
      setLoading("");
      setNavEnabled(true);
    }
  }

  async function toggleModel(modelName) {
    console.log("[viewer_predictions] toggling model", modelName, "from active", activeModel);
    if (activeModel === modelName) {
      activeModel = null;
      setActiveModelUI(null);
      clearGroup(predRoot);
      return;
    }

    const sceneModels = scenePredictionCache.get(currentSceneId);
    console.log("[viewer_predictions] scene models", sceneModels);
    if (!sceneModels || !sceneModels.has(modelName)) return;
    activeModel = modelName;
    setActiveModelUI(modelName);
    renderPredictionsForCurrent();
  }

  sceneSelect.addEventListener("change", () => {
    void applySceneChange();
  });

  btnPrev.addEventListener("click", () => {
    const list = byScene.get(currentSceneId) || [];
    if (list.length === 0) return;
    descIndex = (descIndex - 1 + list.length) % list.length;
    updateDescriptionUI();
  });

  btnNext.addEventListener("click", () => {
    const list = byScene.get(currentSceneId) || [];
    if (list.length === 0) return;
    descIndex = (descIndex + 1) % list.length;
    updateDescriptionUI();
  });

  for (const btn of modelButtons) {
    btn.addEventListener("click", () => {
      void toggleModel(btn.dataset.model);
    });
  }

  function tick() {
    requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
  }
  tick();

  setNavEnabled(false);
  await ensurePointCloud(currentSceneId);
  setNavEnabled(false);
  setLoading("Loading predictions…");
  await ensureScenePredictionsLoaded(currentSceneId);
  setLoading("");
  setNavEnabled(true);
  updateDescriptionUI();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => void main());
} else {
  void main();
}
