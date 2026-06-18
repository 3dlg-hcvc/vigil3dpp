import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

const MANIFEST_URL = new URL("../viewer/vigil3dpp_descriptions.json", import.meta.url);
const MESH_BASE = new URL("../viewer/meshes/", import.meta.url);

const TARGET_COLOR = 0x22c55e;
const BOX_LINE_WIDTH = 3;

function groupByScene(groundingList) {
  const map = new Map();
  for (const g of groundingList) {
    const sid = g.scene_id;
    if (!map.has(sid)) map.set(sid, []);
    map.get(sid).push(g);
  }
  return map;
}

function makeBoxWireframe(box, color) {
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
  const q = box.rotation;
  lines.quaternion.set(q[0], q[1], q[2], q[3]);
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

/** Ensures per-vertex colors are usable (linear-ish 0–1 floats, count matches position). */
function prepareVertexColors(geometry) {
  const pos = geometry.getAttribute("position");
  const col = geometry.getAttribute("color");
  if (!pos || !col || col.itemSize < 3) return false;
  if (col.count !== pos.count) {
    console.warn("PLY color/position count mismatch", {
      colorCount: col.count,
      positionCount: pos.count,
      attributeKeys: Object.keys(geometry.attributes),
    });
    return false;
  }

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

async function main() {
  const wrap = document.getElementById("vigil3d-viewer-canvas-wrap");
  const sceneSelect = document.getElementById("vigil3d-scene-select");
  const descEl = document.getElementById("vigil3d-desc-text");
  const idxEl = document.getElementById("vigil3d-desc-index");
  const btnPrev = document.getElementById("vigil3d-btn-prev");
  const btnNext = document.getElementById("vigil3d-btn-next");
  const loadingEl = document.getElementById("vigil3d-viewer-loading");

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
  // scene.background = new THREE.Color(0xf4f4f5);
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
  const boxRoot = new THREE.Group();
  scene.add(pcRoot);
  scene.add(boxRoot);

  const plyCache = new Map();
  const loader = new PLYLoader();
  let lineResolutionW = 1;
  let lineResolutionH = 1;

  let currentSceneId = sceneIds[0];
  let descIndex = 0;

  function resize() {
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    if (w < 2 || h < 2) return;
    lineResolutionW = w;
    lineResolutionH = h;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    for (const ch of boxRoot.children) {
      if (ch.material && ch.material.resolution) {
        ch.material.resolution.set(lineResolutionW, lineResolutionH);
      }
    }
  }

  const ro = new ResizeObserver(() => resize());
  ro.observe(wrap);
  resize();

  function renderBoxesFor(grounding) {
    clearGroup(boxRoot);
    const entities = grounding.entities || [];
    for (const ent of entities) {
      if (!ent.is_target || !ent.box) continue;
      const boxLines = makeBoxWireframe(ent.box, TARGET_COLOR);
      if (boxLines.material && boxLines.material.resolution) {
        boxLines.material.resolution.set(lineResolutionW, lineResolutionH);
      }
      boxRoot.add(boxLines);
    }
  }

  function updateDescriptionUI() {
    const list = byScene.get(currentSceneId) || [];
    const g = list[descIndex];
    if (!g) return;
    descEl.textContent = g.text || "";
    if (idxEl) idxEl.textContent = `${descIndex + 1} / ${list.length}`;
    renderBoxesFor(g);
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

  async function applySceneChange() {
    currentSceneId = sceneSelect.value;
    descIndex = 0;
    setNavEnabled(false);
    try {
      await ensurePointCloud(currentSceneId);
      updateDescriptionUI();
    } catch {
      /* handled in loader */
    }
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

  function tick() {
    requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
  }
  tick();

  setNavEnabled(false);
  await ensurePointCloud(currentSceneId);
  updateDescriptionUI();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => void main());
} else {
  void main();
}
