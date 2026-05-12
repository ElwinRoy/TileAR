import * as THREE from 'three';

// ─── State ────────────────────────────────────────────────────────────────
let renderer, scene, camera;
let xrSession = null;
let hitTestSource = null;
let hitTestSourceRequested = false;
let reticle;

let isLocked = false;
let currentTileIdx = 0;
let tileOpacity = 0.85;
let surfaceMode = 'both'; // 'floor', 'wall', 'both'

// World-anchored plane tracking — each XRPlane gets its own tile mesh
const trackedPlanes = new Map(); // XRPlane -> { mesh, lastUpdate, type }

// Hit-test fallback anchored tiles (when plane detection unavailable)
const anchoredTiles = [];
let anchorBasePos = null; // first hit-test world position, stays fixed
let anchorBaseMat = null;
let anchorMesh = null;
let hasPlaneDetection = false;
let firstHitPlaced = false;

// ─── Tile definitions ─────────────────────────────────────────────────────
const TILES = [
  { name:'Gold',       base:'#c8a84b', grout:'#6b4c1e', sheen:0.30 },
  { name:'Marble',     base:'#ddd8d0', grout:'#9a9590', sheen:0.50 },
  { name:'Charcoal',   base:'#2e2e2e', grout:'#111111', sheen:0.18 },
  { name:'Terracotta', base:'#c97b4b', grout:'#7a3e1e', sheen:0.14 },
  { name:'Green',      base:'#3d7a52', grout:'#1a3d28', sheen:0.10 },
];

// ─── Canvas helpers ───────────────────────────────────────────────────────
const cl       = v => Math.max(0, Math.min(255, v));
const hexToRgb = h => { const n = parseInt(h.replace('#',''), 16); return [(n>>16)&255,(n>>8)&255,n&255]; };
const rgbToHex = (r,g,b) => '#' + [r,g,b].map(v => Math.round(v).toString(16).padStart(2,'0')).join('');
const lighten  = (h, a) => { const [r,g,b] = hexToRgb(h); return rgbToHex(cl(r+255*a), cl(g+255*a), cl(b+255*a)); };
const darken   = (h, a) => lighten(h, -a);

function rRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.lineTo(x+w-r,y); ctx.quadraticCurveTo(x+w,y,x+w,y+r);
  ctx.lineTo(x+w,y+h-r); ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
  ctx.lineTo(x+r,y+h); ctx.quadraticCurveTo(x,y+h,x,y+h-r);
  ctx.lineTo(x,y+r); ctx.quadraticCurveTo(x,y,x+r,y);
  ctx.closePath();
}

function addGrain(ctx, x, y, w, h, amt) {
  const id = ctx.getImageData(x,y,w,h), d = id.data;
  for (let i = 0; i < d.length; i += 4) {
    const n = (Math.random()-0.5)*255*amt;
    d[i]=cl(d[i]+n); d[i+1]=cl(d[i+1]+n); d[i+2]=cl(d[i+2]+n);
  }
  ctx.putImageData(id, x, y);
}

// ─── Tile texture builder ─────────────────────────────────────────────────
function makeTileTexture(idx) {
  const SZ = 512, GROUT = 6, N = 2;
  const cv  = document.createElement('canvas');
  cv.width  = cv.height = SZ;
  const ctx = cv.getContext('2d');
  const t   = TILES[idx];
  const tw  = SZ / N;

  for (let row = 0; row < N; row++) {
    for (let col = 0; col < N; col++) {
      const x = col*tw + GROUT/2, y = row*tw + GROUT/2;
      const w = tw - GROUT, h = tw - GROUT;
      const grd = ctx.createLinearGradient(x,y,x+w,y+h);
      grd.addColorStop(0.0, lighten(t.base, 0.12));
      grd.addColorStop(0.5, t.base);
      grd.addColorStop(1.0, darken(t.base, 0.10));
      ctx.fillStyle = grd;
      rRect(ctx,x,y,w,h,4); ctx.fill();
      addGrain(ctx,x,y,w,h,0.04);
      const hi = ctx.createLinearGradient(x,y,x+w*0.6,y+h*0.6);
      hi.addColorStop(0, `rgba(255,255,255,${t.sheen})`);
      hi.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = hi;
      rRect(ctx,x,y,w,h,4); ctx.fill();
    }
  }
  ctx.fillStyle = t.grout;
  for (let i = 0; i <= N; i++) {
    ctx.fillRect(0, i*tw - GROUT/2, SZ, GROUT);
    ctx.fillRect(i*tw - GROUT/2, 0, GROUT, SZ);
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

// ─── Get tile size from slider ────────────────────────────────────────────
function getTileRepeatSize() {
  const slider = document.getElementById('tile-size-slider');
  const tileM = parseFloat(slider.value) * 0.05 + 0.10;
  return tileM * 2; // texture has 2 tiles per repeat
}

// ─── Create geometry from XRPlane polygon ─────────────────────────────────
// Polygon vertices are in the plane's local space (XZ plane, Y≈0)
function createPlaneGeometry(polygon) {
  if (!polygon || polygon.length < 3) return null;

  const shape = new THREE.Shape();
  shape.moveTo(polygon[0].x, polygon[0].z);
  for (let i = 1; i < polygon.length; i++) {
    shape.lineTo(polygon[i].x, polygon[i].z);
  }

  const geometry = new THREE.ShapeGeometry(shape);
  // ShapeGeometry creates in XY plane — rotate to XZ so it lies flat in plane-local space
  geometry.rotateX(-Math.PI / 2);

  // Compute tiling UVs from local coordinates for consistent tile sizing
  const repeatSize = getTileRepeatSize();
  const pos = geometry.attributes.position;
  const uvs = new Float32Array(pos.count * 2);
  for (let i = 0; i < pos.count; i++) {
    uvs[i * 2]     = pos.getX(i) / repeatSize;
    uvs[i * 2 + 1] = pos.getZ(i) / repeatSize;
  }
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));

  return geometry;
}

// ─── Create tile material ─────────────────────────────────────────────────
function createTileMaterial(tileIdx) {
  const tex = makeTileTexture(tileIdx);
  return new THREE.MeshBasicMaterial({
    map:                 tex,
    transparent:         true,
    opacity:             tileOpacity,
    depthWrite:          false,
    depthTest:           true,
    polygonOffset:       true,
    polygonOffsetFactor: -4,
    polygonOffsetUnits:  -4,
    side:                THREE.DoubleSide,
  });
}

// ─── Create mesh for a detected XRPlane ───────────────────────────────────
function createPlaneTileMesh(plane, poseMatrix) {
  const geo = createPlaneGeometry(plane.polygon);
  if (!geo) return null;

  const mat = createTileMaterial(currentTileIdx);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.matrixAutoUpdate = false;
  mesh.matrix.fromArray(poseMatrix);
  mesh.renderOrder = 1;
  // Start transparent, fade in
  mat.opacity = 0;
  return mesh;
}

// ─── Update geometry when plane polygon changes ───────────────────────────
function updatePlaneMeshGeometry(mesh, polygon) {
  const newGeo = createPlaneGeometry(polygon);
  if (!newGeo) return;
  mesh.geometry.dispose();
  mesh.geometry = newGeo;
}

// ─── Rebuild all plane meshes (tile change / size change) ─────────────────
function rebuildAllPlaneMeshes() {
  for (const [plane, data] of trackedPlanes) {
    const oldMesh = data.mesh;
    const geo = createPlaneGeometry(plane.polygon);
    if (!geo) continue;
    const mat = createTileMaterial(currentTileIdx);
    mat.opacity = tileOpacity;
    const newMesh = new THREE.Mesh(geo, mat);
    newMesh.matrixAutoUpdate = false;
    newMesh.matrix.copy(oldMesh.matrix);
    newMesh.renderOrder = 1;

    scene.remove(oldMesh);
    oldMesh.geometry.dispose();
    if (oldMesh.material.map) oldMesh.material.map.dispose();
    oldMesh.material.dispose();

    scene.add(newMesh);
    data.mesh = newMesh;
  }

  // Also rebuild fallback anchor mesh if exists
  if (anchorMesh) {
    rebuildAnchorMesh();
  }
}

// ─── Hit-test fallback: create/update anchored tile ───────────────────────
function createAnchorMesh(poseMatrix) {
  const slider = document.getElementById('tile-size-slider');
  const sizeM = parseFloat(slider.value) * 0.55;

  const tex = makeTileTexture(currentTileIdx);
  const rep = sizeM / getTileRepeatSize();
  tex.repeat.set(rep, rep);

  const mat = new THREE.MeshBasicMaterial({
    map: tex, transparent: true, opacity: tileOpacity,
    depthWrite: false, depthTest: true,
    polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
    side: THREE.DoubleSide,
  });

  const geo = new THREE.PlaneGeometry(sizeM, sizeM);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.matrixAutoUpdate = false;
  mesh.matrix.fromArray(poseMatrix);
  mesh.renderOrder = 1;
  return mesh;
}

function rebuildAnchorMesh() {
  if (!anchorMesh) return;
  const savedMatrix = anchorMesh.matrix.clone();
  scene.remove(anchorMesh);
  anchorMesh.geometry.dispose();
  if (anchorMesh.material.map) anchorMesh.material.map.dispose();
  anchorMesh.material.dispose();

  anchorMesh = createAnchorMesh(savedMatrix.elements);
  scene.add(anchorMesh);
}

// ─── Lock / unlock ────────────────────────────────────────────────────────
function toggleLock() {
  isLocked = !isLocked;
  const btn = document.getElementById('lock-btn');
  if (isLocked) {
    btn.textContent = '🔓 Unlock';
    btn.classList.add('active-lock');
    document.getElementById('lock-badge').style.display = 'block';
    setStatus('Tiles locked — walk around to inspect!');
  } else {
    btn.textContent = '🔒 Lock';
    btn.classList.remove('active-lock');
    document.getElementById('lock-badge').style.display = 'none';
    setStatus('Scanning for surfaces…');
  }
}
window.toggleLock = toggleLock;

// ─── Reset ────────────────────────────────────────────────────────────────
function resetAR() {
  // Remove all tracked plane meshes
  for (const [plane, data] of trackedPlanes) {
    scene.remove(data.mesh);
    data.mesh.geometry.dispose();
    if (data.mesh.material.map) data.mesh.material.map.dispose();
    data.mesh.material.dispose();
  }
  trackedPlanes.clear();

  // Remove anchor mesh
  if (anchorMesh) {
    scene.remove(anchorMesh);
    anchorMesh.geometry.dispose();
    if (anchorMesh.material.map) anchorMesh.material.map.dispose();
    anchorMesh.material.dispose();
    anchorMesh = null;
  }
  anchorBasePos = null;
  anchorBaseMat = null;
  firstHitPlaced = false;

  isLocked = false;
  const btn = document.getElementById('lock-btn');
  btn.textContent = '🔒 Lock';
  btn.classList.remove('active-lock');

  document.getElementById('scan-ring').style.opacity = '1';
  document.getElementById('lock-badge').style.display = 'none';
  reticle.visible = false;
  setStatus('Scanning for surfaces…');
  updatePlaneCount(0, 0);
}
window.resetAR = resetAR;

// ─── Three.js bootstrap ──────────────────────────────────────────────────
function initThree() {
  scene    = new THREE.Scene();
  camera   = new THREE.PerspectiveCamera();
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.xr.enabled = true;
  renderer.setClearColor(0x000000, 0);
  renderer.sortObjects = true;
  document.getElementById('canvas-container').appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 1.0));

  const rGeo = new THREE.RingGeometry(0.05, 0.08, 36);
  rGeo.applyMatrix4(new THREE.Matrix4().makeRotationX(-Math.PI / 2));
  reticle = new THREE.Mesh(rGeo, new THREE.MeshBasicMaterial({
    color: 0xf0d080, side: THREE.DoubleSide,
    depthWrite: false, transparent: true, opacity: 0.85,
  }));
  reticle.matrixAutoUpdate = false;
  reticle.visible = false;
  scene.add(reticle);
}

// ─── AR session ───────────────────────────────────────────────────────────
async function startAR() {
  if (!navigator.xr) { showNotSupported(); return; }
  const supported = await navigator.xr.isSessionSupported('immersive-ar').catch(() => false);
  if (!supported) { showNotSupported(); return; }

  document.getElementById('landing').style.display = 'none';
  initThree();

  try {
    xrSession = await navigator.xr.requestSession('immersive-ar', {
      requiredFeatures: ['hit-test'],
      optionalFeatures: ['dom-overlay', 'plane-detection', 'local-floor'],
      domOverlay: { root: document.getElementById('hud') },
    });
  } catch (e) {
    try {
      xrSession = await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['hit-test'],
        optionalFeatures: ['dom-overlay'],
        domOverlay: { root: document.getElementById('hud') },
      });
    } catch (e2) { showNotSupported(); return; }
  }

  renderer.xr.setReferenceSpaceType('local-floor');
  await renderer.xr.setSession(xrSession);

  xrSession.addEventListener('end', () => {
    document.getElementById('hud').style.display = 'none';
    document.getElementById('landing').style.display = 'flex';
    hitTestSource = null; hitTestSourceRequested = false;
    trackedPlanes.clear();
    anchorMesh = null; anchorBasePos = null; firstHitPlaced = false;
    isLocked = false; hasPlaneDetection = false;
  });

  document.getElementById('hud').style.display = 'block';
  renderer.setAnimationLoop(onXRFrame);
  setupUI();
}
window.startAR = startAR;

// ─── XR render loop ───────────────────────────────────────────────────────
function onXRFrame(time, frame) {
  if (!frame) return;
  const refSpace = renderer.xr.getReferenceSpace();

  // Lazy-init hit-test source
  if (!hitTestSourceRequested) {
    frame.session.requestReferenceSpace('viewer').then(vs => {
      frame.session.requestHitTestSource({ space: vs })
        .then(src => { hitTestSource = src; })
        .catch(() => setStatus('Hit-test not available'));
    });
    hitTestSourceRequested = true;
  }

  // ── 1. XR Plane Detection — PRIMARY PATH ─────────────────────────────
  // Each detected plane gets its own world-anchored tile mesh.
  // The mesh position/rotation comes from the plane's world pose,
  // so tiles STAY on the surface when camera moves.
  if (frame.detectedPlanes && frame.detectedPlanes.size > 0) {
    hasPlaneDetection = true;
    const currentPlanes = new Set(frame.detectedPlanes);

    // Remove meshes for planes that disappeared
    for (const [plane, data] of trackedPlanes) {
      if (!currentPlanes.has(plane)) {
        scene.remove(data.mesh);
        data.mesh.geometry.dispose();
        if (data.mesh.material.map) data.mesh.material.map.dispose();
        data.mesh.material.dispose();
        trackedPlanes.delete(plane);
      }
    }

    if (!isLocked) {
      let floorCount = 0, wallCount = 0;

      for (const plane of frame.detectedPlanes) {
        const isHoriz = plane.orientation === 'horizontal';
        const isVert  = plane.orientation === 'vertical';

        // Filter by surface mode
        if (surfaceMode === 'floor' && !isHoriz) continue;
        if (surfaceMode === 'wall'  && !isVert)  continue;
        if (!isHoriz && !isVert) continue;

        if (isHoriz) floorCount++; else wallCount++;

        const pose = frame.getPose(plane.planeSpace, refSpace);
        if (!pose) continue;
        const poseMatrix = pose.transform.matrix;

        if (trackedPlanes.has(plane)) {
          const data = trackedPlanes.get(plane);

          // Update world-space transform (keeps tile anchored to real surface)
          data.mesh.matrix.fromArray(poseMatrix);

          // Update geometry if plane polygon grew (live spreading)
          const changeTime = plane.lastChangedTime || 0;
          if (changeTime > data.lastUpdate) {
            updatePlaneMeshGeometry(data.mesh, plane.polygon);
            data.lastUpdate = changeTime;
          }

          // Fade in
          if (data.mesh.material.opacity < tileOpacity) {
            data.mesh.material.opacity = Math.min(tileOpacity,
              data.mesh.material.opacity + 0.02);
          }
        } else {
          // New plane detected — create tile mesh
          const mesh = createPlaneTileMesh(plane, poseMatrix);
          if (!mesh) continue;
          scene.add(mesh);
          trackedPlanes.set(plane, {
            mesh,
            lastUpdate: plane.lastChangedTime || 0,
            type: isVert ? 'wall' : 'floor',
          });
        }
      }

      updatePlaneCount(floorCount, wallCount);

      if (trackedPlanes.size > 0) {
        document.getElementById('scan-ring').style.opacity = '0';
        reticle.visible = false;
        setStatus(`✨ Tiling ${trackedPlanes.size} surface${trackedPlanes.size > 1 ? 's' : ''}`);
      }
    }
  }

  // ── 2. Hit-test fallback — when NO plane detection ───────────────────
  // Places a single tile at the first detected point. Tile is ANCHORED
  // and does NOT follow the camera.
  if (hitTestSource && refSpace) {
    const hits = frame.getHitTestResults(hitTestSource);
    if (hits.length > 0) {
      const pose = hits[0].getPose(refSpace);
      if (pose) {
        if (!hasPlaneDetection) {
          // Show reticle until first placement
          if (!firstHitPlaced) {
            reticle.visible = true;
            reticle.matrix.fromArray(Array.from(pose.transform.matrix));
            setStatus('Surface found — tap to place tiles');
          } else {
            reticle.visible = false;
          }
        } else if (trackedPlanes.size === 0) {
          // Plane detection exists but no planes yet — show scanning reticle
          reticle.visible = true;
          reticle.matrix.fromArray(Array.from(pose.transform.matrix));
          setStatus('Scanning for surfaces…');
        }
      }
    } else if (!hasPlaneDetection && !firstHitPlaced) {
      reticle.visible = false;
      setStatus('Move phone slowly over surfaces…');
    }
  }

  renderer.render(scene, camera);
}

// ─── Tap to place (hit-test fallback only) ────────────────────────────────
function onTapPlace(e) {
  if (hasPlaneDetection || isLocked || firstHitPlaced) return;
  if (!reticle.visible) return;

  // Place tile at reticle's current world position
  const mat4 = reticle.matrix.clone();
  anchorMesh = createAnchorMesh(mat4.elements);
  scene.add(anchorMesh);
  firstHitPlaced = true;

  document.getElementById('scan-ring').style.opacity = '0';
  reticle.visible = false;
  setStatus('✨ Tile placed — anchored to surface!');
}

// ─── UI setup ─────────────────────────────────────────────────────────────
function setupUI() {
  // Tile picker
  document.querySelectorAll('.tile-thumb').forEach(el => {
    el.addEventListener('click', () => {
      document.querySelectorAll('.tile-thumb').forEach(t => t.classList.remove('active'));
      el.classList.add('active');
      currentTileIdx = parseInt(el.dataset.idx);
      rebuildAllPlaneMeshes();
    });
  });

  // Surface mode toggle
  document.querySelectorAll('.mode-btn').forEach(el => {
    el.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      el.classList.add('active');
      surfaceMode = el.dataset.mode;
      // Remove planes that don't match new mode
      for (const [plane, data] of trackedPlanes) {
        const isHoriz = data.type === 'floor';
        const isVert  = data.type === 'wall';
        const keep = surfaceMode === 'both' ||
                     (surfaceMode === 'floor' && isHoriz) ||
                     (surfaceMode === 'wall' && isVert);
        if (!keep) {
          scene.remove(data.mesh);
          data.mesh.geometry.dispose();
          if (data.mesh.material.map) data.mesh.material.map.dispose();
          data.mesh.material.dispose();
          trackedPlanes.delete(plane);
        }
      }
    });
  });

  // Tile size
  document.getElementById('tile-size-slider').addEventListener('input', () => {
    rebuildAllPlaneMeshes();
  });

  // Opacity
  document.getElementById('opacity-slider').addEventListener('input', e => {
    tileOpacity = parseFloat(e.target.value) / 100;
    for (const [, data] of trackedPlanes) {
      data.mesh.material.opacity = tileOpacity;
    }
    if (anchorMesh) anchorMesh.material.opacity = tileOpacity;
  });

  // Tap to place (fallback)
  document.getElementById('hud').addEventListener('click', onTapPlace);
}

function setStatus(msg) {
  const el = document.getElementById('status');
  if (el) el.textContent = msg;
}

function updatePlaneCount(floors, walls) {
  const el = document.getElementById('plane-count');
  if (!el) return;
  if (floors + walls === 0) { el.textContent = ''; return; }
  const parts = [];
  if (floors > 0) parts.push(`${floors} floor`);
  if (walls > 0)  parts.push(`${walls} wall`);
  el.textContent = parts.join(' · ');
}

function showNotSupported() {
  document.getElementById('landing').style.display = 'none';
  document.getElementById('not-supported').style.display = 'flex';
}

window.addEventListener('resize', () => {
  if (renderer) renderer.setSize(window.innerWidth, window.innerHeight);
});
