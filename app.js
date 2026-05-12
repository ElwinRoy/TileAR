import * as THREE from 'three';

// ─── State ────────────────────────────────────────────────────────────────
let renderer, scene, camera, gl;
let xrSession = null;
let hitTestSource = null;
let hitTestSourceRequested = false;
let reticle;
let glBinding = null;
let depthOccluder = null;

let isLocked = false;
let currentTileIdx = 0;
let tileOpacity = 0.85;
let surfaceMode = 'both';

const trackedPlanes = new Map();
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

function makeTileTexture(idx) {
  const SZ = 512, GROUT = 6, N = 2;
  const cv = document.createElement('canvas');
  cv.width = cv.height = SZ;
  const ctx = cv.getContext('2d');
  const t = TILES[idx];
  const tw = SZ / N;
  for (let row = 0; row < N; row++) {
    for (let col = 0; col < N; col++) {
      const x = col*tw + GROUT/2, y = row*tw + GROUT/2;
      const w = tw - GROUT, h = tw - GROUT;
      const grd = ctx.createLinearGradient(x,y,x+w,y+h);
      grd.addColorStop(0, lighten(t.base, 0.12));
      grd.addColorStop(0.5, t.base);
      grd.addColorStop(1, darken(t.base, 0.10));
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

function getTileRepeatSize() {
  const slider = document.getElementById('tile-size-slider');
  const tileM = parseFloat(slider.value) * 0.05 + 0.10;
  return tileM * 2;
}

// ─── Depth Occlusion (WebXR Depth API) ────────────────────────────────────
// Writes real-world depth to z-buffer so tiles behind objects get occluded
function initDepthOcclusion() {
  if (!gl) return;
  const isWGL2 = typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext;
  if (!isWGL2) return;

  const vsrc = `#version 300 es
    in vec2 aPos;
    out vec2 vUv;
    void main() { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;
  const fsrc = `#version 300 es
    precision highp float;
    uniform sampler2D uDepthTex;
    uniform float uRawToMeters;
    in vec2 vUv;
    out vec4 outColor;
    void main() {
      float rawDepth = texture(uDepthTex, vUv).r;
      float depthM = rawDepth * uRawToMeters;
      if (depthM <= 0.0 || depthM > 20.0) discard;
      float near = 0.01, far = 100.0;
      gl_FragDepth = (far * (depthM - near)) / (depthM * (far - near));
      outColor = vec4(0.0);
    }`;

  const vs = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vs, vsrc); gl.compileShader(vs);
  const fs = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fs, fsrc); gl.compileShader(fs);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { console.warn('Depth shader fail'); return; }

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const loc = gl.getAttribLocation(prog, 'aPos');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  depthOccluder = {
    prog, vao,
    uDepthTex: gl.getUniformLocation(prog, 'uDepthTex'),
    uRawToMeters: gl.getUniformLocation(prog, 'uRawToMeters'),
  };
}

function renderDepthOcclusionPass(depthTexture, rawToMeters) {
  if (!depthOccluder || !depthTexture) return;
  gl.useProgram(depthOccluder.prog);
  gl.bindVertexArray(depthOccluder.vao);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, depthTexture);
  gl.uniform1i(depthOccluder.uDepthTex, 0);
  gl.uniform1f(depthOccluder.uRawToMeters, rawToMeters);
  // Write depth only, no color
  gl.colorMask(false, false, false, false);
  gl.depthMask(true);
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.ALWAYS);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  // Restore
  gl.colorMask(true, true, true, true);
  gl.depthFunc(gl.LEQUAL);
  gl.bindVertexArray(null);
}

// ─── Plane geometry from XRPlane polygon ──────────────────────────────────
function createPlaneGeometry(polygon) {
  if (!polygon || polygon.length < 3) return null;
  const shape = new THREE.Shape();
  shape.moveTo(polygon[0].x, polygon[0].z);
  for (let i = 1; i < polygon.length; i++) shape.lineTo(polygon[i].x, polygon[i].z);
  const geometry = new THREE.ShapeGeometry(shape);
  geometry.rotateX(-Math.PI / 2);
  const repeatSize = getTileRepeatSize();
  const pos = geometry.attributes.position;
  const uvs = new Float32Array(pos.count * 2);
  for (let i = 0; i < pos.count; i++) {
    uvs[i * 2] = pos.getX(i) / repeatSize;
    uvs[i * 2 + 1] = pos.getZ(i) / repeatSize;
  }
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  return geometry;
}

function createTileMaterial() {
  const tex = makeTileTexture(currentTileIdx);
  return new THREE.MeshBasicMaterial({
    map: tex, transparent: true, opacity: tileOpacity,
    depthWrite: false, depthTest: true,
    polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
    side: THREE.DoubleSide,
  });
}

function createPlaneTileMesh(plane, poseMatrix) {
  const geo = createPlaneGeometry(plane.polygon);
  if (!geo) return null;
  const mat = createTileMaterial();
  const mesh = new THREE.Mesh(geo, mat);
  mesh.matrixAutoUpdate = false;
  mesh.matrix.fromArray(poseMatrix);
  mesh.renderOrder = 1;
  mat.opacity = 0;
  return mesh;
}

function updatePlaneMeshGeometry(mesh, polygon) {
  const newGeo = createPlaneGeometry(polygon);
  if (!newGeo) return;
  mesh.geometry.dispose();
  mesh.geometry = newGeo;
}

function rebuildAllPlaneMeshes() {
  for (const [plane, data] of trackedPlanes) {
    const oldMesh = data.mesh;
    const geo = createPlaneGeometry(plane.polygon);
    if (!geo) continue;
    const mat = createTileMaterial();
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
  if (anchorMesh) rebuildAnchorMesh();
}

function createAnchorMesh(poseMatrix) {
  const sizeM = parseFloat(document.getElementById('tile-size-slider').value) * 0.55;
  const tex = makeTileTexture(currentTileIdx);
  const rep = sizeM / getTileRepeatSize();
  tex.repeat.set(rep, rep);
  const mat = new THREE.MeshBasicMaterial({
    map: tex, transparent: true, opacity: tileOpacity,
    depthWrite: false, depthTest: true,
    polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(sizeM, sizeM), mat);
  mesh.matrixAutoUpdate = false;
  mesh.matrix.fromArray(poseMatrix);
  mesh.renderOrder = 1;
  return mesh;
}

function rebuildAnchorMesh() {
  if (!anchorMesh) return;
  const saved = anchorMesh.matrix.clone();
  scene.remove(anchorMesh);
  anchorMesh.geometry.dispose();
  if (anchorMesh.material.map) anchorMesh.material.map.dispose();
  anchorMesh.material.dispose();
  anchorMesh = createAnchorMesh(saved.elements);
  scene.add(anchorMesh);
}

// ─── Controls ─────────────────────────────────────────────────────────────
function toggleLock() {
  isLocked = !isLocked;
  const btn = document.getElementById('lock-btn');
  if (isLocked) {
    btn.textContent = '🔓 Unlock'; btn.classList.add('active-lock');
    document.getElementById('lock-badge').style.display = 'block';
    setStatus('Tiles locked — walk around to inspect!');
  } else {
    btn.textContent = '🔒 Lock'; btn.classList.remove('active-lock');
    document.getElementById('lock-badge').style.display = 'none';
    setStatus('Scanning for surfaces…');
  }
}
window.toggleLock = toggleLock;

function resetAR() {
  for (const [, data] of trackedPlanes) {
    scene.remove(data.mesh); data.mesh.geometry.dispose();
    if (data.mesh.material.map) data.mesh.material.map.dispose();
    data.mesh.material.dispose();
  }
  trackedPlanes.clear();
  if (anchorMesh) {
    scene.remove(anchorMesh); anchorMesh.geometry.dispose();
    if (anchorMesh.material.map) anchorMesh.material.map.dispose();
    anchorMesh.material.dispose(); anchorMesh = null;
  }
  firstHitPlaced = false; isLocked = false;
  document.getElementById('lock-btn').textContent = '🔒 Lock';
  document.getElementById('lock-btn').classList.remove('active-lock');
  document.getElementById('scan-ring').style.opacity = '1';
  document.getElementById('lock-badge').style.display = 'none';
  reticle.visible = false;
  setStatus('Scanning for surfaces…');
  updatePlaneCount(0, 0);
}
window.resetAR = resetAR;

// ─── Three.js + Depth Init ───────────────────────────────────────────────
function initThree() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera();
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.xr.enabled = true;
  renderer.setClearColor(0x000000, 0);
  renderer.sortObjects = true;
  document.getElementById('canvas-container').appendChild(renderer.domElement);
  gl = renderer.getContext();

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

  // Init depth occlusion shader
  initDepthOcclusion();
}

// ─── AR session ───────────────────────────────────────────────────────────
async function startAR() {
  if (!navigator.xr) { showNotSupported(); return; }
  const supported = await navigator.xr.isSessionSupported('immersive-ar').catch(() => false);
  if (!supported) { showNotSupported(); return; }

  document.getElementById('landing').style.display = 'none';
  initThree();

  // Request depth-sensing for object occlusion
  try {
    xrSession = await navigator.xr.requestSession('immersive-ar', {
      requiredFeatures: ['hit-test'],
      optionalFeatures: ['dom-overlay', 'plane-detection', 'local-floor', 'depth-sensing'],
      domOverlay: { root: document.getElementById('hud') },
      depthSensing: {
        usagePreference: ['gpu-optimized', 'cpu-optimized'],
        dataFormatPreference: ['luminance-alpha', 'float32'],
      },
    });
  } catch (e) {
    try {
      xrSession = await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['hit-test'],
        optionalFeatures: ['dom-overlay', 'plane-detection', 'local-floor'],
        domOverlay: { root: document.getElementById('hud') },
      });
    } catch (e2) { showNotSupported(); return; }
  }

  // Create WebGL binding for depth texture access
  try { glBinding = new XRWebGLBinding(xrSession, gl); } catch (e) { glBinding = null; }

  renderer.xr.setReferenceSpaceType('local-floor');
  await renderer.xr.setSession(xrSession);

  xrSession.addEventListener('end', () => {
    document.getElementById('hud').style.display = 'none';
    document.getElementById('landing').style.display = 'flex';
    hitTestSource = null; hitTestSourceRequested = false;
    trackedPlanes.clear();
    anchorMesh = null; firstHitPlaced = false;
    isLocked = false; hasPlaneDetection = false; glBinding = null;
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

  if (!hitTestSourceRequested) {
    frame.session.requestReferenceSpace('viewer').then(vs => {
      frame.session.requestHitTestSource({ space: vs })
        .then(src => { hitTestSource = src; })
        .catch(() => setStatus('Hit-test not available'));
    });
    hitTestSourceRequested = true;
  }

  // ── DEPTH OCCLUSION PASS ──────────────────────────────────────────────
  // Writes real-world depth to z-buffer BEFORE tiles render.
  // Objects on the floor (chairs, shoes, etc.) will occlude the tiles.
  let depthPassRan = false;
  if (glBinding && depthOccluder) {
    const viewerPose = frame.getViewerPose(refSpace);
    if (viewerPose) {
      for (const view of viewerPose.views) {
        try {
          const depthInfo = glBinding.getDepthInformation(view);
          if (depthInfo && depthInfo.texture) {
            renderDepthOcclusionPass(depthInfo.texture, depthInfo.rawValueToMeters);
            depthPassRan = true;
          }
        } catch (e) { /* depth not available this frame */ }
      }
    }
    // Reset Three.js GL state after raw WebGL calls
    renderer.state.reset();
  }

  // ── PLANE DETECTION ───────────────────────────────────────────────────
  if (frame.detectedPlanes && frame.detectedPlanes.size > 0) {
    hasPlaneDetection = true;
    const currentPlanes = new Set(frame.detectedPlanes);

    for (const [plane, data] of trackedPlanes) {
      if (!currentPlanes.has(plane)) {
        scene.remove(data.mesh); data.mesh.geometry.dispose();
        if (data.mesh.material.map) data.mesh.material.map.dispose();
        data.mesh.material.dispose(); trackedPlanes.delete(plane);
      }
    }

    if (!isLocked) {
      let floorCount = 0, wallCount = 0;

      for (const plane of frame.detectedPlanes) {
        const isHoriz = plane.orientation === 'horizontal';
        const isVert = plane.orientation === 'vertical';
        if (surfaceMode === 'floor' && !isHoriz) continue;
        if (surfaceMode === 'wall' && !isVert) continue;
        if (!isHoriz && !isVert) continue;
        if (isHoriz) floorCount++; else wallCount++;

        const pose = frame.getPose(plane.planeSpace, refSpace);
        if (!pose) continue;

        if (trackedPlanes.has(plane)) {
          const data = trackedPlanes.get(plane);
          data.mesh.matrix.fromArray(pose.transform.matrix);
          const changeTime = plane.lastChangedTime || 0;
          if (changeTime > data.lastUpdate) {
            updatePlaneMeshGeometry(data.mesh, plane.polygon);
            data.lastUpdate = changeTime;
          }
          if (data.mesh.material.opacity < tileOpacity) {
            data.mesh.material.opacity = Math.min(tileOpacity, data.mesh.material.opacity + 0.02);
          }
        } else {
          const mesh = createPlaneTileMesh(plane, pose.transform.matrix);
          if (!mesh) continue;
          scene.add(mesh);
          trackedPlanes.set(plane, {
            mesh, lastUpdate: plane.lastChangedTime || 0,
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

  // ── HIT-TEST FALLBACK ─────────────────────────────────────────────────
  if (hitTestSource && refSpace) {
    const hits = frame.getHitTestResults(hitTestSource);
    if (hits.length > 0) {
      const pose = hits[0].getPose(refSpace);
      if (pose) {
        if (!hasPlaneDetection) {
          if (!firstHitPlaced) {
            reticle.visible = true;
            reticle.matrix.fromArray(Array.from(pose.transform.matrix));
            setStatus('Surface found — tap to place tiles');
          } else { reticle.visible = false; }
        } else if (trackedPlanes.size === 0) {
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

  // Only skip depth clear when depth occlusion pass actually wrote depth values
  if (depthPassRan) {
    renderer.autoClearDepth = false;
    renderer.render(scene, camera);
    renderer.autoClearDepth = true;
  } else {
    renderer.render(scene, camera);
  }
}

function onTapPlace() {
  if (hasPlaneDetection || isLocked || firstHitPlaced || !reticle.visible) return;
  anchorMesh = createAnchorMesh(reticle.matrix.clone().elements);
  scene.add(anchorMesh); firstHitPlaced = true;
  document.getElementById('scan-ring').style.opacity = '0';
  reticle.visible = false;
  setStatus('✨ Tile placed — anchored to surface!');
}

// ─── UI ───────────────────────────────────────────────────────────────────
function setupUI() {
  document.querySelectorAll('.tile-thumb').forEach(el => {
    el.addEventListener('click', () => {
      document.querySelectorAll('.tile-thumb').forEach(t => t.classList.remove('active'));
      el.classList.add('active');
      currentTileIdx = parseInt(el.dataset.idx);
      rebuildAllPlaneMeshes();
    });
  });
  document.querySelectorAll('.mode-btn').forEach(el => {
    el.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      el.classList.add('active');
      surfaceMode = el.dataset.mode;
      for (const [plane, data] of trackedPlanes) {
        const keep = surfaceMode === 'both' ||
          (surfaceMode === 'floor' && data.type === 'floor') ||
          (surfaceMode === 'wall' && data.type === 'wall');
        if (!keep) {
          scene.remove(data.mesh); data.mesh.geometry.dispose();
          if (data.mesh.material.map) data.mesh.material.map.dispose();
          data.mesh.material.dispose(); trackedPlanes.delete(plane);
        }
      }
    });
  });
  document.getElementById('tile-size-slider').addEventListener('input', () => rebuildAllPlaneMeshes());
  document.getElementById('opacity-slider').addEventListener('input', e => {
    tileOpacity = parseFloat(e.target.value) / 100;
    for (const [, data] of trackedPlanes) data.mesh.material.opacity = tileOpacity;
    if (anchorMesh) anchorMesh.material.opacity = tileOpacity;
  });
  document.getElementById('hud').addEventListener('click', onTapPlace);
}

function setStatus(msg) { const el = document.getElementById('status'); if (el) el.textContent = msg; }
function updatePlaneCount(f, w) {
  const el = document.getElementById('plane-count'); if (!el) return;
  if (f + w === 0) { el.textContent = ''; return; }
  const p = []; if (f > 0) p.push(`${f} floor`); if (w > 0) p.push(`${w} wall`);
  el.textContent = p.join(' · ');
}
function showNotSupported() {
  document.getElementById('landing').style.display = 'none';
  document.getElementById('not-supported').style.display = 'flex';
}
window.addEventListener('resize', () => { if (renderer) renderer.setSize(window.innerWidth, window.innerHeight); });
