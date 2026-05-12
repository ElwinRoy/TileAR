import * as THREE from 'three';

let renderer, scene, camera, gl;
let xrSession = null, hitTestSource = null, hitTestSourceRequested = false;
let reticle, isLocked = false, currentTileIdx = 0, tileOpacity = 0.85;
let surfaceMode = 'both', hasPlaneDetection = false;
let glBinding = null, depthOccluder = null;

// SINGLE mesh per surface type — no more overlapping
let floorMesh = null, wallMesh = null;
let floorPlaneRef = null, wallPlaneRef = null; // track which XRPlane we're anchored to
let floorY = null;

// Fallback
let firstHitPlaced = false, anchorMesh = null;

const TILES = [
  { name:'Gold',       base:'#c8a84b', grout:'#6b4c1e', sheen:0.30 },
  { name:'Marble',     base:'#ddd8d0', grout:'#9a9590', sheen:0.50 },
  { name:'Charcoal',   base:'#2e2e2e', grout:'#111111', sheen:0.18 },
  { name:'Terracotta', base:'#c97b4b', grout:'#7a3e1e', sheen:0.14 },
  { name:'Green',      base:'#3d7a52', grout:'#1a3d28', sheen:0.10 },
];

const cl=v=>Math.max(0,Math.min(255,v));
const hexToRgb=h=>{const n=parseInt(h.replace('#',''),16);return[(n>>16)&255,(n>>8)&255,n&255];};
const rgbToHex=(r,g,b)=>'#'+[r,g,b].map(v=>Math.round(v).toString(16).padStart(2,'0')).join('');
const lighten=(h,a)=>{const[r,g,b]=hexToRgb(h);return rgbToHex(cl(r+255*a),cl(g+255*a),cl(b+255*a));};
const darken=(h,a)=>lighten(h,-a);

function rRect(ctx,x,y,w,h,r){
  ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.quadraticCurveTo(x+w,y,x+w,y+r);
  ctx.lineTo(x+w,y+h-r);ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
  ctx.lineTo(x+r,y+h);ctx.quadraticCurveTo(x,y+h,x,y+h-r);
  ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);ctx.closePath();
}
function addGrain(ctx,x,y,w,h,amt){
  const id=ctx.getImageData(x,y,w,h),d=id.data;
  for(let i=0;i<d.length;i+=4){const n=(Math.random()-.5)*255*amt;d[i]=cl(d[i]+n);d[i+1]=cl(d[i+1]+n);d[i+2]=cl(d[i+2]+n);}
  ctx.putImageData(id,x,y);
}

const texCache=new Map();
function makeTileTexture(idx){
  if(texCache.has(idx)) return texCache.get(idx).clone();
  const SZ=512,GR=6,N=2,cv=document.createElement('canvas');cv.width=cv.height=SZ;
  const ctx=cv.getContext('2d'),t=TILES[idx],tw=SZ/N;
  for(let r=0;r<N;r++)for(let c=0;c<N;c++){
    const x=c*tw+GR/2,y=r*tw+GR/2,w=tw-GR,h=tw-GR;
    const grd=ctx.createLinearGradient(x,y,x+w,y+h);
    grd.addColorStop(0,lighten(t.base,.12));grd.addColorStop(.5,t.base);grd.addColorStop(1,darken(t.base,.10));
    ctx.fillStyle=grd;rRect(ctx,x,y,w,h,4);ctx.fill();addGrain(ctx,x,y,w,h,.04);
    const hi=ctx.createLinearGradient(x,y,x+w*.6,y+h*.6);
    hi.addColorStop(0,`rgba(255,255,255,${t.sheen})`);hi.addColorStop(1,'rgba(255,255,255,0)');
    ctx.fillStyle=hi;rRect(ctx,x,y,w,h,4);ctx.fill();
  }
  ctx.fillStyle=t.grout;
  for(let i=0;i<=N;i++){ctx.fillRect(0,i*tw-GR/2,SZ,GR);ctx.fillRect(i*tw-GR/2,0,GR,SZ);}
  const tex=new THREE.CanvasTexture(cv);tex.wrapS=tex.wrapT=THREE.RepeatWrapping;
  texCache.set(idx,tex);return tex.clone();
}

function getTileM(){return parseFloat(document.getElementById('tile-size-slider').value)*0.05+0.10;}

// ── Depth occlusion setup (WebGL2) ──
function initDepthOcclusion(){
  if(!gl||!(gl instanceof WebGL2RenderingContext)) return;
  const vs=gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vs,`#version 300 es
    in vec2 aP;out vec2 vU;void main(){vU=aP*.5+.5;gl_Position=vec4(aP,0,1);}`);
  gl.compileShader(vs);
  const fs=gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fs,`#version 300 es
    precision highp float;uniform sampler2D uD;uniform float uR;in vec2 vU;out vec4 oC;
    void main(){float r=texture(uD,vU).r;float d=r*uR;if(d<=0.||d>20.)discard;
    float n=.01,f=100.;gl_FragDepth=(f*(d-n))/(d*(f-n));oC=vec4(0);}`);
  gl.compileShader(fs);
  const pg=gl.createProgram();gl.attachShader(pg,vs);gl.attachShader(pg,fs);gl.linkProgram(pg);
  if(!gl.getProgramParameter(pg,gl.LINK_STATUS))return;
  const buf=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buf);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);
  const vao=gl.createVertexArray();gl.bindVertexArray(vao);
  const loc=gl.getAttribLocation(pg,'aP');gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);gl.bindVertexArray(null);
  depthOccluder={pg,vao,uD:gl.getUniformLocation(pg,'uD'),uR:gl.getUniformLocation(pg,'uR')};
}
function renderDepthPass(tex,raw){
  if(!depthOccluder||!tex)return;
  gl.useProgram(depthOccluder.pg);gl.bindVertexArray(depthOccluder.vao);
  gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,tex);
  gl.uniform1i(depthOccluder.uD,0);gl.uniform1f(depthOccluder.uR,raw);
  gl.colorMask(false,false,false,false);gl.depthMask(true);
  gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.ALWAYS);
  gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
  gl.colorMask(true,true,true,true);gl.depthFunc(gl.LEQUAL);gl.bindVertexArray(null);
}

// ── Compute plane area from polygon ──
function planeArea(polygon){
  if(!polygon||polygon.length<3)return 0;
  let a=0;
  for(let i=0,n=polygon.length;i<n;i++){
    const p=polygon[i],q=polygon[(i+1)%n];a+=p.x*q.z-q.x*p.z;
  }
  return Math.abs(a)*0.5;
}
// Get bounding box of polygon in plane-local XZ
function planeBounds(polygon){
  let mnX=Infinity,mxX=-Infinity,mnZ=Infinity,mxZ=-Infinity;
  for(const p of polygon){
    if(p.x<mnX)mnX=p.x;if(p.x>mxX)mxX=p.x;
    if(p.z<mnZ)mnZ=p.z;if(p.z>mxZ)mxZ=p.z;
  }
  return {w:mxX-mnX,h:mxZ-mnZ,cx:(mnX+mxX)/2,cz:(mnZ+mxZ)/2};
}

// ── Build single tile mesh for a surface ──
function buildSurfaceMesh(sizeX, sizeZ){
  const tileM=getTileM();
  const tex=makeTileTexture(currentTileIdx);
  const repX=sizeX/(tileM*2), repZ=sizeZ/(tileM*2);
  tex.repeat.set(repX,repZ);
  const mat=new THREE.MeshBasicMaterial({
    map:tex,transparent:true,opacity:tileOpacity,
    depthWrite:false,depthTest:true,
    polygonOffset:true,polygonOffsetFactor:-4,polygonOffsetUnits:-4,
    side:THREE.DoubleSide,
  });
  // PlaneGeometry lies in XY by default; rotate to XZ for horizontal surfaces
  const geo=new THREE.PlaneGeometry(sizeX,sizeZ,1,1);
  const mesh=new THREE.Mesh(geo,mat);
  mesh.matrixAutoUpdate=false;
  mesh.renderOrder=1;
  return mesh;
}

// ── Create or update the single floor tile ──
function updateFloorTile(pose, sizeX, sizeZ, cx, cz){
  if(isLocked) return;
  const mat4=new THREE.Matrix4().fromArray(pose.transform.matrix);
  // Decompose to get position and quaternion
  const pos=new THREE.Vector3(), quat=new THREE.Quaternion(), scl=new THREE.Vector3();
  mat4.decompose(pos,quat,scl);

  if(!floorMesh){
    // First time: create mesh
    floorMesh=buildSurfaceMesh(Math.max(sizeX,4),Math.max(sizeZ,4));
    scene.add(floorMesh);
    floorY=pos.y;
  }

  // Update floor Y (use direct value from plane pose — it IS the real floor)
  floorY=pos.y;

  // Build transform: keep plane orientation but use a large coverage
  // Position at plane origin, flat on the floor
  const curSizeX=Math.max(sizeX+1,4); // extend 0.5m beyond detected edges
  const curSizeZ=Math.max(sizeZ+1,4);

  // Rebuild geometry if size grew significantly
  const curGeo=floorMesh.geometry;
  const oldW=curGeo.parameters.width, oldH=curGeo.parameters.height;
  if(curSizeX>oldW*1.1||curSizeZ>oldH*1.1){
    floorMesh.geometry.dispose();
    floorMesh.geometry=new THREE.PlaneGeometry(curSizeX,curSizeZ,1,1);
    // Update texture repeat
    const tileM=getTileM();
    floorMesh.material.map.repeat.set(curSizeX/(tileM*2),curSizeZ/(tileM*2));
  }

  // Set mesh transform from plane pose (world-anchored, flushed to real floor)
  // Use the full plane pose matrix so the tile lies exactly on the detected surface
  floorMesh.matrix.copy(mat4);
}

// ── Create or update the single wall tile ──
function updateWallTile(pose, sizeX, sizeZ){
  if(isLocked) return;
  const mat4=new THREE.Matrix4().fromArray(pose.transform.matrix);

  if(!wallMesh){
    wallMesh=buildSurfaceMesh(Math.max(sizeX,3),Math.max(sizeZ,3));
    scene.add(wallMesh);
  }

  const curSizeX=Math.max(sizeX+0.5,3);
  const curSizeZ=Math.max(sizeZ+0.5,3);
  const curGeo=wallMesh.geometry;
  const oldW=curGeo.parameters.width,oldH=curGeo.parameters.height;
  if(curSizeX>oldW*1.1||curSizeZ>oldH*1.1){
    wallMesh.geometry.dispose();
    wallMesh.geometry=new THREE.PlaneGeometry(curSizeX,curSizeZ,1,1);
    const tileM=getTileM();
    wallMesh.material.map.repeat.set(curSizeX/(tileM*2),curSizeZ/(tileM*2));
  }
  wallMesh.matrix.copy(mat4);
}

// ── Rebuild meshes on tile/size change ──
function rebuildMeshes(){
  if(floorMesh){
    const saved=floorMesh.matrix.clone();
    const gp=floorMesh.geometry.parameters;
    scene.remove(floorMesh);floorMesh.geometry.dispose();
    if(floorMesh.material.map)floorMesh.material.map.dispose();floorMesh.material.dispose();
    floorMesh=buildSurfaceMesh(gp.width,gp.height);
    floorMesh.matrix.copy(saved);scene.add(floorMesh);
  }
  if(wallMesh){
    const saved=wallMesh.matrix.clone();
    const gp=wallMesh.geometry.parameters;
    scene.remove(wallMesh);wallMesh.geometry.dispose();
    if(wallMesh.material.map)wallMesh.material.map.dispose();wallMesh.material.dispose();
    wallMesh=buildSurfaceMesh(gp.width,gp.height);
    wallMesh.matrix.copy(saved);scene.add(wallMesh);
  }
  if(anchorMesh) rebuildAnchorMesh();
}

// ── Fallback anchor mesh ──
function createAnchorMesh(pm){
  const s=parseFloat(document.getElementById('tile-size-slider').value)*0.55;
  const tex=makeTileTexture(currentTileIdx);const rep=s/(getTileM()*2);tex.repeat.set(rep,rep);
  const mat=new THREE.MeshBasicMaterial({map:tex,transparent:true,opacity:tileOpacity,
    depthWrite:false,depthTest:true,polygonOffset:true,polygonOffsetFactor:-4,polygonOffsetUnits:-4,side:THREE.DoubleSide});
  const mesh=new THREE.Mesh(new THREE.PlaneGeometry(s,s),mat);
  mesh.matrixAutoUpdate=false;mesh.matrix.fromArray(pm);mesh.renderOrder=1;return mesh;
}
function rebuildAnchorMesh(){
  if(!anchorMesh)return;const sv=anchorMesh.matrix.clone();
  scene.remove(anchorMesh);anchorMesh.geometry.dispose();
  if(anchorMesh.material.map)anchorMesh.material.map.dispose();anchorMesh.material.dispose();
  anchorMesh=createAnchorMesh(sv.elements);scene.add(anchorMesh);
}

// ── Controls ──
function toggleLock(){
  isLocked=!isLocked;const btn=document.getElementById('lock-btn');
  if(isLocked){btn.textContent='🔓 Unlock';btn.classList.add('active-lock');
    document.getElementById('lock-badge').style.display='block';setStatus('Tiles locked');
  }else{btn.textContent='🔒 Lock';btn.classList.remove('active-lock');
    document.getElementById('lock-badge').style.display='none';setStatus('Scanning…');}
}
window.toggleLock=toggleLock;

function resetAR(){
  if(floorMesh){scene.remove(floorMesh);floorMesh.geometry.dispose();
    if(floorMesh.material.map)floorMesh.material.map.dispose();floorMesh.material.dispose();floorMesh=null;}
  if(wallMesh){scene.remove(wallMesh);wallMesh.geometry.dispose();
    if(wallMesh.material.map)wallMesh.material.map.dispose();wallMesh.material.dispose();wallMesh=null;}
  floorPlaneRef=null;wallPlaneRef=null;floorY=null;
  if(anchorMesh){scene.remove(anchorMesh);anchorMesh.geometry.dispose();
    if(anchorMesh.material.map)anchorMesh.material.map.dispose();anchorMesh.material.dispose();anchorMesh=null;}
  firstHitPlaced=false;isLocked=false;
  document.getElementById('lock-btn').textContent='🔒 Lock';
  document.getElementById('lock-btn').classList.remove('active-lock');
  document.getElementById('scan-ring').style.opacity='1';
  document.getElementById('lock-badge').style.display='none';
  reticle.visible=false;setStatus('Scanning for surfaces…');
  const pc=document.getElementById('plane-count');if(pc)pc.textContent='';
}
window.resetAR=resetAR;

// ── Three.js ──
function initThree(){
  scene=new THREE.Scene();camera=new THREE.PerspectiveCamera();
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth,window.innerHeight);
  renderer.xr.enabled=true;renderer.setClearColor(0x000000,0);
  renderer.sortObjects=true;renderer.autoClearDepth=false;
  document.getElementById('canvas-container').appendChild(renderer.domElement);
  gl=renderer.getContext();
  scene.add(new THREE.AmbientLight(0xffffff,1));
  const rG=new THREE.RingGeometry(.05,.08,36);
  rG.applyMatrix4(new THREE.Matrix4().makeRotationX(-Math.PI/2));
  reticle=new THREE.Mesh(rG,new THREE.MeshBasicMaterial({color:0xf0d080,side:THREE.DoubleSide,depthWrite:false,transparent:true,opacity:.85}));
  reticle.matrixAutoUpdate=false;reticle.visible=false;scene.add(reticle);
  initDepthOcclusion();
}

async function startAR(){
  if(!navigator.xr){showNotSupported();return;}
  if(!await navigator.xr.isSessionSupported('immersive-ar').catch(()=>false)){showNotSupported();return;}
  document.getElementById('landing').style.display='none';initThree();
  try{xrSession=await navigator.xr.requestSession('immersive-ar',{
    requiredFeatures:['hit-test'],
    optionalFeatures:['dom-overlay','plane-detection','local-floor','depth-sensing'],
    domOverlay:{root:document.getElementById('hud')},
    depthSensing:{usagePreference:['gpu-optimized','cpu-optimized'],dataFormatPreference:['luminance-alpha','float32']},
  });}catch(e){
    try{xrSession=await navigator.xr.requestSession('immersive-ar',{
      requiredFeatures:['hit-test'],optionalFeatures:['dom-overlay','plane-detection','local-floor'],
      domOverlay:{root:document.getElementById('hud')},
    });}catch(e2){showNotSupported();return;}
  }
  try{glBinding=new XRWebGLBinding(xrSession,gl);}catch(e){glBinding=null;}
  renderer.xr.setReferenceSpaceType('local-floor');
  await renderer.xr.setSession(xrSession);
  xrSession.addEventListener('end',()=>{
    document.getElementById('hud').style.display='none';
    document.getElementById('landing').style.display='flex';
    hitTestSource=null;hitTestSourceRequested=false;
    floorMesh=null;wallMesh=null;floorPlaneRef=null;wallPlaneRef=null;
    anchorMesh=null;firstHitPlaced=false;isLocked=false;hasPlaneDetection=false;glBinding=null;
  });
  document.getElementById('hud').style.display='block';
  renderer.setAnimationLoop(onXRFrame);setupUI();
}
window.startAR=startAR;

// ── XR Frame ──
function onXRFrame(time,frame){
  if(!frame)return;
  const ref=renderer.xr.getReferenceSpace();

  if(!hitTestSourceRequested){
    frame.session.requestReferenceSpace('viewer').then(vs=>{
      frame.session.requestHitTestSource({space:vs}).then(s=>{hitTestSource=s;}).catch(()=>{});
    });hitTestSourceRequested=true;
  }

  // Depth occlusion pass
  if(glBinding&&depthOccluder){
    const vp=frame.getViewerPose(ref);
    if(vp){for(const v of vp.views){try{
      const di=glBinding.getDepthInformation(v);
      if(di&&di.texture)renderDepthPass(di.texture,di.rawValueToMeters);
    }catch(e){}}}
    renderer.state.reset();
  }

  // ── Plane Detection: find LARGEST floor & wall plane ──
  if(frame.detectedPlanes&&frame.detectedPlanes.size>0){
    hasPlaneDetection=true;
    if(!isLocked){
      let bestFloor=null,bestFloorArea=0;
      let bestWall=null,bestWallArea=0;

      for(const plane of frame.detectedPlanes){
        const a=planeArea(plane.polygon);
        if(plane.orientation==='horizontal'&&a>bestFloorArea){bestFloor=plane;bestFloorArea=a;}
        if(plane.orientation==='vertical'&&a>bestWallArea){bestWall=plane;bestWallArea=a;}
      }

      let fc=0,wc=0;

      // FLOOR: single mesh anchored to largest horizontal plane
      if(bestFloor&&(surfaceMode==='both'||surfaceMode==='floor')){
        fc=1;
        const pose=frame.getPose(bestFloor.planeSpace,ref);
        if(pose){
          const b=planeBounds(bestFloor.polygon);
          updateFloorTile(pose,b.w,b.h,b.cx,b.cz);
          floorPlaneRef=bestFloor;
        }
      }

      // WALL: single mesh anchored to largest vertical plane
      if(bestWall&&(surfaceMode==='both'||surfaceMode==='wall')){
        wc=1;
        const pose=frame.getPose(bestWall.planeSpace,ref);
        if(pose){
          const b=planeBounds(bestWall.polygon);
          updateWallTile(pose,b.w,b.h);
          wallPlaneRef=bestWall;
        }
      }

      // Remove meshes if mode changed
      if(surfaceMode==='wall'&&floorMesh){
        scene.remove(floorMesh);floorMesh.geometry.dispose();
        if(floorMesh.material.map)floorMesh.material.map.dispose();floorMesh.material.dispose();
        floorMesh=null;floorPlaneRef=null;
      }
      if(surfaceMode==='floor'&&wallMesh){
        scene.remove(wallMesh);wallMesh.geometry.dispose();
        if(wallMesh.material.map)wallMesh.material.map.dispose();wallMesh.material.dispose();
        wallMesh=null;wallPlaneRef=null;
      }

      const pc=document.getElementById('plane-count');
      if(pc){const p=[];if(fc)p.push('Floor ✓');if(wc)p.push('Wall ✓');pc.textContent=p.join(' · ');}
      if(floorMesh||wallMesh){
        document.getElementById('scan-ring').style.opacity='0';
        reticle.visible=false;
        setStatus('✨ Tile applied to surface');
      }
    }
  }

  // Hit-test fallback
  if(hitTestSource&&ref){
    const hits=frame.getHitTestResults(hitTestSource);
    if(hits.length>0){
      const pose=hits[0].getPose(ref);
      if(pose&&!hasPlaneDetection){
        if(!firstHitPlaced){
          reticle.visible=true;reticle.matrix.fromArray(Array.from(pose.transform.matrix));
          setStatus('Surface found — tap to place tile');
        }else reticle.visible=false;
      }else if(pose&&!floorMesh&&!wallMesh){
        reticle.visible=true;reticle.matrix.fromArray(Array.from(pose.transform.matrix));
        setStatus('Scanning…');
      }
    }else if(!hasPlaneDetection&&!firstHitPlaced){reticle.visible=false;setStatus('Move phone slowly…');}
  }

  renderer.render(scene,camera);
}

function onTapPlace(){
  if(hasPlaneDetection||isLocked||firstHitPlaced||!reticle.visible)return;
  anchorMesh=createAnchorMesh(reticle.matrix.elements);
  scene.add(anchorMesh);firstHitPlaced=true;
  document.getElementById('scan-ring').style.opacity='0';reticle.visible=false;
  setStatus('✨ Tile anchored!');
}

function setupUI(){
  document.querySelectorAll('.tile-thumb').forEach(el=>{el.addEventListener('click',()=>{
    document.querySelectorAll('.tile-thumb').forEach(t=>t.classList.remove('active'));
    el.classList.add('active');currentTileIdx=parseInt(el.dataset.idx);texCache.clear();rebuildMeshes();
  });});
  document.querySelectorAll('.mode-btn').forEach(el=>{el.addEventListener('click',()=>{
    document.querySelectorAll('.mode-btn').forEach(b=>b.classList.remove('active'));
    el.classList.add('active');surfaceMode=el.dataset.mode;
  });});
  document.getElementById('tile-size-slider').addEventListener('input',()=>rebuildMeshes());
  document.getElementById('opacity-slider').addEventListener('input',e=>{
    tileOpacity=parseFloat(e.target.value)/100;
    if(floorMesh)floorMesh.material.opacity=tileOpacity;
    if(wallMesh)wallMesh.material.opacity=tileOpacity;
    if(anchorMesh)anchorMesh.material.opacity=tileOpacity;
  });
  document.getElementById('hud').addEventListener('click',onTapPlace);
}

function setStatus(m){const e=document.getElementById('status');if(e)e.textContent=m;}
function showNotSupported(){document.getElementById('landing').style.display='none';document.getElementById('not-supported').style.display='flex';}
window.addEventListener('resize',()=>{if(renderer)renderer.setSize(window.innerWidth,window.innerHeight);});
