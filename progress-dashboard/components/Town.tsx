"use client";

import { useFrame } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import {
  ROAD_WIDTH, bayState, lightState, signalLamps, stepTraffic, streetParts, vehiclePose,
  type Depot, type Part, type Town, type Traffic, type Vehicle,
} from "@/lib/world";

import { Block, Forklift, PALETTE, Pallet } from "./models";

const noRaycast = () => null;
const BOX = new THREE.BoxGeometry(1, 1, 1);
/** 單位山形屋頂：底部在 y=0、屋脊高 1，屋脊沿 x */
const ROOF = (() => {
  const shape = new THREE.Shape();
  shape.moveTo(-0.5, 0);
  shape.lineTo(0.5, 0);
  shape.lineTo(0, 1);
  shape.closePath();
  const geometry = new THREE.ExtrudeGeometry(shape, { depth: 1, bevelEnabled: false });
  geometry.translate(0, 0, -0.5);
  geometry.rotateY(Math.PI / 2);
  return geometry;
})();

/** 每帶推進一次車流模擬；必須排在讀取車輛位置的元件之前。 */
export function TrafficDriver({ traffic, paused }: { traffic: Traffic; paused: boolean }) {
  useFrame((_, dt) => { if (!paused) stepTraffic(traffic, Math.min(dt, 0.05)); });
  return null;
}

function InstancedParts({ parts, geometry, shadows = true }: { parts: Part[]; geometry: THREE.BufferGeometry; shadows?: boolean }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const matrix = new THREE.Matrix4(), rotation = new THREE.Quaternion(), color = new THREE.Color(), up = new THREE.Vector3(0, 1, 0);
    parts.forEach((part, i) => {
      matrix.compose(new THREE.Vector3(part.x, part.y, part.z), rotation.setFromAxisAngle(up, part.rotation), new THREE.Vector3(part.sx, part.sy, part.sz));
      mesh.setMatrixAt(i, matrix);
      mesh.setColorAt(i, color.set(part.color));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [parts]);
  return (
    <instancedMesh key={parts.length} ref={ref} args={[geometry, undefined, parts.length]} castShadow={shadows} receiveShadow raycast={noRaycast}>
      <meshStandardMaterial roughness={0.8} metalness={0.02} />
    </instancedMesh>
  );
}

const TRUNK = new THREE.CylinderGeometry(0.1, 0.14, 1.1, 8).translate(0, 0.55, 0);
const CROWN = new THREE.SphereGeometry(0.75, 14, 10).translate(0, 1.7, 0);
const CROWN_TOP = new THREE.SphereGeometry(0.48, 14, 10).translate(0.15, 2.25, 0.05);

/** 行道樹與城鎮樹木：三個實例化網格一次畫完（外型同 models.tsx 的 Tree）。 */
export function Trees({ trees }: { trees: { x: number; z: number; scale: number }[] }) {
  const parts = useMemo(() => {
    const make = (color: string) => trees.map((tree) => ({ x: tree.x, y: 0, z: tree.z, sx: tree.scale, sy: tree.scale, sz: tree.scale, rotation: 0, color }));
    return { trunk: make(PALETTE.trunk), crown: make(PALETTE.leaf), top: make(PALETTE.leafDark) };
  }, [trees]);
  return (
    <group>
      <InstancedParts parts={parts.trunk} geometry={TRUNK} />
      <InstancedParts parts={parts.crown} geometry={CROWN} />
      <InstancedParts parts={parts.top} geometry={CROWN_TOP} />
    </group>
  );
}

export function TownView({ town }: { town: Town }) {
  return (
    <group>
      <InstancedParts parts={town.lawns} geometry={BOX} shadows={false} />
      <InstancedParts parts={town.boxes} geometry={BOX} />
      <InstancedParts parts={town.roofs} geometry={ROOF} />
    </group>
  );
}

/** 路網：人行道、柏油、標線（中線、斑馬線、停止線）與號誌燈桿，全部實例化繪製。 */
export function StreetsView({ traffic }: { traffic: Traffic }) {
  const parts = useMemo(() => streetParts(traffic), [traffic]);
  return (
    <group>
      <InstancedParts parts={parts.sidewalks} geometry={BOX} shadows={false} />
      <InstancedParts parts={parts.asphalt} geometry={BOX} shadows={false} />
      <InstancedParts parts={parts.markings} geometry={BOX} shadows={false} />
      <InstancedParts parts={parts.poles} geometry={BOX} />
    </group>
  );
}

const LIGHT_COLORS = { green: new THREE.Color("#22c55e"), amber: new THREE.Color("#f59e0b"), red: new THREE.Color("#ef4444") };

/** 號誌燈：每個路口每個進入方向一盞，每幀依燈號變色。 */
export function TrafficLights({ traffic }: { traffic: Traffic }) {
  const lamps = useMemo(() => signalLamps(traffic), [traffic]);
  const ref = useRef<THREE.InstancedMesh>(null);
  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const matrix = new THREE.Matrix4();
    lamps.forEach((lamp, i) => {
      mesh.setMatrixAt(i, matrix.makeTranslation(lamp.x, lamp.y, lamp.z));
      mesh.setColorAt(i, LIGHT_COLORS.red);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [lamps]);
  useFrame(() => {
    const mesh = ref.current;
    if (!mesh || !lamps.length) return;
    lamps.forEach((lamp, i) => mesh.setColorAt(i, LIGHT_COLORS[lightState(traffic, lamp.node, lamp.lane)]));
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });
  return (
    <instancedMesh key={lamps.length} ref={ref} args={[LAMP_GEOMETRY, undefined, Math.max(1, lamps.length)]} count={lamps.length} raycast={noRaycast}>
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

const LAMP_GEOMETRY = new THREE.SphereGeometry(0.17, 10, 8);

const CAR_COLORS = ["#ec5a5f", "#f6c544", "#f8fafc", "#1fb5a8", "#3a4256", "#6f91ff", "#ff8a3d"];

/** 小客車與廂型車（車頭朝 +x）。 */
function CarModel({ vehicle }: { vehicle: Vehicle }) {
  const color = CAR_COLORS[vehicle.color % CAR_COLORS.length];
  const van = vehicle.kind === "van";
  return (
    <group>
      {van ? (
        <>
          <Block size={[2.5, 1.05, 1.1]} position={[0, 0.78, 0]} color={color} />
          <Block size={[0.06, 0.42, 0.9]} position={[1.26, 1.0, 0]} color="#1c2433" />
          <Block size={[1.0, 0.36, 0.04]} position={[0.55, 1.0, 0.56]} color="#1c2433" />
          <Block size={[1.0, 0.36, 0.04]} position={[0.55, 1.0, -0.56]} color="#1c2433" />
        </>
      ) : (
        <>
          <Block size={[2.1, 0.5, 1.0]} position={[0, 0.5, 0]} color={color} />
          <Block size={[1.1, 0.42, 0.9]} position={[-0.15, 0.95, 0]} color={color} />
          <Block size={[1.12, 0.3, 0.92]} position={[-0.15, 0.97, 0]} color="#2b3550" />
        </>
      )}
      {[-0.7, 0.7].map((x) => [-0.48, 0.48].map((z) => (
        <mesh key={`${x}${z}`} position={[x, 0.24, z * (van ? 1.08 : 1)]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.22, 0.22, 0.16, 12]} />
          <meshStandardMaterial color={PALETTE.dark} />
        </mesh>
      )))}
    </group>
  );
}

export function Cars({ traffic }: { traffic: Traffic }) {
  const cars = useMemo(() => traffic.vehicles.filter((vehicle) => vehicle.kind !== "truck"), [traffic]);
  const refs = useRef<(THREE.Group | null)[]>([]);
  useFrame(() => {
    cars.forEach((car, i) => {
      const group = refs.current[i];
      if (!group) return;
      const pose = vehiclePose(traffic, car);
      group.position.set(pose.x, 0, pose.z);
      group.rotation.y = pose.heading;
    });
  });
  return (
    <group>
      {cars.map((car, i) => (
        <group key={car.id} ref={(element) => { refs.current[i] = element; }}>
          <CarModel vehicle={car} />
        </group>
      ))}
    </group>
  );
}

/** 卸貨道路面：沿路徑的平帶 */
function Driveway({ path }: { path: { x: number; z: number }[] }) {
  const geometry = useMemo(() => {
    const positions: number[] = [], half = ROAD_WIDTH / 3;
    for (let k = 0; k < path.length; k++) {
      const a = path[Math.max(0, k - 1)], b = path[Math.min(path.length - 1, k + 1)];
      const dx = b.x - a.x, dz = b.z - a.z, len = Math.hypot(dx, dz) || 1;
      const nx = -dz / len, nz = dx / len;
      positions.push(path[k].x + nx * half, 0, path[k].z + nz * half, path[k].x - nx * half, 0, path[k].z - nz * half);
    }
    const index: number[] = [];
    for (let k = 0; k < path.length - 1; k++) {
      const i = k * 2;
      index.push(i, i + 2, i + 1, i + 1, i + 2, i + 3);
    }
    const result = new THREE.BufferGeometry();
    result.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    result.setIndex(index);
    result.computeVertexNormals();
    return result;
  }, [path]);
  return (
    <mesh geometry={geometry} position-y={0.004} receiveShadow raycast={noRaycast}>
      <meshStandardMaterial color="#c7cfe3" side={THREE.DoubleSide} />
    </mesh>
  );
}

const LAMP = { free: "#24b46e", moving: "#f0a623", unloading: "#e0564b" };
const DOOR_HEIGHT = 2.6, WALL_HEIGHT = 3.3, OPENING = 2.3;

/** 物流倉庫：兩條穿越式卸貨道，卡車從一端捲門開進去、卸貨、從另一端開出來。 */
export function DepotView({ traffic, depot }: { traffic: Traffic; depot: Depot }) {
  const doorRefs = useRef<(THREE.Mesh | null)[]>([]);
  const lampRefs = useRef<(THREE.MeshStandardMaterial | null)[]>([]);
  const halfL = depot.length / 2, halfD = depot.depth / 2;
  // 卸貨道在倉庫內的 z 座標（倉庫中心為原點）
  const center = (depot.bays[0].v + depot.bays[depot.bays.length - 1].v) / 2;
  const aisles = depot.bays.map((bay) => bay.v - center);
  useFrame(() => {
    depot.bays.forEach((_, bay) => {
      for (const side of [0, 1]) {
        const door = doorRefs.current[bay * 2 + side];
        if (!door) continue;
        const open = traffic.doors[bay][side];
        const height = DOOR_HEIGHT * (1 - open * 0.9);
        door.scale.y = height;
        door.position.y = DOOR_HEIGHT - height / 2;
        const lamp = lampRefs.current[bay * 2 + side];
        if (lamp) { const color = LAMP[bayState(traffic, bay)]; lamp.color.set(color); lamp.emissive.set(color); }
      }
    });
  });
  // 兩端牆面：卸貨道處留出門洞
  const endWall = (x: number) => {
    const edges = [-halfD, ...aisles.flatMap((z) => [z - OPENING / 2, z + OPENING / 2]), halfD];
    const pieces = [];
    for (let k = 0; k < edges.length; k += 2) {
      const width = edges[k + 1] - edges[k];
      if (width > 0.01) pieces.push(<Block key={`${x}-${k}`} size={[0.25, WALL_HEIGHT, width]} position={[x, WALL_HEIGHT / 2, (edges[k] + edges[k + 1]) / 2]} color="#eef2f9" />);
    }
    for (const z of aisles) pieces.push(<Block key={`${x}-lintel-${z}`} size={[0.25, WALL_HEIGHT - DOOR_HEIGHT, OPENING]} position={[x, (WALL_HEIGHT + DOOR_HEIGHT) / 2, z]} color="#eef2f9" />);
    return pieces;
  };
  return (
    <group>
      {depot.bays.map((bay, i) => <Driveway key={i} path={bay.path} />)}
      <group position={[depot.x, 0, depot.z]} rotation-y={depot.rotation}>
        <Block size={[depot.length + 6, 0.06, depot.depth + 1.2]} position={[0, 0.03, 0]} color="#dde3ee" />
        <Block size={[depot.length, WALL_HEIGHT, 0.25]} position={[0, WALL_HEIGHT / 2, -halfD]} color={PALETTE.wall} />
        <Block size={[depot.length, WALL_HEIGHT, 0.25]} position={[0, WALL_HEIGHT / 2, halfD]} color={PALETTE.wall} />
        {endWall(-halfL)}
        {endWall(halfL)}
        {/* 屋頂：遮住倉庫內的卡車 */}
        <Block size={[depot.length + 0.5, 0.3, depot.depth + 0.5]} position={[0, WALL_HEIGHT + 0.15, 0]} color={PALETTE.blueDark} />
        {[-3, 0, 3].map((x) => <Block key={x} size={[1.6, 0.12, depot.depth * 0.7]} position={[x, WALL_HEIGHT + 0.36, 0]} color="#cfe0ff" />)}
        {aisles.flatMap((z, bay) => [-1, 1].map((sideSign, sideIndex) => (
          <group key={`${bay}-${sideIndex}`}>
            <mesh ref={(element) => { doorRefs.current[bay * 2 + sideIndex] = element; }} position={[sideSign * (halfL + 0.05), DOOR_HEIGHT / 2, z]} castShadow>
              <boxGeometry args={[0.1, 1, OPENING - 0.1]} />
              <meshStandardMaterial color="#d6dbe5" roughness={0.6} />
            </mesh>
            <mesh position={[sideSign * (halfL + 0.2), DOOR_HEIGHT + 0.35, z + OPENING / 2 - 0.2]}>
              <boxGeometry args={[0.12, 0.22, 0.22]} />
              <meshStandardMaterial ref={(element) => { lampRefs.current[bay * 2 + sideIndex] = element; }} emissiveIntensity={0.8} />
            </mesh>
            {/* 地面導引線 */}
            <mesh rotation-x={-Math.PI / 2} position={[sideSign * (halfL + 1.6), 0.07, z]}>
              <planeGeometry args={[2.6, 0.12]} />
              <meshBasicMaterial color={PALETTE.yellow} />
            </mesh>
          </group>
        )))}
        <Pallet position={[-2, 0.06, halfD + 1.1]} layers={3} />
        <Pallet position={[-0.6, 0.06, halfD + 1.1]} layers={2} rotation={0.1} />
        <Forklift position={[2.2, 0.06, halfD + 1.2]} range={1.4} />
      </group>
    </group>
  );
}

/** 倉庫屋頂上的看板（canvas 貼圖的 sprite）：名稱與卸貨道使用數，數字變化時重畫。 */
export function DepotSign({ traffic, depot, label }: { traffic: Traffic; depot: Depot; label: string }) {
  const sign = useMemo(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 420;
    canvas.height = 112;
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 4;
    return { canvas, texture, busy: -1 };
  }, []);
  const sprite = useRef<THREE.Sprite>(null);
  useEffect(() => () => sign.texture.dispose(), [sign]);
  useFrame(() => {
    const busy = traffic.occupants.filter(Boolean).length;
    if (busy === sign.busy) return;
    sign.busy = busy;
    const font = '700 50px Inter, "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", sans-serif';
    let ctx = sign.canvas.getContext("2d")!;
    ctx.font = font;
    // 依字寬決定看板大小，避免名稱與數字重疊（改 width 會重設畫布）
    sign.canvas.width = Math.ceil(ctx.measureText(label).width + ctx.measureText("8/8").width + 96);
    ctx = sign.canvas.getContext("2d")!;
    const { width, height } = sign.canvas;
    sprite.current?.scale.set((2.8 * width) / height, 2.8, 1);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "rgba(31, 52, 102, 0.16)";
    roundRect(ctx, 6, 10, width - 12, height - 14, 22);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    roundRect(ctx, 4, 4, width - 8, height - 14, 22);
    ctx.fill();
    ctx.font = font;
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#1f2937";
    ctx.fillText(label, 28, height / 2 - 4);
    const count = `${busy}/${depot.bays.length}`;
    ctx.fillStyle = busy ? "#c2410c" : "#6b7280";
    ctx.textAlign = "right";
    ctx.fillText(count, width - 28, height / 2 - 4);
    sign.texture.needsUpdate = true;
  });
  return (
    <sprite ref={sprite} position={[depot.x, WALL_HEIGHT + 2.2, depot.z]} scale={[10.5, 2.8, 1]} raycast={noRaycast}>
      <spriteMaterial map={sign.texture} transparent depthWrite={false} />
    </sprite>
  );
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
