"use client";

import { useFrame } from "@react-three/fiber";
import { useMemo, useRef, type ReactNode } from "react";
import * as THREE from "three";

export const PALETTE = {
  blue: "#3563f0",
  blueDark: "#2447c4",
  blueLight: "#6f91ff",
  wall: "#4570f5",
  white: "#f8fafc",
  offWhite: "#e7ecf5",
  grey: "#9aa5b8",
  concrete: "#cfd6e3",
  dark: "#2b3140",
  yellow: "#f6c544",
  yellowDark: "#d9a21c",
  cardboard: "#e3b27c",
  cardboardDark: "#c98f55",
  wood: "#c58a52",
  teal: "#1fb5a8",
  tealDark: "#178f85",
  red: "#ec5a5f",
  redDark: "#c43d43",
  orange: "#ff8a3d",
  leaf: "#6cc785",
  leafDark: "#4fae6b",
  trunk: "#a8754a",
};

type Vec3 = [number, number, number];

/** 共用的實心方塊（投射與接收陰影）。 */
export function Block({ size, position, color, rotation }: { size: Vec3; position: Vec3; color: string; rotation?: Vec3 }) {
  return (
    <mesh position={position} rotation={rotation} castShadow receiveShadow>
      <boxGeometry args={size} />
      <meshStandardMaterial color={color} roughness={0.75} metalness={0.05} />
    </mesh>
  );
}

/** 山形屋頂：三角形斷面沿 z 擠出。 */
function GableRoof({ width, depth, rise, color }: { width: number; depth: number; rise: number; color: string }) {
  const geometry = useMemo(() => {
    const shape = new THREE.Shape();
    shape.moveTo(-width / 2, 0);
    shape.lineTo(width / 2, 0);
    shape.lineTo(0, rise);
    shape.closePath();
    const geo = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: false });
    geo.translate(0, 0, -depth / 2);
    return geo;
  }, [width, depth, rise]);
  return (
    <mesh geometry={geometry} castShadow receiveShadow>
      <meshStandardMaterial color={color} roughness={0.6} />
    </mesh>
  );
}

/** 完工建物：藍色鐵皮倉庫，正面有捲門與屋頂標誌。 */
export function Warehouse({ width, depth, height, highlight }: { width: number; depth: number; height: number; highlight: boolean }) {
  const ribs = Math.max(4, Math.round(depth / 0.55));
  const doors = Math.max(1, Math.floor(width / 1.6));
  return (
    <group>
      <Block size={[width, height, depth]} position={[0, height / 2, 0]} color={highlight ? PALETTE.blueLight : PALETTE.wall} />
      {/* 側牆鐵皮肋條 */}
      {[-1, 1].map((side) =>
        Array.from({ length: ribs }, (_, i) => (
          <Block
            key={`${side}-${i}`}
            size={[0.06, height * 0.96, 0.14]}
            position={[side * (width / 2 + 0.03), height * 0.48, -depth / 2 + (i + 0.5) * (depth / ribs)]}
            color={PALETTE.blueDark}
          />
        )),
      )}
      <group position={[0, height, 0]} rotation={[0, Math.PI / 2, 0]}>
        <GableRoof width={depth + 0.35} depth={width + 0.35} rise={depth * 0.22} color={PALETTE.blue} />
      </group>
      {/* 屋頂圓形標誌 */}
      <mesh position={[0, height + depth * 0.11 + 0.03, depth * 0.25]} rotation={[-Math.PI / 2 + Math.atan(0.44), 0, 0]}>
        <circleGeometry args={[Math.min(width, depth) * 0.16, 32]} />
        <meshStandardMaterial color={PALETTE.white} />
      </mesh>
      {/* 正面（+z）捲門 */}
      {Array.from({ length: doors }, (_, i) => {
        const x = -width / 2 + (i + 0.5) * (width / doors);
        return (
          <group key={i} position={[x, 0, depth / 2 + 0.02]}>
            <Block size={[1.0, height * 0.72, 0.06]} position={[0, height * 0.36, 0]} color={PALETTE.white} />
            <Block size={[0.82, height * 0.62, 0.08]} position={[0, height * 0.31, 0.02]} color={PALETTE.offWhite} />
            {Array.from({ length: 5 }, (_, k) => (
              <Block key={k} size={[0.82, 0.02, 0.1]} position={[0, 0.12 + k * (height * 0.12), 0.03]} color={PALETTE.grey} />
            ))}
          </group>
        );
      })}
      {/* 卸貨平台 */}
      <Block size={[width * 0.9, 0.18, 0.7]} position={[0, 0.09, depth / 2 + 0.38]} color={PALETTE.concrete} />
    </group>
  );
}

/** 棧板＋紙箱堆。 */
export function Pallet({ position, layers = 2, rotation = 0 }: { position: Vec3; layers?: number; rotation?: number }) {
  return (
    <group position={position} rotation={[0, rotation, 0]}>
      <Block size={[1.3, 0.14, 1.1]} position={[0, 0.07, 0]} color={PALETTE.wood} />
      {Array.from({ length: layers }, (_, layer) =>
        [[-0.32, -0.26], [0.32, -0.26], [-0.32, 0.26], [0.32, 0.26]].map(([dx, dz], k) => (
          <group key={`${layer}-${k}`} position={[dx, 0.14 + layer * 0.48 + 0.24, dz]}>
            <Block size={[0.6, 0.46, 0.5]} position={[0, 0, 0]} color={(layer + k) % 2 ? PALETTE.cardboard : PALETTE.cardboardDark} />
            <Block size={[0.61, 0.04, 0.12]} position={[0, 0.2, 0]} color="#f1d3a8" />
          </group>
        )),
      )}
    </group>
  );
}

/** 交通錐。 */
export function Cone({ position }: { position: Vec3 }) {
  return (
    <group position={position}>
      <Block size={[0.42, 0.06, 0.42]} position={[0, 0.03, 0]} color={PALETTE.orange} />
      <mesh position={[0, 0.3, 0]} castShadow>
        <coneGeometry args={[0.16, 0.5, 12]} />
        <meshStandardMaterial color={PALETTE.orange} />
      </mesh>
      <mesh position={[0, 0.3, 0]}>
        <cylinderGeometry args={[0.105, 0.125, 0.08, 12]} />
        <meshStandardMaterial color={PALETTE.white} />
      </mesh>
    </group>
  );
}

/** 地塊虛線框（黃色地面標線）。 */
export function LotOutline({ size, color = PALETTE.yellow }: { size: number; color?: string }) {
  const half = size / 2, dash = 0.7, gap = 0.45;
  const segments: { x: number; z: number; horizontal: boolean }[] = [];
  for (let s = -half; s < half - 0.1; s += dash + gap) {
    const mid = Math.min(s + dash / 2, half - dash / 2);
    segments.push({ x: mid, z: -half, horizontal: true }, { x: mid, z: half, horizontal: true });
    segments.push({ x: -half, z: mid, horizontal: false }, { x: half, z: mid, horizontal: false });
  }
  return (
    <group position={[0, 0.03, 0]}>
      {segments.map((seg, i) => (
        <mesh key={i} position={[seg.x, 0, seg.z]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
          <planeGeometry args={seg.horizontal ? [dash, 0.12] : [0.12, dash]} />
          <meshBasicMaterial color={color} />
        </mesh>
      ))}
    </group>
  );
}

/** 定位圖釘：浮動上下並緩慢旋轉。 */
export function MapPin({ position, color = PALETTE.blue, scale = 1 }: { position: Vec3; color?: string; scale?: number }) {
  const ref = useRef<THREE.Group>(null);
  const phase = useMemo(() => Math.random() * Math.PI * 2, []);
  useFrame(({ clock }) => {
    if (!ref.current) return;
    ref.current.position.y = position[1] + Math.sin(clock.elapsedTime * 2.2 + phase) * 0.18;
    ref.current.rotation.y = clock.elapsedTime * 0.8;
  });
  return (
    <group ref={ref} position={position} scale={scale}>
      <mesh position={[0, 1.05, 0]} castShadow>
        <sphereGeometry args={[0.55, 24, 16]} />
        <meshStandardMaterial color={color} roughness={0.35} />
      </mesh>
      <mesh position={[0, 0.42, 0]} rotation={[Math.PI, 0, 0]} castShadow>
        <coneGeometry args={[0.42, 0.95, 24]} />
        <meshStandardMaterial color={color} roughness={0.35} />
      </mesh>
      <mesh position={[0, 1.05, 0.47]}>
        <circleGeometry args={[0.24, 24]} />
        <meshBasicMaterial color={PALETTE.white} />
      </mesh>
      <mesh position={[0, 1.05, -0.47]} rotation={[0, Math.PI, 0]}>
        <circleGeometry args={[0.24, 24]} />
        <meshBasicMaterial color={PALETTE.white} />
      </mesh>
    </group>
  );
}

/** 塔式起重機：吊臂緩慢旋轉，吊鉤掛著一個棧板。 */
export function TowerCrane({ position, height = 7, jib = 5.5 }: { position: Vec3; height?: number; jib?: number }) {
  const head = useRef<THREE.Group>(null);
  useFrame((_, dt) => { if (head.current) head.current.rotation.y += dt * 0.25; });
  const mastSegments = Math.floor(height / 0.8);
  return (
    <group position={position}>
      <Block size={[1.1, 0.3, 1.1]} position={[0, 0.15, 0]} color={PALETTE.concrete} />
      {/* 桁架塔身：四根立柱＋斜撐 */}
      {[[-0.22, -0.22], [0.22, -0.22], [-0.22, 0.22], [0.22, 0.22]].map(([x, z], i) => (
        <Block key={i} size={[0.08, height, 0.08]} position={[x, height / 2 + 0.3, z]} color={PALETTE.yellow} />
      ))}
      {Array.from({ length: mastSegments }, (_, i) => (
        <group key={i} position={[0, 0.7 + i * 0.8, 0]}>
          <Block size={[0.5, 0.05, 0.05]} position={[0, 0, -0.22]} color={PALETTE.yellowDark} />
          <Block size={[0.5, 0.05, 0.05]} position={[0, 0, 0.22]} color={PALETTE.yellowDark} />
          <Block size={[0.05, 0.05, 0.5]} position={[-0.22, 0, 0]} color={PALETTE.yellowDark} />
          <Block size={[0.05, 0.05, 0.5]} position={[0.22, 0, 0]} color={PALETTE.yellowDark} />
        </group>
      ))}
      <group ref={head} position={[0, height + 0.3, 0]}>
        <Block size={[0.6, 0.55, 0.6]} position={[0.1, 0.28, 0]} color={PALETTE.white} />
        <Block size={[0.45, 0.35, 0.62]} position={[0.32, 0.35, 0]} color="#8fb4ff" />
        <Block size={[jib, 0.22, 0.32]} position={[jib / 2, 0.65, 0]} color={PALETTE.yellow} />
        <Block size={[2, 0.22, 0.32]} position={[-1, 0.65, 0]} color={PALETTE.yellow} />
        <Block size={[0.7, 0.55, 0.5]} position={[-1.7, 0.4, 0]} color={PALETTE.grey} />
        <Block size={[0.1, 1.1, 0.1]} position={[0, 1.2, 0]} color={PALETTE.yellowDark} />
        <Block size={[0.03, 2.4, 0.03]} position={[jib * 0.7, -0.55, 0]} color={PALETTE.dark} />
        <group position={[jib * 0.7, -1.9, 0]} scale={0.6}>
          <Pallet position={[0, 0, 0]} layers={1} />
        </group>
      </group>
    </group>
  );
}

/** 施工中：混凝土基座、半高牆、黃色鷹架。 */
export function ConstructionSite({ width, depth, height, progress }: { width: number; depth: number; height: number; progress: number }) {
  const built = Math.max(0.4, height * progress);
  const poles: [number, number][] = [];
  for (let x = -width / 2 - 0.25; x <= width / 2 + 0.26; x += width / 3 + 0.17) {
    poles.push([x, -depth / 2 - 0.25], [x, depth / 2 + 0.25]);
  }
  const levels = [0.9, 1.8, height + 0.1].filter((y) => y <= height + 0.2);
  return (
    <group>
      <Block size={[width + 0.6, 0.18, depth + 0.6]} position={[0, 0.09, 0]} color={PALETTE.concrete} />
      <Block size={[width, built, depth]} position={[0, 0.18 + built / 2, 0]} color={PALETTE.wall} />
      <Block size={[width * 0.98, 0.06, depth * 0.98]} position={[0, 0.18 + built + 0.03, 0]} color={PALETTE.offWhite} />
      {poles.map(([x, z], i) => (
        <Block key={i} size={[0.07, height + 0.4, 0.07]} position={[x, (height + 0.4) / 2, z]} color={PALETTE.yellow} />
      ))}
      {levels.map((y) =>
        [-1, 1].map((side) => (
          <group key={`${y}-${side}`}>
            <Block size={[width + 0.6, 0.06, 0.06]} position={[0, y, side * (depth / 2 + 0.25)]} color={PALETTE.yellowDark} />
            <Block size={[width + 0.6, 0.04, 0.4]} position={[0, y - 0.05, side * (depth / 2 + 0.45)]} color={PALETTE.wood} />
          </group>
        )),
      )}
    </group>
  );
}

/** 堆高機：在地塊前來回行駛。 */
export function Forklift({ position, range = 2, rotation = 0 }: { position: Vec3; range?: number; rotation?: number }) {
  const ref = useRef<THREE.Group>(null);
  const phase = useMemo(() => Math.random() * 10, []);
  useFrame(({ clock }) => {
    if (ref.current) ref.current.position.x = Math.sin(clock.elapsedTime * 0.6 + phase) * range;
  });
  return (
    <group position={position} rotation={[0, rotation, 0]}>
      <group ref={ref}>
        <Block size={[1.1, 0.45, 0.75]} position={[0, 0.42, 0]} color={PALETTE.yellow} />
        <Block size={[0.4, 0.5, 0.72]} position={[-0.35, 0.85, 0]} color={PALETTE.yellowDark} />
        {[[-0.05, -0.3], [-0.05, 0.3], [0.38, -0.3], [0.38, 0.3]].map(([x, z], i) => (
          <Block key={i} size={[0.05, 0.75, 0.05]} position={[x, 1.02, z]} color={PALETTE.dark} />
        ))}
        <Block size={[0.5, 0.05, 0.7]} position={[0.16, 1.4, 0]} color={PALETTE.dark} />
        <Block size={[0.08, 1.2, 0.5]} position={[0.62, 0.75, 0]} color={PALETTE.dark} />
        <Block size={[0.6, 0.05, 0.1]} position={[0.9, 0.2, -0.16]} color={PALETTE.grey} />
        <Block size={[0.6, 0.05, 0.1]} position={[0.9, 0.2, 0.16]} color={PALETTE.grey} />
        <Block size={[0.24, 0.3, 0.22]} position={[0.02, 0.8, 0]} color={PALETTE.orange} />
        {[[-0.32, 0.4], [-0.32, -0.4], [0.32, 0.4], [0.32, -0.4]].map(([x, z], i) => (
          <mesh key={i} position={[x, 0.18, z]} rotation={[Math.PI / 2, 0, 0]} castShadow>
            <cylinderGeometry args={[0.18, 0.18, 0.14, 14]} />
            <meshStandardMaterial color={PALETTE.dark} />
          </mesh>
        ))}
      </group>
    </group>
  );
}

/** 貨運卡車（車頭朝 +x）。 */
export function Truck({ stripe, highlight }: { stripe: string; highlight: boolean }) {
  return (
    <group scale={0.95}>
      <Block size={[3.4, 1.5, 1.35]} position={[-0.5, 1.2, 0]} color={highlight ? "#ffffff" : PALETTE.white} />
      <Block size={[3.42, 0.32, 1.37]} position={[-0.5, 0.62, 0]} color={stripe} />
      <Block size={[0.6, 0.5, 1.37]} position={[-1.6, 1.4, 0]} color={stripe} />
      <Block size={[1.0, 1.1, 1.3]} position={[1.75, 0.95, 0]} color={PALETTE.blue} />
      <Block size={[0.06, 0.45, 1.1]} position={[2.26, 1.2, 0]} color="#1c2433" />
      <Block size={[0.7, 0.42, 0.04]} position={[1.8, 1.2, 0.66]} color="#1c2433" />
      <Block size={[0.7, 0.42, 0.04]} position={[1.8, 1.2, -0.66]} color="#1c2433" />
      <Block size={[4.5, 0.2, 1.1]} position={[-0.1, 0.38, 0]} color={PALETTE.dark} />
      {[-1.6, -0.9, 1.75].map((x) =>
        [-0.6, 0.6].map((z) => (
          <mesh key={`${x}-${z}`} position={[x, 0.3, z]} rotation={[Math.PI / 2, 0, 0]} castShadow>
            <cylinderGeometry args={[0.3, 0.3, 0.24, 16]} />
            <meshStandardMaterial color={PALETTE.dark} />
          </mesh>
        )),
      )}
    </group>
  );
}

/** 海運貨櫃（長邊沿 x）。 */
export function ShippingContainer({ color, ribColor, highlight }: { color: string; ribColor: string; highlight: boolean }) {
  const length = 5.4, height = 2.1, width = 2.2;
  return (
    <group>
      <Block size={[length, height, width]} position={[0, height / 2, 0]} color={highlight ? ribColor : color} />
      {Array.from({ length: 14 }, (_, i) => {
        const x = -length / 2 + 0.3 + i * ((length - 0.6) / 13);
        return [-1, 1].map((side) => (
          <Block key={`${i}-${side}`} size={[0.08, height * 0.86, 0.05]} position={[x, height / 2, side * (width / 2 + 0.02)]} color={ribColor} />
        ));
      })}
      <Block size={[length + 0.04, 0.12, width + 0.04]} position={[0, height - 0.06, 0]} color={ribColor} />
      <Block size={[length + 0.04, 0.12, width + 0.04]} position={[0, 0.06, 0]} color={ribColor} />
    </group>
  );
}

export function Tree({ position, scale = 1 }: { position: Vec3; scale?: number }) {
  return (
    <group position={position} scale={scale}>
      <mesh position={[0, 0.55, 0]} castShadow>
        <cylinderGeometry args={[0.1, 0.14, 1.1, 8]} />
        <meshStandardMaterial color={PALETTE.trunk} />
      </mesh>
      <mesh position={[0, 1.7, 0]} castShadow>
        <sphereGeometry args={[0.75, 16, 12]} />
        <meshStandardMaterial color={PALETTE.leaf} roughness={0.8} />
      </mesh>
      <mesh position={[0.15, 2.25, 0.05]} castShadow>
        <sphereGeometry args={[0.48, 16, 12]} />
        <meshStandardMaterial color={PALETTE.leafDark} roughness={0.8} />
      </mesh>
    </group>
  );
}

/** 地塊可點擊區域＋懸停效果的外殼。 */
export function Hoverable({ children, onClick, onHover }: { children: ReactNode; onClick: () => void; onHover: (on: boolean) => void }) {
  return (
    <group
      onClick={(event) => { event.stopPropagation(); onClick(); }}
      onPointerOver={(event) => { event.stopPropagation(); onHover(true); document.body.style.cursor = "pointer"; }}
      onPointerOut={() => { onHover(false); document.body.style.cursor = ""; }}
    >
      {children}
    </group>
  );
}

const noRaycast = () => null;

/** 選取框：WareTrack 風格的藍色角框＋淡藍填色＋地面光暈；不參與點擊判定，避免擋住其他物件。 */
export function SelectionBrackets({ size, color = PALETTE.blue }: { size: Vec3; color?: string }) {
  const [width, height, depth] = size;
  const geometry = useMemo(() => {
    const points: number[] = [];
    const arm = Math.min(width, height, depth) * 0.28 + 0.18;
    for (const x of [-width / 2, width / 2]) {
      for (const y of [0, height]) {
        for (const z of [-depth / 2, depth / 2]) {
          const sx = x < 0 ? 1 : -1, sy = y === 0 ? 1 : -1, sz = z < 0 ? 1 : -1;
          points.push(x, y, z, x + sx * arm, y, z, x, y, z, x, y + sy * arm, z, x, y, z, x, y, z + sz * arm);
        }
      }
    }
    return new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  }, [width, height, depth]);
  return (
    <group>
      <lineSegments geometry={geometry} raycast={noRaycast}>
        <lineBasicMaterial color={color} />
      </lineSegments>
      <mesh position-y={height / 2} raycast={noRaycast}>
        <boxGeometry args={size} />
        <meshBasicMaterial color={color} transparent opacity={0.07} depthWrite={false} />
      </mesh>
      <mesh rotation-x={-Math.PI / 2} position-y={0.02} raycast={noRaycast}>
        <planeGeometry args={[width + 1.4, depth + 1.4]} />
        <meshBasicMaterial color={color} transparent opacity={0.14} depthWrite={false} />
      </mesh>
    </group>
  );
}
