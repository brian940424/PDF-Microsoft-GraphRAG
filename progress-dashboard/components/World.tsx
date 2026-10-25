"use client";

import { Html, OrbitControls, OrthographicCamera } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
  createContext, useContext, useEffect, useMemo, useRef, useState,
  type ComponentProps, type ComponentRef, type MutableRefObject, type RefObject,
} from "react";
import * as THREE from "three";

import { LOT_SIZE, layoutSite, type Lot, type SiteLayout } from "@/lib/layout";
import { createTraffic, layoutTown, syncTrucks, vehiclePose, type Traffic } from "@/lib/world";
import type { Commit, ProgressData } from "@/lib/types";

import {
  Cone, ConstructionSite, Forklift, Hoverable, LotOutline, MapPin, PALETTE, Pallet,
  SelectionBrackets, ShippingContainer, TowerCrane, Truck, Warehouse,
} from "./models";
import { Cars, DepotSign, DepotView, StreetsView, TownView, TrafficDriver, TrafficLights, Trees } from "./Town";
import { commitColor, recentCommits, statusStyle, type CameraCommand, type Selection } from "./theme";

interface WorldProps {
  data: ProgressData;
  selection: Selection;
  statusFilter: Set<string>;
  cameraCommand: CameraCommand | null;
  /** 暫停時停止動畫並改為按需渲染，只在鏡頭或狀態改變時重繪 */
  paused: boolean;
  /** 鏡頭持續跟隨目前選取的卡車 */
  following: boolean;
  onFollowEnd: () => void;
  onSelect: (selection: Selection) => void;
}

/** 各卡車目前的道路座標，供跟隨鏡頭讀取 */
type TruckPositions = MutableRefObject<Map<string, THREE.Vector3>>;

const VIEW_DIRECTION = new THREE.Vector3(1, 0.95, 1).normalize();
const VIEW_DISTANCE = 120;
const HUD_RIGHT = 400;

/**
 * 3D 標籤的固定容器。drei 的 Html 預設掛在畫布的父節點上，但第一次渲染時畫布可能還沒進 DOM，
 * 之後目標改變會重建內部 root，舊 root 延遲卸載時把新內容一起清空，造成標籤隨機消失；
 * 改掛到從一開始就存在的容器即可避免。
 */
const HtmlLayer = createContext<RefObject<HTMLDivElement | null> | null>(null);

function Label(props: ComponentProps<typeof Html>) {
  const layer = useContext(HtmlLayer);
  return <Html {...props} portal={(layer ?? undefined) as MutableRefObject<HTMLElement> | undefined} />;
}

export default function World(props: WorldProps) {
  const htmlLayer = useRef<HTMLDivElement>(null);
  const layout = useMemo(() => layoutSite(props.data.plan, props.data.tests?.modules ?? []), [props.data]);
  const trucks = useMemo(() => recentCommits(props.data.git.commits), [props.data.git.commits]);
  const truckPositions = useRef(new Map<string, THREE.Vector3>());
  // 城市（路網、城鎮、車流）只在街廓格線改變時重建；提交清單變動時同步卡車，其他車輛維持原位
  const roadKey = [...layout.grid.xs, 0, ...layout.grid.zs].map((value) => value.toFixed(2)).join(",");
  const truckIds = trucks.map((commit) => commit.hash).join(",");
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const traffic = useMemo(() => createTraffic(layout.grid, truckIds.split(",").filter(Boolean)), [roadKey]);
  useEffect(() => { syncTrucks(traffic, truckIds.split(",").filter(Boolean)); }, [traffic, truckIds]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const town = useMemo(() => layoutTown(traffic, layout.parks), [traffic]);
  const shadowRange = layout.radius + 22;
  const followHash = props.following && props.selection?.kind === "commit" ? props.selection.hash : null;
  return (
    <HtmlLayer.Provider value={htmlLayer}>
      <Canvas shadows flat dpr={[1, 2]} frameloop={props.paused ? "demand" : "always"} onPointerMissed={() => props.onSelect(null)}>
        <color attach="background" args={["#e9eef7"]} />
        <OrthographicCamera makeDefault position={VIEW_DIRECTION.clone().multiplyScalar(VIEW_DISTANCE).toArray()} near={1} far={500} zoom={10} />
        <ambientLight intensity={1.15} />
        <hemisphereLight args={["#ffffff", "#c9d3e6", 0.7]} />
        <directionalLight
          castShadow
          position={[-30, 60, 40]}
          intensity={1.5}
          shadow-mapSize={[4096, 4096]}
          shadow-bias={-0.0004}
          shadow-normalBias={0.02}
          shadow-camera-left={-shadowRange}
          shadow-camera-right={shadowRange}
          shadow-camera-top={shadowRange}
          shadow-camera-bottom={-shadowRange}
          shadow-camera-far={260}
        />
        <Ground layout={layout} />
        <TrafficDriver traffic={traffic} paused={props.paused} />
        <StreetsView traffic={traffic} />
        <TrafficLights traffic={traffic} />
        <TownView town={town} />
        {traffic.depot && <DepotView traffic={traffic} depot={traffic.depot} />}
        {traffic.depot && <DepotSign traffic={traffic} depot={traffic.depot} label="Git 倉庫 · 卸貨" />}
        <Cars traffic={traffic} />
        {layout.zones.map((zone) => (
          <group key={zone.name}>
            <ZonePad x={zone.x} z={zone.z} w={zone.w} d={zone.d} label={`${zone.name}區`} count={zone.lots.length} />
            {zone.lots.map((lot) => (
              <LotView
                key={lot.item.id}
                lot={lot}
                selected={props.selection?.kind === "plan" && props.selection.id === lot.item.id}
                dimmed={props.statusFilter.size > 0 && !props.statusFilter.has(lot.item.status)}
                onSelect={props.onSelect}
              />
            ))}
          </group>
        ))}
        {layout.yard && <YardView layout={layout} selection={props.selection} onSelect={props.onSelect} />}
        {trucks.map((commit) => (
          <MovingTruck
            key={commit.hash}
            commit={commit}
            traffic={traffic}
            selected={props.selection?.kind === "commit" && props.selection.hash === commit.hash}
            following={followHash === commit.hash}
            positions={truckPositions}
            onSelect={props.onSelect}
          />
        ))}
        <Trees trees={town.trees} />
        <CameraRig
          layout={layout}
          selection={props.selection}
          command={props.cameraCommand}
          followHash={followHash}
          truckPositions={truckPositions}
          onFollowEnd={props.onFollowEnd}
        />
      </Canvas>
      <div ref={htmlLayer} className="html-layer" />
    </HtmlLayer.Provider>
  );
}

/** 地面：遠處淺灰、城鎮綠地、分區街廓的淺色鋪面（街道由 StreetsView 畫）。 */
function Ground({ layout }: { layout: SiteLayout }) {
  const { road } = layout;
  return (
    <group>
      <mesh rotation-x={-Math.PI / 2} position-y={-0.03} receiveShadow>
        <planeGeometry args={[layout.radius * 8, layout.radius * 8]} />
        <meshStandardMaterial color="#e3e9f4" />
      </mesh>
      <mesh rotation-x={-Math.PI / 2} position-y={-0.02} receiveShadow>
        <planeGeometry args={[road.w + 44, road.d + 44]} />
        <meshStandardMaterial color="#d9eedb" />
      </mesh>
      <mesh rotation-x={-Math.PI / 2} position-y={-0.01} receiveShadow>
        <planeGeometry args={[road.w, road.d]} />
        <meshStandardMaterial color="#f1f4fa" />
      </mesh>
    </group>
  );
}

function ZonePad({ x, z, w, d, label, count }: { x: number; z: number; w: number; d: number; label: string; count: number }) {
  return (
    <group position={[x, 0, z]}>
      <mesh position-y={0.06} receiveShadow>
        <boxGeometry args={[w, 0.12, d]} />
        <meshStandardMaterial color="#fbfcfe" />
      </mesh>
      {[[0, -d / 2, w, 0.16], [0, d / 2, w, 0.16], [-w / 2, 0, 0.16, d], [w / 2, 0, 0.16, d]].map(([px, pz, sw, sd], i) => (
        <mesh key={i} position={[px, 0.125, pz]} rotation-x={-Math.PI / 2}>
          <planeGeometry args={[sw, sd]} />
          <meshBasicMaterial color="#c9d4ea" />
        </mesh>
      ))}
      <Label position={[-w / 2 + 0.4, 0.2, -d / 2 + 0.8]} zIndexRange={[5, 0]} className="zone-label">
        {label}<span>{count}</span>
      </Label>
    </group>
  );
}

function LotView({ lot, selected, dimmed, onSelect }: {
  lot: Lot; selected: boolean; dimmed: boolean; onSelect: WorldProps["onSelect"];
}) {
  const [hovered, setHovered] = useState(false);
  const { item } = lot;
  const [width, depth, height] = item.priority === "高" ? [4.6, 3.6, 2.6] : item.priority === "中" ? [4.0, 3.2, 2.2] : [3.4, 2.8, 1.9];
  // 選取框與圖釘的高度：倉庫含屋頂、施工中含鷹架，待辦只有棧板
  const top = item.status === "完成" ? height + depth * 0.22 : item.status === "進行中" ? height + 0.2 : 1.6;
  if (dimmed) {
    return (
      <group position={[lot.x, 0.12, lot.z]}>
        <LotOutline size={LOT_SIZE - 0.9} color="#d5dbe7" />
      </group>
    );
  }
  return (
    <group position={[lot.x, 0.12, lot.z]}>
      <Hoverable onClick={() => onSelect({ kind: "plan", id: item.id })} onHover={setHovered}>
        {/* 透明點擊區，讓整個地塊都可點 */}
        <mesh position-y={1} visible={false}>
          <boxGeometry args={[LOT_SIZE - 0.6, 2, LOT_SIZE - 0.6]} />
        </mesh>
        <LotOutline size={LOT_SIZE - 0.9} color={selected ? PALETTE.blue : hovered ? PALETTE.blueLight : PALETTE.yellow} />
        {item.status === "完成" && <Warehouse width={width} depth={depth} height={height} highlight={hovered || selected} />}
        {item.status === "進行中" && (
          <>
            <ConstructionSite width={width} depth={depth} height={height} progress={0.55} />
            <TowerCrane position={[-LOT_SIZE / 2 + 0.9, 0, -LOT_SIZE / 2 + 0.9]} height={height + 4.5} jib={4.8} />
            <Forklift position={[0, 0, LOT_SIZE / 2 - 0.9]} range={1.6} />
          </>
        )}
        {(item.status === "待辦" || item.status === "待確認") && (
          <>
            <Pallet position={[-1.1, 0, -0.6]} layers={2} />
            <Pallet position={[0.6, 0, -0.9]} layers={1} rotation={0.2} />
            <Pallet position={[0.9, 0, 0.9]} layers={item.priority === "高" ? 3 : 2} rotation={-0.1} />
            <Cone position={[-2.4, 0, 2.4]} />
            <Cone position={[2.4, 0, -2.4]} />
          </>
        )}
        {item.status === "不處理" && <Cone position={[0, 0, 0]} />}
        {item.status === "待確認" && !selected && <MapPin position={[0, 2.2, 0]} color={PALETTE.yellow} />}
        {selected && <MapPin position={[0, top + 0.6, 0]} color={PALETTE.blue} scale={1.15} />}
      </Hoverable>
      {selected && <SelectionBrackets size={[LOT_SIZE - 0.6, top, LOT_SIZE - 0.6]} />}
      {selected && (
        <Label position={[0, top + 3.2, 0]} center zIndexRange={[8, 0]} className="tag3d">
          {item.id}<small>{statusStyle(item.status).label}</small>
        </Label>
      )}
      <Label position={[0, 0.2, LOT_SIZE / 2 - 0.55]} center zIndexRange={[6, 0]} className={`lot-sign ${selected ? "on" : ""}`}>
        {item.id}
        {item.uncommitted && <i title="已完成但未提交" />}
      </Label>
      {hovered && !selected && (
        <Label position={[0, 4.6, 0]} center zIndexRange={[7, 0]} className="tooltip3d">
          <b>{item.id}</b> {item.title.replace(/`/g, "")}
        </Label>
      )}
    </group>
  );
}

function YardView({ layout, selection, onSelect }: { layout: SiteLayout; selection: Selection; onSelect: WorldProps["onSelect"] }) {
  const yard = layout.yard!;
  return (
    <group>
      <ZonePad x={yard.x} z={yard.z} w={yard.w} d={yard.d} label="測試貨櫃場" count={yard.slots.length} />
      {yard.slots.map((slot) => (
        <ContainerView
          key={slot.module.module}
          slot={slot}
          selected={selection?.kind === "module" && selection.module === slot.module.module}
          onSelect={onSelect}
        />
      ))}
    </group>
  );
}

function ContainerView({ slot, selected, onSelect }: {
  slot: NonNullable<SiteLayout["yard"]>["slots"][number]; selected: boolean; onSelect: WorldProps["onSelect"];
}) {
  const [hovered, setHovered] = useState(false);
  const failing = slot.module.failed > 0;
  const total = slot.module.passed + slot.module.failed + slot.module.skipped;
  return (
    <group position={[slot.x, 0.12, slot.z]}>
      <Hoverable onClick={() => onSelect({ kind: "module", module: slot.module.module })} onHover={setHovered}>
        <ShippingContainer
          color={failing ? PALETTE.red : PALETTE.teal}
          ribColor={failing ? PALETTE.redDark : PALETTE.tealDark}
          highlight={hovered || selected}
        />
      </Hoverable>
      {(failing || selected) && <MapPin position={[0, 2.5, 0]} color={selected ? PALETTE.blue : PALETTE.red} scale={0.8} />}
      {selected && <SelectionBrackets size={[3.2, 1.9, 1.7]} />}
      {(hovered || selected) && (
        <Label position={[0, 4.4, 0]} center zIndexRange={[7, 0]} className={selected ? "tag3d" : "tooltip3d"}>
          {selected ? <>{slot.module.module}<small>{slot.module.passed}/{total} 通過</small></> : <><b>{slot.module.module}</b> {slot.module.passed}/{total} 通過</>}
        </Label>
      )}
    </group>
  );
}

function MovingTruck({ commit, traffic, selected, following, positions, onSelect }: {
  commit: Commit; traffic: Traffic; selected: boolean; following: boolean;
  positions: TruckPositions; onSelect: WorldProps["onSelect"];
}) {
  const ref = useRef<THREE.Group>(null);
  const [hovered, setHovered] = useState(false);
  useEffect(() => () => { positions.current.delete(commit.hash); }, [positions, commit.hash]);
  useFrame(() => {
    const vehicle = traffic.vehicles.find((candidate) => candidate.id === commit.hash);
    if (!ref.current || !vehicle) return;
    // 跟隨中的卡車繼續行駛；否則選取或滑過時停下來方便查看（後方車輛會排隊）
    vehicle.held = !following && (selected || hovered);
    const pose = vehiclePose(traffic, vehicle);
    ref.current.position.set(pose.x, 0, pose.z);
    ref.current.rotation.y = pose.heading;
    const stored = positions.current.get(commit.hash);
    if (stored) stored.set(pose.x, 0, pose.z);
    else positions.current.set(commit.hash, new THREE.Vector3(pose.x, 0, pose.z));
  });
  return (
    <group ref={ref}>
      <group>
        <Hoverable onClick={() => onSelect({ kind: "commit", hash: commit.hash })} onHover={setHovered}>
          <Truck stripe={commitColor(commit.type)} highlight={hovered || selected} />
        </Hoverable>
        {selected && <SelectionBrackets size={[3.6, 1.9, 1.7]} />}
        {(hovered || selected) && (
          <Label position={[0, 3.2, 0]} center zIndexRange={[7, 0]} className={selected ? "tag3d" : "tooltip3d"}>
            {selected ? <>{commit.hash}<small>{following ? "跟隨中" : commit.type}</small></> : <><b>{commit.hash}</b> {commit.subject}</>}
          </Label>
        )}
      </group>
    </group>
  );
}

interface CameraGoal {
  target: THREE.Vector3;
  direction: THREE.Vector3;
  zoom: number;
}

/** 等角正交鏡頭：平滑移動到選取目標，並處理縮放、旋轉、回到全景等指令。 */
function CameraRig({ layout, selection, command, followHash, truckPositions, onFollowEnd }: {
  layout: SiteLayout; selection: Selection; command: CameraCommand | null;
  followHash: string | null; truckPositions: TruckPositions; onFollowEnd: () => void;
}) {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const { camera, size, gl, invalidate } = useThree();
  const ortho = camera as THREE.OrthographicCamera;
  // 右側卡片遮住約 HUD_RIGHT 像素：以 view offset 讓場景中心落在可見區域中央
  const hudRight = size.width > 1100 ? HUD_RIGHT : 0;
  const fitZoom = Math.min((size.width - hudRight) / (layout.radius * 1.9), size.height / (layout.radius * 1.35));
  const goal = useRef<CameraGoal | null>(null);
  const initialized = useRef(false);
  // 跟隨時鏡頭目標每幀追著卡車；followZoom 為 null 代表使用者已自行縮放，不再拉回
  const following = useRef(followHash);
  const followZoom = useRef<number | null>(null);

  const currentDirection = () =>
    controls.current ? camera.position.clone().sub(controls.current.target).normalize() : VIEW_DIRECTION.clone();

  useEffect(() => {
    if (!size.width) return;
    ortho.setViewOffset(size.width, size.height, hudRight / 2, 0, size.width, size.height);
    ortho.updateProjectionMatrix();
  }, [ortho, size.width, size.height, hudRight]);

  useEffect(() => {
    if (initialized.current || !size.width) return;
    initialized.current = true;
    ortho.zoom = fitZoom;
    ortho.updateProjectionMatrix();
  }, [ortho, fitZoom, size.width]);

  useEffect(() => {
    const point = focusPoint(layout, selection);
    if (point) goal.current = { target: point, direction: currentDirection(), zoom: Math.max(fitZoom * 2, ortho.zoom) };
    // 只在選取改變時移動鏡頭
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection]);

  useEffect(() => {
    following.current = followHash;
    followZoom.current = followHash ? Math.max(fitZoom * 2, ortho.zoom) : null;
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [followHash]);

  useEffect(() => {
    if (!command || !controls.current) return;
    const target = controls.current.target.clone();
    const direction = currentDirection();
    const base = goal.current ?? { target, direction, zoom: ortho.zoom };
    if (command.kind === "zoomIn") goal.current = { ...base, zoom: Math.min(fitZoom * 6, base.zoom * 1.4) };
    if (command.kind === "zoomOut") goal.current = { ...base, zoom: Math.max(fitZoom * 0.6, base.zoom / 1.4) };
    if (command.kind === "rotateLeft" || command.kind === "rotateRight") {
      const angle = command.kind === "rotateLeft" ? Math.PI / 4 : -Math.PI / 4;
      goal.current = { ...base, direction: base.direction.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), angle) };
    }
    if (command.kind === "home") goal.current = { target: new THREE.Vector3(), direction: VIEW_DIRECTION.clone(), zoom: fitZoom };
    if (following.current) followZoom.current = goal.current?.zoom ?? null;
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [command]);

  useEffect(() => {
    const instance = controls.current;
    if (!instance) return;
    // 跟隨中滾輪縮放不中斷跟隨，只是不再把縮放拉回
    const cancel = () => {
      if (following.current) followZoom.current = null;
      else goal.current = null;
    };
    instance.addEventListener("start", cancel);
    return () => instance.removeEventListener("start", cancel);
  }, []);

  useEffect(() => {
    // 拖曳平移超過 6px 視為手動操作，結束跟隨（同 WareTrack）
    const element = gl.domElement;
    let down: [number, number] | null = null;
    const onDown = (event: PointerEvent) => { down = [event.clientX, event.clientY]; };
    const onMove = (event: PointerEvent) => {
      if (!down || Math.hypot(event.clientX - down[0], event.clientY - down[1]) < 6) return;
      down = null;
      if (following.current) { goal.current = null; onFollowEnd(); }
    };
    const onUp = () => { down = null; };
    element.addEventListener("pointerdown", onDown);
    element.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      element.removeEventListener("pointerdown", onDown);
      element.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [gl, onFollowEnd]);

  useFrame((_, dt) => {
    const instance = controls.current;
    if (!instance) return;
    const truck = following.current ? truckPositions.current.get(following.current) : undefined;
    if (truck) {
      goal.current = { target: truck.clone(), direction: goal.current?.direction ?? currentDirection(), zoom: followZoom.current ?? ortho.zoom };
    }
    if (!goal.current) return;
    // 按需渲染（暫停）時，鏡頭移動期間持續要求下一幀
    invalidate();
    const k = 1 - Math.exp(-dt * 4);
    instance.target.lerp(goal.current.target, k);
    const direction = currentDirection().lerp(goal.current.direction, k).normalize();
    camera.position.copy(instance.target).add(direction.multiplyScalar(VIEW_DISTANCE));
    ortho.zoom += (goal.current.zoom - ortho.zoom) * k;
    ortho.updateProjectionMatrix();
    if (!truck && instance.target.distanceTo(goal.current.target) < 0.02 && Math.abs(ortho.zoom - goal.current.zoom) < 0.05
      && currentDirection().distanceTo(goal.current.direction) < 0.002) {
      goal.current = null;
    }
  });

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={0.12}
      minPolarAngle={Math.PI * 0.12}
      maxPolarAngle={Math.PI * 0.38}
      minZoom={fitZoom * 0.5}
      maxZoom={fitZoom * 6}
      screenSpacePanning={false}
    />
  );
}

function focusPoint(layout: SiteLayout, selection: Selection): THREE.Vector3 | null {
  if (!selection) return null;
  if (selection.kind === "plan") {
    for (const zone of layout.zones) {
      const lot = zone.lots.find((candidate) => candidate.item.id === selection.id);
      if (lot) return new THREE.Vector3(lot.x, 0, lot.z);
    }
  }
  if (selection.kind === "module" && layout.yard) {
    const slot = layout.yard.slots.find((candidate) => candidate.module.module === selection.module);
    if (slot) return new THREE.Vector3(slot.x, 0, slot.z);
  }
  if (selection.kind === "zone") {
    const zone = layout.zones.find((candidate) => candidate.name === selection.name);
    if (zone) return new THREE.Vector3(zone.x, 0, zone.z);
  }
  return null;
}
