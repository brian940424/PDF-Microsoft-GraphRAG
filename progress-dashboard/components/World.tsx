"use client";

import { Html, OrbitControls, OrthographicCamera } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState, type ComponentRef } from "react";
import * as THREE from "three";

import { LOT_SIZE, layoutSite, roadPoint, type Lot, type SiteLayout } from "@/lib/layout";
import type { Commit, ProgressData } from "@/lib/types";

import {
  Cone, ConstructionSite, Forklift, Hoverable, LotOutline, MapPin, PALETTE, Pallet,
  ShippingContainer, TowerCrane, Tree, Truck, Warehouse,
} from "./models";
import { commitColor, recentCommits, type CameraCommand, type Selection } from "./theme";

interface WorldProps {
  data: ProgressData;
  selection: Selection;
  statusFilter: Set<string>;
  cameraCommand: CameraCommand | null;
  onSelect: (selection: Selection) => void;
}

const VIEW_DIRECTION = new THREE.Vector3(1, 0.95, 1).normalize();
const VIEW_DISTANCE = 120;
const HUD_RIGHT = 400;

export default function World(props: WorldProps) {
  const layout = useMemo(() => layoutSite(props.data.plan, props.data.tests?.modules ?? []), [props.data]);
  const trucks = useMemo(() => recentCommits(props.data.git.commits), [props.data.git.commits]);
  return (
    <Canvas shadows flat dpr={[1, 2]} onPointerMissed={() => props.onSelect(null)}>
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
        shadow-camera-left={-layout.radius}
        shadow-camera-right={layout.radius}
        shadow-camera-top={layout.radius}
        shadow-camera-bottom={-layout.radius}
        shadow-camera-far={200}
      />
      <Ground layout={layout} />
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
      {trucks.map((commit, index) => (
        <MovingTruck
          key={commit.hash}
          commit={commit}
          road={layout.road}
          offset={index / trucks.length}
          selected={props.selection?.kind === "commit" && props.selection.hash === commit.hash}
          onSelect={props.onSelect}
        />
      ))}
      {layout.trees.map((tree, i) => <Tree key={i} position={[tree.x, 0, tree.z]} scale={tree.scale} />)}
      <CameraRig layout={layout} selection={props.selection} command={props.cameraCommand} />
    </Canvas>
  );
}

function Ground({ layout }: { layout: SiteLayout }) {
  const { road } = layout;
  const roadWidth = 3.6;
  const dashes = useMemo(() => {
    const result: { x: number; z: number; horizontal: boolean }[] = [];
    for (let x = -road.w / 2 + 2; x < road.w / 2 - 1; x += 2.4) result.push({ x, z: -road.d / 2, horizontal: true }, { x, z: road.d / 2, horizontal: true });
    for (let z = -road.d / 2 + 2; z < road.d / 2 - 1; z += 2.4) result.push({ x: -road.w / 2, z, horizontal: false }, { x: road.w / 2, z, horizontal: false });
    return result;
  }, [road]);
  return (
    <group>
      <mesh rotation-x={-Math.PI / 2} position-y={-0.03} receiveShadow>
        <planeGeometry args={[layout.radius * 8, layout.radius * 8]} />
        <meshStandardMaterial color="#e3e9f4" />
      </mesh>
      <mesh rotation-x={-Math.PI / 2} position-y={-0.02} receiveShadow>
        <planeGeometry args={[road.w + 20, road.d + 20]} />
        <meshStandardMaterial color="#d9eedb" />
      </mesh>
      <mesh rotation-x={-Math.PI / 2} position-y={-0.01} receiveShadow>
        <planeGeometry args={[road.w + roadWidth, road.d + roadWidth]} />
        <meshStandardMaterial color="#c7cfe3" />
      </mesh>
      <mesh rotation-x={-Math.PI / 2} position-y={0} receiveShadow>
        <planeGeometry args={[road.w - roadWidth, road.d - roadWidth]} />
        <meshStandardMaterial color="#f1f4fa" />
      </mesh>
      {dashes.map((dash, i) => (
        <mesh key={i} rotation-x={-Math.PI / 2} position={[dash.x, 0.005, dash.z]}>
          <planeGeometry args={dash.horizontal ? [1.2, 0.14] : [0.14, 1.2]} />
          <meshBasicMaterial color="#ffffff" />
        </mesh>
      ))}
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
      <Html position={[-w / 2 + 0.4, 0.2, -d / 2 + 0.8]} zIndexRange={[5, 0]} className="zone-label">
        {label}<span>{count}</span>
      </Html>
    </group>
  );
}

function LotView({ lot, selected, dimmed, onSelect }: {
  lot: Lot; selected: boolean; dimmed: boolean; onSelect: WorldProps["onSelect"];
}) {
  const [hovered, setHovered] = useState(false);
  const { item } = lot;
  const [width, depth, height] = item.priority === "高" ? [4.6, 3.6, 2.6] : item.priority === "中" ? [4.0, 3.2, 2.2] : [3.4, 2.8, 1.9];
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
        {selected && (
          <MapPin
            position={[0, item.status === "完成" ? height + depth * 0.22 + 0.6 : item.status === "進行中" ? height + 0.8 : 2.6, 0]}
            color={PALETTE.blue}
            scale={1.15}
          />
        )}
      </Hoverable>
      <Html position={[0, 0.2, LOT_SIZE / 2 - 0.55]} center zIndexRange={[6, 0]} className={`lot-sign ${selected ? "on" : ""}`}>
        {item.id}
        {item.uncommitted && <i title="已完成但未提交" />}
      </Html>
      {hovered && !selected && (
        <Html position={[0, 4.6, 0]} center zIndexRange={[7, 0]} className="tooltip3d">
          <b>{item.id}</b> {item.title.replace(/`/g, "")}
        </Html>
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
      {(hovered || selected) && (
        <Html position={[0, 4.4, 0]} center zIndexRange={[7, 0]} className="tooltip3d">
          <b>{slot.module.module}</b> {slot.module.passed}/{total} 通過
        </Html>
      )}
    </group>
  );
}

function MovingTruck({ commit, road, offset, selected, onSelect }: {
  commit: Commit; road: SiteLayout["road"]; offset: number; selected: boolean; onSelect: WorldProps["onSelect"];
}) {
  const ref = useRef<THREE.Group>(null);
  const [hovered, setHovered] = useState(false);
  const progress = useRef(offset);
  const perimeter = 2 * (road.w + road.d);
  useFrame((_, dt) => {
    if (!ref.current) return;
    if (!selected && !hovered) progress.current += (dt * 3.2) / perimeter;
    const point = roadPoint(road, progress.current);
    ref.current.position.set(point.x, 0, point.z);
    ref.current.rotation.y = point.heading;
  });
  return (
    <group ref={ref}>
      {/* 靠右行駛 */}
      <group position={[0, 0, 0.9]}>
        <Hoverable onClick={() => onSelect({ kind: "commit", hash: commit.hash })} onHover={setHovered}>
          <Truck stripe={commitColor(commit.type)} highlight={hovered || selected} />
        </Hoverable>
        {selected && <MapPin position={[0, 2.6, 0]} color={PALETTE.blue} scale={0.8} />}
        {(hovered || selected) && (
          <Html position={[0, 4.4, 0]} center zIndexRange={[7, 0]} className="tooltip3d">
            <b>{commit.hash}</b> {commit.subject}
          </Html>
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
function CameraRig({ layout, selection, command }: { layout: SiteLayout; selection: Selection; command: CameraCommand | null }) {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const { camera, size } = useThree();
  const ortho = camera as THREE.OrthographicCamera;
  // 右側卡片遮住約 HUD_RIGHT 像素：以 view offset 讓場景中心落在可見區域中央
  const hudRight = size.width > 1100 ? HUD_RIGHT : 0;
  const fitZoom = Math.min((size.width - hudRight) / (layout.radius * 1.9), size.height / (layout.radius * 1.35));
  const goal = useRef<CameraGoal | null>(null);
  const initialized = useRef(false);

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [command]);

  useEffect(() => {
    const instance = controls.current;
    if (!instance) return;
    const cancel = () => { goal.current = null; };
    instance.addEventListener("start", cancel);
    return () => instance.removeEventListener("start", cancel);
  }, []);

  useFrame((_, dt) => {
    const instance = controls.current;
    if (!instance || !goal.current) return;
    const k = 1 - Math.exp(-dt * 4);
    instance.target.lerp(goal.current.target, k);
    const direction = currentDirection().lerp(goal.current.direction, k).normalize();
    camera.position.copy(instance.target).add(direction.multiplyScalar(VIEW_DISTANCE));
    ortho.zoom += (goal.current.zoom - ortho.zoom) * k;
    ortho.updateProjectionMatrix();
    if (instance.target.distanceTo(goal.current.target) < 0.02 && Math.abs(ortho.zoom - goal.current.zoom) < 0.05
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
