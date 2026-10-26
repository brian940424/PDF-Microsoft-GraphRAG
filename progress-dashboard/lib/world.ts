import type { Rect } from "./layout";

/**
 * 工地周圍的「活的世界」（參考 WareTrack）：以馬路為界的格狀重劃區。
 * - 分區排成整齊的街廓，四周都是街道；外圍再一圈住宅街廓與外環道
 * - 路網上的雙向車流：靠右行駛、保持車距、路口號誌、不堵塞路口
 * - 卡車依路網開到物流倉庫，穿過倉庫卸貨後讓行匯入
 * 全部是純計算；畫面元件只讀取結果（「5 3D 儀表板」經 npm run export-world 共用同一份）。
 */

export interface Pose {
  x: number;
  z: number;
  heading: number;
}

/** 方塊零件（實例化繪製）：中心座標、尺寸、繞 y 旋轉與顏色 */
export interface Part {
  x: number;
  y: number;
  z: number;
  sx: number;
  sy: number;
  sz: number;
  rotation: number;
  color: string;
}

export const ROAD_WIDTH = 3.6;
export const LANE_OFFSET = 0.9;
const SIDEWALK = 1.0;
/** 相鄰街廓之間的距離（中間是街道） */
const STREET_GAP = 6.4;
/** 外圍住宅街廓的深度（街道中心線間距） */
const TOWN_DEPTH = 15;
const TRIM_SIGNAL = 2.6, TRIM_CORNER = 2.2, TRIM_MINOR = 0.6;
const SIGNAL = { green: 4.5, amber: 1, allRed: 2.5 };

// three.js 的 rotation.y = heading 會把車頭（+x）轉到 forward(heading)；行進方向右側為 right(heading)
const forward = (heading: number): [number, number] => [Math.cos(heading), -Math.sin(heading)];
const right = (heading: number): [number, number] => [Math.sin(heading), Math.cos(heading)];
const headingOf = (dx: number, dz: number) => Math.atan2(-dz, dx);
const angleDiff = (a: number, b: number) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
const hash = (n: number) => Math.abs(Math.sin(n * 12.9898 + 78.233) * 43758.5453) % 1;

/* ---------------- 格狀重劃：把分區排進街廓 ---------------- */

export interface BlockGrid {
  /** 每個區塊的中心（順序同輸入） */
  centers: { x: number; z: number }[];
  /** 街道中心線（含外框） */
  xs: number[];
  zs: number[];
  /** 街廓內沒被區塊占滿的空地，種樹當公園 */
  parks: Rect[];
  bounds: Rect;
}

/** 把大小不一的區塊排成整齊的棋盤格：同欄同寬、同列同深，格與格之間都是街道。 */
export function packBlocks(sizes: { w: number; d: number }[]): BlockGrid {
  if (!sizes.length) return { centers: [], xs: [-12, 12], zs: [-12, 12], parks: [], bounds: { x: 0, z: 0, w: 24, d: 24 } };
  // 大的先放，讓同欄／同列的區塊大小接近，空地較少
  const order = sizes.map((_, i) => i).sort((a, b) => sizes[b].w * sizes[b].d - sizes[a].w * sizes[a].d);
  const used = sizes.reduce((sum, size) => sum + size.w * size.d, 0);
  let best: { cols: number; colW: number[]; rowD: number[]; score: number } | null = null;
  for (let cols = 1; cols <= sizes.length; cols++) {
    const rows = Math.ceil(sizes.length / cols);
    const colW = new Array(cols).fill(0), rowD = new Array(rows).fill(0);
    order.forEach((index, k) => {
      colW[k % cols] = Math.max(colW[k % cols], sizes[index].w);
      rowD[Math.floor(k / cols)] = Math.max(rowD[Math.floor(k / cols)], sizes[index].d);
    });
    const width = sum(colW), depth = sum(rowD);
    const score = Math.abs(Math.log((width + STREET_GAP * cols) / (depth + STREET_GAP * rows) / 1.3)) + (1 - used / (width * depth));
    if (!best || score < best.score) best = { cols, colW, rowD, score };
  }
  const { cols, colW, rowD } = best!;
  const xs = [0], zs = [0];
  colW.forEach((w) => xs.push(xs[xs.length - 1] + w + STREET_GAP));
  rowD.forEach((d) => zs.push(zs[zs.length - 1] + d + STREET_GAP));
  const shiftX = xs[xs.length - 1] / 2, shiftZ = zs[zs.length - 1] / 2;
  for (let i = 0; i < xs.length; i++) xs[i] -= shiftX;
  for (let i = 0; i < zs.length; i++) zs[i] -= shiftZ;
  const centers: BlockGrid["centers"] = new Array(sizes.length);
  const parks: Rect[] = [];
  const cellCount = colW.length * rowD.length;
  for (let k = 0; k < cellCount; k++) {
    const c = k % cols, r = Math.floor(k / cols);
    const left = xs[c] + STREET_GAP / 2, top = zs[r] + STREET_GAP / 2;
    const index = order[k];
    if (index === undefined) {
      parks.push({ x: left + colW[c] / 2, z: top + rowD[r] / 2, w: colW[c], d: rowD[r] });
      continue;
    }
    const { w, d } = sizes[index];
    // 區塊靠左上，右側與下方的空地當公園
    centers[index] = { x: left + w / 2, z: top + d / 2 };
    if (colW[c] - w >= 2.5) parks.push({ x: left + w + (colW[c] - w) / 2, z: top + rowD[r] / 2, w: colW[c] - w, d: rowD[r] });
    if (rowD[r] - d >= 2.5) parks.push({ x: left + w / 2, z: top + d + (rowD[r] - d) / 2, w, d: rowD[r] - d });
  }
  return { centers, xs, zs, parks, bounds: { x: 0, z: 0, w: xs[xs.length - 1] - xs[0], d: zs[zs.length - 1] - zs[0] } };
}

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

/* ---------------- 路網 ---------------- */

export interface RoadNode {
  id: number;
  x: number;
  z: number;
  kind: "grid" | "depotEntry" | "depotExit";
  signal: boolean;
  incoming: number[];
  outgoing: number[];
  /** 號誌時相錯開的秒數 */
  offset: number;
}

export interface Lane {
  id: number;
  from: number;
  to: number;
  path: { x: number; z: number }[];
  cumulative: number[];
  length: number;
  heading: number;
  /** 倉庫卸貨道的編號；一般車道為 -1 */
  bay: number;
}

interface Connector {
  key: string;
  from: number;
  to: number;
  path: { x: number; z: number }[];
  cumulative: number[];
  length: number;
}

export interface Segment {
  x1: number;
  z1: number;
  x2: number;
  z2: number;
}

export interface DepotBay {
  lane: number;
  /** 卸貨道中心線在倉庫內的座標（沿道路 u、往外 v） */
  v: number;
  path: { x: number; z: number }[];
  length: number;
  stop: number;
  doors: [number, number];
  /** 完全離開車道、開始回到車道的路徑距離；兩者之外的路段仍壓在外側車道上 */
  leave: number;
  merge: number;
}

export interface Depot {
  x: number;
  z: number;
  rotation: number;
  length: number;
  depth: number;
  bays: DepotBay[];
  base: Pose;
  entry: number;
  exit: number;
  /** 駛入倉庫前的街道車道、駛離入口後的街道車道、匯入前與匯入後的街道車道 */
  entryLane: number;
  afterEntryLane: number;
  beforeExitLane: number;
  afterExitLane: number;
}

/** 住宅街廓（街道中心線圍成的矩形） */
export interface TownCell extends Rect {
  depot: boolean;
}

const DEPOT = { halfLength: 5, near: 3.8, far: 10.3, aisles: [5.4, 8.7], approach: 17 };
const DEPOT_SPAN = DEPOT.approach + 3.5;

/* ---------------- 車流 ---------------- */

export interface Vehicle {
  id: string;
  kind: "car" | "van" | "truck";
  /** 所在車道；在路口轉彎時 connector 不為 null */
  lane: number;
  connector: Connector | null;
  /** 車身中心在車道（或轉彎路徑）上的距離 */
  s: number;
  speed: number;
  max: number;
  length: number;
  color: number;
  /** 被選取或滑過時停車；後方車輛會跟著排隊 */
  held: boolean;
  /** 下一條車道（進入目前車道時決定） */
  next: number;
  /** 已通過停止線，不再因號誌停下 */
  committed: boolean;
  /** 卡車：距離下次進倉的秒數、是否正前往倉庫、卸貨剩餘秒數 */
  cooldown: number;
  toDepot: boolean;
  unload: number;
  unloaded: boolean;
}

export interface Traffic {
  road: Rect;
  nodes: RoadNode[];
  lanes: Lane[];
  segments: Segment[];
  cells: TownCell[];
  depot: Depot | null;
  vehicles: Vehicle[];
  /** 各卸貨道入口／出口捲門的開啟程度 0～1 */
  doors: [number, number][];
  occupants: (Vehicle | null)[];
  time: number;
  connectors: Map<string, Connector>;
  /** 各節點到倉庫入口車道的最短距離，卡車依此找路 */
  depotDistance: number[];
  seed: number;
}

const TRUCK = { length: 4.4, max: 3.2 };
const DEPOT_SPEED = 2.4;
const UNLOAD_SECONDS = 5;
const SAFE_GAP = 1.3;

/** 由街道格線建出整個城市：外圍住宅街廓、倉庫、路網與車輛。 */
export function createTraffic(grid: { xs: number[]; zs: number[] }, truckIds: string[]): Traffic {
  const xs = [grid.xs[0] - TOWN_DEPTH, ...grid.xs, grid.xs[grid.xs.length - 1] + TOWN_DEPTH];
  const zs = [grid.zs[0] - TOWN_DEPTH, ...grid.zs, grid.zs[grid.zs.length - 1] + TOWN_DEPTH];
  const plan = planDepot(xs, zs);
  // 棋盤街道；倉庫所在的住宅帶移除與它重疊的橫向街道，讓出一整塊地
  let segments: Segment[] = [];
  for (const z of zs) for (let i = 0; i < xs.length - 1; i++) segments.push({ x1: xs[i], z1: z, x2: xs[i + 1], z2: z });
  for (const x of xs) for (let i = 0; i < zs.length - 1; i++) segments.push({ x1: x, z1: zs[i], x2: x, z2: zs[i + 1] });
  if (plan) segments = segments.filter((segment) => !plan.removes(segment));
  const traffic: Traffic = {
    road: { x: 0, z: 0, w: grid.xs[grid.xs.length - 1] - grid.xs[0], d: grid.zs[grid.zs.length - 1] - grid.zs[0] },
    nodes: [], lanes: [], segments, cells: townCells(xs, zs, segments, plan), depot: null, vehicles: [], doors: [], occupants: [],
    time: 0, connectors: new Map(), depotDistance: [], seed: 20261025,
  };
  // 倉庫出入口把臨街的那段街道切開
  const graphSegments = plan ? splitAt(splitAt(segments, plan.entry), plan.exit) : segments;
  buildGraph(traffic, graphSegments, plan);
  if (plan) buildDepot(traffic, plan);
  placeVehicles(traffic, truckIds);
  return traffic;
}

interface DepotPlan {
  base: Pose;
  entry: { x: number; z: number };
  exit: { x: number; z: number };
  removes: (segment: Segment) => boolean;
  /** 倉庫占用的範圍（沿道路 u、往外 v），城鎮建物要避開 */
  contains: (x: number, z: number, margin: number) => boolean;
}

/** 倉庫放在外圍住宅帶，優先放畫面上方（遠側）；臨街長度要夠進出車道使用。 */
function planDepot(xs: number[], zs: number[]): DepotPlan | null {
  const last = (values: number[]) => values.length - 1;
  const sides = [
    { horizontal: true, line: zs[1], heading: Math.PI, lo: xs[0], hi: xs[last(xs)], band: [zs[0], zs[1]], nodes: xs },
    { horizontal: false, line: xs[last(xs) - 1], heading: Math.PI / 2, lo: zs[0], hi: zs[last(zs)], band: [xs[last(xs) - 1], xs[last(xs)]], nodes: zs },
    { horizontal: false, line: xs[1], heading: Math.PI * 1.5, lo: zs[0], hi: zs[last(zs)], band: [xs[0], xs[1]], nodes: zs },
    { horizontal: true, line: zs[last(zs) - 1], heading: 0, lo: xs[0], hi: xs[last(xs)], band: [zs[last(zs) - 1], zs[last(zs)]], nodes: xs },
  ];
  for (const side of sides) {
    const middle = (side.lo + side.hi) / 2;
    const candidates: number[] = [];
    for (let c = side.lo + DEPOT_SPAN + 3; c <= side.hi - DEPOT_SPAN - 3; c += 0.5) candidates.push(c);
    candidates.sort((a, b) => Math.abs(a - middle) - Math.abs(b - middle));
    // 出入口離路口至少 3.5，避免和路口轉彎路徑重疊
    const center = candidates.find((c) => side.nodes.every((n) => Math.abs(n - (c - DEPOT.approach)) >= 3.5 && Math.abs(n - (c + DEPOT.approach)) >= 3.5));
    if (center === undefined) continue;
    const base = side.horizontal ? { x: center, z: side.line, heading: side.heading } : { x: side.line, z: center, heading: side.heading };
    const at = (u: number, v: number) => localPoint(base, u, v);
    const [bandLo, bandHi] = side.band;
    return {
      base, entry: at(-DEPOT.approach, 0), exit: at(DEPOT.approach, 0),
      removes: (segment) => {
        // 住宅帶裡與倉庫重疊、垂直於臨街道路的街道
        const perpendicular = side.horizontal ? segment.x1 === segment.x2 : segment.z1 === segment.z2;
        if (!perpendicular) return false;
        const along = side.horizontal ? segment.x1 : segment.z1;
        const across = side.horizontal ? [segment.z1, segment.z2] : [segment.x1, segment.x2];
        return Math.abs(along - center) < DEPOT_SPAN && Math.min(...across) >= bandLo - 1e-6 && Math.max(...across) <= bandHi + 1e-6;
      },
      contains: (x, z, margin) => {
        const [fx, fz] = forward(base.heading), [rx, rz] = right(base.heading);
        const u = (x - base.x) * fx + (z - base.z) * fz, v = (x - base.x) * rx + (z - base.z) * rz;
        return Math.abs(u) < DEPOT.approach + margin && v > -margin && v < DEPOT.far + 1.5 + margin;
      },
    };
  }
  return null;
}

function localPoint(base: Pose, u: number, v: number): { x: number; z: number } {
  const [fx, fz] = forward(base.heading), [rx, rz] = right(base.heading);
  return { x: base.x + fx * u + rx * v, z: base.z + fz * u + rz * v };
}

/** 把通過 point 的街道切成兩段。 */
function splitAt(segments: Segment[], point: { x: number; z: number }): Segment[] {
  const result: Segment[] = [];
  for (const segment of segments) {
    const onX = Math.abs(segment.z1 - segment.z2) < 1e-6 && Math.abs(point.z - segment.z1) < 1e-6
      && point.x > Math.min(segment.x1, segment.x2) && point.x < Math.max(segment.x1, segment.x2);
    const onZ = Math.abs(segment.x1 - segment.x2) < 1e-6 && Math.abs(point.x - segment.x1) < 1e-6
      && point.z > Math.min(segment.z1, segment.z2) && point.z < Math.max(segment.z1, segment.z2);
    if (onX || onZ) result.push({ ...segment, x2: point.x, z2: point.z }, { x1: point.x, z1: point.z, x2: segment.x2, z2: segment.z2 });
    else result.push(segment);
  }
  return result;
}

function townCells(xs: number[], zs: number[], segments: Segment[], plan: DepotPlan | null): TownCell[] {
  const has = (x1: number, z1: number, x2: number, z2: number) => segments.some((s) =>
    (Math.abs(s.x1 - x1) < 1e-6 && Math.abs(s.z1 - z1) < 1e-6 && Math.abs(s.x2 - x2) < 1e-6 && Math.abs(s.z2 - z2) < 1e-6));
  const cells: TownCell[] = [];
  const add = (x1: number, x2: number, z1: number, z2: number) => {
    const cell = { x: (x1 + x2) / 2, z: (z1 + z2) / 2, w: x2 - x1, d: z2 - z1 };
    cells.push({ ...cell, depot: plan ? overlaps(cell, plan) : false });
  };
  const last = xs.length - 1, lastZ = zs.length - 1;
  // 上、下兩條住宅帶（含四角）：以仍存在的縱向街道切格
  for (const [z1, z2] of [[zs[0], zs[1]], [zs[lastZ - 1], zs[lastZ]]]) {
    const cuts = xs.filter((x, i) => i === 0 || i === last || has(x, z1, x, z2));
    for (let i = 0; i < cuts.length - 1; i++) add(cuts[i], cuts[i + 1], z1, z2);
  }
  // 左、右兩條住宅帶：以仍存在的橫向街道切格
  for (const [x1, x2] of [[xs[0], xs[1]], [xs[last - 1], xs[last]]]) {
    const cuts = zs.slice(1, lastZ).filter((z, i, list) => i === 0 || i === list.length - 1 || has(x1, z, x2, z));
    for (let i = 0; i < cuts.length - 1; i++) add(x1, x2, cuts[i], cuts[i + 1]);
  }
  return cells;
}

function overlaps(cell: Rect, plan: DepotPlan): boolean {
  for (let fx = -0.45; fx <= 0.45; fx += 0.15) for (let fz = -0.45; fz <= 0.45; fz += 0.15) {
    if (plan.contains(cell.x + cell.w * fx, cell.z + cell.d * fz, 0)) return true;
  }
  return false;
}

function buildGraph(traffic: Traffic, segments: Segment[], plan: DepotPlan | null): void {
  const keyOf = (x: number, z: number) => `${x.toFixed(3)},${z.toFixed(3)}`;
  const index = new Map<string, number>();
  const nodeAt = (x: number, z: number) => {
    const key = keyOf(x, z);
    if (!index.has(key)) {
      const kind = plan && Math.hypot(x - plan.entry.x, z - plan.entry.z) < 1e-6 ? "depotEntry"
        : plan && Math.hypot(x - plan.exit.x, z - plan.exit.z) < 1e-6 ? "depotExit" : "grid";
      index.set(key, traffic.nodes.length);
      traffic.nodes.push({ id: traffic.nodes.length, x, z, kind, signal: false, incoming: [], outgoing: [], offset: 0 });
    }
    return index.get(key)!;
  };
  const pairs = segments.map((segment) => [nodeAt(segment.x1, segment.z1), nodeAt(segment.x2, segment.z2)]);
  const degree = new Map<number, number>();
  for (const [a, b] of pairs) { degree.set(a, (degree.get(a) ?? 0) + 1); degree.set(b, (degree.get(b) ?? 0) + 1); }
  for (const node of traffic.nodes) {
    node.signal = node.kind === "grid" && (degree.get(node.id) ?? 0) >= 3;
    node.offset = hash(node.x * 0.37 + node.z * 1.91) * 40;
  }
  const trim = (id: number) => {
    const node = traffic.nodes[id];
    return node.kind !== "grid" ? TRIM_MINOR : node.signal ? TRIM_SIGNAL : TRIM_CORNER;
  };
  for (const [a, b] of pairs) {
    for (const [from, to] of [[a, b], [b, a]]) {
      const p = traffic.nodes[from], q = traffic.nodes[to];
      const length = Math.hypot(q.x - p.x, q.z - p.z);
      const dx = (q.x - p.x) / length, dz = (q.z - p.z) / length;
      const heading = headingOf(dx, dz);
      const [rx, rz] = right(heading);
      const start = { x: p.x + dx * trim(from) + rx * LANE_OFFSET, z: p.z + dz * trim(from) + rz * LANE_OFFSET };
      const end = { x: q.x - dx * trim(to) + rx * LANE_OFFSET, z: q.z - dz * trim(to) + rz * LANE_OFFSET };
      addLane(traffic, from, to, [start, end], heading, -1);
    }
  }
}

function addLane(traffic: Traffic, from: number, to: number, path: { x: number; z: number }[], heading: number, bay: number): Lane {
  const cumulative = cumulate(path);
  const lane: Lane = { id: traffic.lanes.length, from, to, path, cumulative, length: cumulative[cumulative.length - 1], heading, bay };
  traffic.lanes.push(lane);
  traffic.nodes[from].outgoing.push(lane.id);
  traffic.nodes[to].incoming.push(lane.id);
  return lane;
}

function cumulate(path: { x: number; z: number }[]): number[] {
  const cumulative = [0];
  for (let k = 1; k < path.length; k++) cumulative.push(cumulative[k - 1] + Math.hypot(path[k].x - path[k - 1].x, path[k].z - path[k - 1].z));
  return cumulative;
}

function buildDepot(traffic: Traffic, plan: DepotPlan): void {
  const { base } = plan;
  const entry = traffic.nodes.findIndex((node) => node.kind === "depotEntry");
  const exit = traffic.nodes.findIndex((node) => node.kind === "depotExit");
  const along = (laneId: number) => angleDiff(traffic.lanes[laneId].heading, base.heading) < 0.1;
  const bays = DEPOT.aisles.map((v, bayIndex) => {
    const control: [number, number][] = [
      [-DEPOT.approach + TRIM_MINOR, LANE_OFFSET], [-DEPOT.approach + 4, LANE_OFFSET], [-DEPOT.halfLength - 3, v], [-DEPOT.halfLength, v], [0, v],
      [DEPOT.halfLength, v], [DEPOT.halfLength + 3, v], [DEPOT.approach - 4, LANE_OFFSET], [DEPOT.approach - TRIM_MINOR, LANE_OFFSET],
    ];
    const smooth = chaikin(control, 3);
    const path = smooth.map(([u, w]) => localPoint(base, u, w));
    const lane = addLane(traffic, entry, exit, path, base.heading, bayIndex);
    const distanceAt = (u: number) => {
      let best = 0;
      smooth.forEach(([pu, pv], k) => {
        if (Math.hypot(pu - u, pv - v) < Math.hypot(smooth[best][0] - u, smooth[best][1] - v)) best = k;
      });
      return lane.cumulative[best];
    };
    const offLane = smooth.map(([, w]) => w >= ROAD_WIDTH / 2 + 0.8);
    return {
      lane: lane.id, v, path, length: lane.length, stop: distanceAt(0),
      doors: [distanceAt(-DEPOT.halfLength), distanceAt(DEPOT.halfLength)] as [number, number],
      leave: lane.cumulative[offLane.indexOf(true)], merge: lane.cumulative[offLane.lastIndexOf(true)],
    };
  });
  const center = localPoint(base, 0, (DEPOT.near + DEPOT.far) / 2);
  traffic.depot = {
    x: center.x, z: center.z, rotation: base.heading, length: DEPOT.halfLength * 2, depth: DEPOT.far - DEPOT.near, bays, base,
    entry, exit,
    entryLane: traffic.nodes[entry].incoming.find((id) => along(id) && traffic.lanes[id].bay < 0)!,
    afterEntryLane: traffic.nodes[entry].outgoing.find((id) => along(id) && traffic.lanes[id].bay < 0)!,
    beforeExitLane: traffic.nodes[exit].incoming.find((id) => along(id) && traffic.lanes[id].bay < 0)!,
    afterExitLane: traffic.nodes[exit].outgoing.find((id) => along(id) && traffic.lanes[id].bay < 0)!,
  };
  traffic.doors = bays.map(() => [0, 0]);
  traffic.occupants = bays.map(() => null);
  // 反向 Dijkstra：各節點經由入口車道抵達倉庫的最短距離
  const distance = new Array(traffic.nodes.length).fill(Infinity);
  const entryLane = traffic.lanes[traffic.depot.entryLane];
  distance[entryLane.from] = entryLane.length;
  const queue = [entryLane.from];
  while (queue.length) {
    queue.sort((a, b) => distance[a] - distance[b]);
    const node = queue.shift()!;
    for (const laneId of traffic.nodes[node].incoming) {
      const lane = traffic.lanes[laneId];
      if (lane.bay >= 0) continue;
      const candidate = distance[node] + lane.length;
      if (candidate < distance[lane.from]) { distance[lane.from] = candidate; queue.push(lane.from); }
    }
  }
  traffic.depotDistance = distance;
}

/** Chaikin 切角平滑，保留起點與終點。 */
function chaikin(points: [number, number][], iterations: number): [number, number][] {
  let current = points;
  for (let i = 0; i < iterations; i++) {
    const next: [number, number][] = [current[0]];
    for (let k = 0; k < current.length - 1; k++) {
      const [ax, az] = current[k], [bx, bz] = current[k + 1];
      next.push([ax * 0.75 + bx * 0.25, az * 0.75 + bz * 0.25], [ax * 0.25 + bx * 0.75, az * 0.25 + bz * 0.75]);
    }
    next.push(current[current.length - 1]);
    current = next;
  }
  return current;
}

/** 路口的轉彎路徑：從進入車道終點到下一條車道起點的二次貝茲曲線。 */
function connector(traffic: Traffic, from: number, to: number): Connector {
  const key = `${from}>${to}`;
  const cached = traffic.connectors.get(key);
  if (cached) return cached;
  const a = traffic.lanes[from], b = traffic.lanes[to];
  const p0 = a.path[a.path.length - 1], p1 = a.path[a.path.length - 2], q0 = b.path[0], q1 = b.path[1];
  const d1 = norm(p0.x - p1.x, p0.z - p1.z), d2 = norm(q1.x - q0.x, q1.z - q0.z);
  // 兩條車道延長線的交點當控制點；平行（直行）時取中點
  const cross = d1[0] * d2[1] - d1[1] * d2[0];
  let control = { x: (p0.x + q0.x) / 2, z: (p0.z + q0.z) / 2 };
  if (Math.abs(cross) > 1e-3) {
    const t = ((q0.x - p0.x) * d2[1] - (q0.z - p0.z) * d2[0]) / cross;
    control = { x: p0.x + d1[0] * t, z: p0.z + d1[1] * t };
  }
  const path = [];
  for (let k = 0; k <= 10; k++) {
    const t = k / 10, u = 1 - t;
    path.push({ x: u * u * p0.x + 2 * u * t * control.x + t * t * q0.x, z: u * u * p0.z + 2 * u * t * control.z + t * t * q0.z });
  }
  const cumulative = cumulate(path);
  const result = { key, from, to, path, cumulative, length: Math.max(0.01, cumulative[cumulative.length - 1]) };
  traffic.connectors.set(key, result);
  return result;
}

function norm(x: number, z: number): [number, number] {
  const length = Math.hypot(x, z) || 1;
  return [x / length, z / length];
}

/** 號誌：路口各進入方向輪流綠燈（保護時相），每次綠燈後接黃燈與全紅清道。 */
export function lightState(traffic: Traffic, nodeId: number, laneId: number): "green" | "amber" | "red" {
  const node = traffic.nodes[nodeId];
  if (!node.signal) return "green";
  const approaches = [...node.incoming].sort((a, b) => traffic.lanes[a].heading - traffic.lanes[b].heading);
  const phase = SIGNAL.green + SIGNAL.amber + SIGNAL.allRed;
  const local = (traffic.time + node.offset) % (phase * approaches.length);
  const active = Math.floor(local / phase), within = local - active * phase;
  if (approaches[active] !== laneId) return "red";
  return within < SIGNAL.green ? "green" : within < SIGNAL.green + SIGNAL.amber ? "amber" : "red";
}

function random(traffic: Traffic): number {
  traffic.seed = (traffic.seed * 1664525 + 1013904223) % 4294967296;
  return traffic.seed / 4294967296;
}

function placeVehicles(traffic: Traffic, truckIds: string[]): void {
  const total = sum(traffic.lanes.filter((lane) => lane.bay < 0).map((lane) => lane.length));
  const cars = Math.max(8, Math.min(44, Math.round(total / 30)));
  truckIds.forEach((id, index) => addVehicle(traffic, makeTruck(id, index)));
  for (let k = 0; k < cars; k++) addVehicle(traffic, makeCar(`car-${k}`, k));
}

function makeTruck(id: string, index: number): Vehicle {
  return { id, kind: "truck", lane: -1, connector: null, s: 0, speed: 0, max: TRUCK.max, length: TRUCK.length, color: 0, held: false,
    next: -1, committed: false, cooldown: 4 + index * 7, toDepot: false, unload: 0, unloaded: false };
}

function makeCar(id: string, seed: number): Vehicle {
  const van = hash(seed + 0.5) < 0.22;
  return { id, kind: van ? "van" : "car", lane: -1, connector: null, s: 0, speed: 0, max: 2.9 + hash(seed) * 0.9, length: van ? 2.6 : 2.2,
    color: Math.floor(hash(seed + 1.7) * 7), held: false, next: -1, committed: false, cooldown: 0, toDepot: false, unload: 0, unloaded: false };
}

/** 把車放到隨機一條有空位的街道車道上。 */
function addVehicle(traffic: Traffic, vehicle: Vehicle): void {
  const lanes = traffic.lanes.filter((lane) => lane.bay < 0 && lane.length > vehicle.length + 3);
  for (let attempt = 0; attempt < 200; attempt++) {
    const lane = lanes[Math.floor(random(traffic) * lanes.length)];
    const s = vehicle.length / 2 + 0.5 + random(traffic) * (lane.length - vehicle.length - 1);
    const clear = traffic.vehicles.every((other) => other.connector || other.lane !== lane.id || Math.abs(other.s - s) > (other.length + vehicle.length) / 2 + 2);
    if (!clear) continue;
    Object.assign(vehicle, { lane: lane.id, s, speed: vehicle.max * 0.6 });
    traffic.vehicles.push(vehicle);
    vehicle.next = chooseNext(traffic, vehicle);
    return;
  }
}

/** 資料更新後同步卡車（提交）清單：保留既有卡車，移除消失的，新的放到空位。 */
export function syncTrucks(traffic: Traffic, truckIds: string[]): void {
  const wanted = new Set(truckIds);
  traffic.vehicles = traffic.vehicles.filter((vehicle) => vehicle.kind !== "truck" || wanted.has(vehicle.id));
  traffic.occupants = traffic.occupants.map((occupant) => (occupant && wanted.has(occupant.id) ? occupant : null));
  const existing = new Set(traffic.vehicles.map((vehicle) => vehicle.id));
  truckIds.forEach((id, index) => { if (!existing.has(id)) addVehicle(traffic, makeTruck(id, index)); });
}

/** 進入一條車道時決定下一條：一般車隨機（直行機率較高、不迴轉），想進倉的卡車沿最短路徑。 */
function chooseNext(traffic: Traffic, vehicle: Vehicle): number {
  const lane = traffic.lanes[vehicle.lane];
  const node = traffic.nodes[lane.to];
  const depot = traffic.depot;
  if (lane.bay >= 0) return depot!.afterExitLane;
  const options = node.outgoing.filter((id) => traffic.lanes[id].bay < 0 && angleDiff(traffic.lanes[id].heading, lane.heading) < Math.PI * 0.8);
  if (vehicle.kind === "truck" && depot && vehicle.toDepot) {
    if (lane.id === depot.entryLane) {
      const bay = traffic.occupants.indexOf(null);
      if (bay >= 0) {
        traffic.occupants[bay] = vehicle;
        return depot.bays[bay].lane;
      }
      // 卸貨道都在使用中：先繞一圈再回來
      vehicle.toDepot = false;
      vehicle.cooldown = 10;
    } else {
      // 入口車道直接通往倉庫；反向駛入入口節點的車道進不了倉庫
      const score = (id: number) => id === depot.entryLane ? 0
        : traffic.lanes[id].to === depot.entry ? Infinity : traffic.lanes[id].length + traffic.depotDistance[traffic.lanes[id].to];
      const best = options.sort((a, b) => score(a) - score(b))[0];
      if (best !== undefined) return best;
    }
  }
  if (!options.length) return node.outgoing.find((id) => traffic.lanes[id].bay < 0) ?? lane.id;
  const weights = options.map((id) => (angleDiff(traffic.lanes[id].heading, lane.heading) < 0.3 ? 3 : 1));
  let pick = random(traffic) * sum(weights);
  for (let k = 0; k < options.length; k++) {
    pick -= weights[k];
    if (pick <= 0) return options[k];
  }
  return options[options.length - 1];
}

interface Occupant { d: number; length: number; vehicle: Vehicle }

/** 每條車道上的佔用：在車道上的車（距離＝s）、正要進入該車道的轉彎中車輛（距離＝負的剩餘長度），以及倉庫卡車壓在車道上的部分。 */
function occupancy(traffic: Traffic): Map<number, Occupant[]> {
  const map = new Map<number, Occupant[]>();
  const push = (lane: number, item: Occupant) => {
    const list = map.get(lane);
    if (list) list.push(item);
    else map.set(lane, [item]);
  };
  const depot = traffic.depot;
  for (const vehicle of traffic.vehicles) {
    if (vehicle.connector) push(vehicle.connector.to, { d: vehicle.s - vehicle.connector.length, length: vehicle.length, vehicle });
    else push(vehicle.lane, { d: vehicle.s, length: vehicle.length, vehicle });
    if (depot && !vehicle.connector && traffic.lanes[vehicle.lane].bay >= 0) {
      const bay = depot.bays[traffic.lanes[vehicle.lane].bay];
      if (vehicle.s < bay.leave) push(depot.afterEntryLane, { d: vehicle.s, length: vehicle.length, vehicle });
      if (vehicle.s > bay.merge) push(depot.beforeExitLane, { d: traffic.lanes[depot.beforeExitLane].length - (bay.length - vehicle.s), length: vehicle.length, vehicle });
    }
  }
  return map;
}

/** 下一條車道入口是否有空位（不堵塞路口）。 */
function hasRoom(occupied: Map<number, Occupant[]>, vehicle: Vehicle, laneId: number, own: Connector): boolean {
  return (occupied.get(laneId) ?? []).every((item) => {
    if (item.vehicle === vehicle) return true;
    if (item.vehicle.connector) return item.vehicle.connector === own;
    return item.d - item.length / 2 >= vehicle.length + 1.5;
  });
}

/** 停止線前是否可以進入路口。 */
function mayEnter(traffic: Traffic, occupied: Map<number, Occupant[]>, vehicle: Vehicle): boolean {
  const lane = traffic.lanes[vehicle.lane];
  const own = connector(traffic, lane.id, vehicle.next);
  if (!hasRoom(occupied, vehicle, vehicle.next, own)) return false;
  const depot = traffic.depot;
  if (lane.bay >= 0 && depot) {
    // 倉庫出口讓行：匯入路段後方與正在通過出口的車輛優先
    const before = traffic.lanes[depot.beforeExitLane];
    const bay = depot.bays[lane.bay];
    const busy = (occupied.get(before.id) ?? []).some((item) => item.vehicle !== vehicle
      && item.d + item.length / 2 > before.length - (bay.length - bay.merge) - 9);
    const crossing = traffic.vehicles.some((other) => other.connector?.to === depot.afterExitLane && other !== vehicle);
    return !busy && !crossing;
  }
  return lightState(traffic, lane.to, lane.id) === "green";
}

export function stepTraffic(traffic: Traffic, dt: number): void {
  if (dt <= 0) return;
  traffic.time += dt;
  const occupied = occupancy(traffic);
  const moves = traffic.vehicles.map((vehicle) => planMove(traffic, occupied, vehicle, dt));
  traffic.vehicles.forEach((vehicle, k) => advance(traffic, vehicle, moves[k]));
  const depot = traffic.depot;
  if (!depot) return;
  depot.bays.forEach((bay, index) => {
    const truck = traffic.occupants[index];
    const doors = traffic.doors[index];
    for (const door of [0, 1] as const) {
      const inside = truck && !truck.connector && truck.lane === bay.lane;
      const near = inside ? Math.abs(truck.s - bay.doors[door]) < 4.6 && !(truck.unload > 0) : false;
      doors[door] += Math.max(-1.2 * dt, Math.min(1.2 * dt, (near ? 1 : 0) - doors[door]));
    }
  });
}

/** 依前車、停止線、捲門與卸貨點計算這一步能前進多少。 */
function planMove(traffic: Traffic, occupied: Map<number, Occupant[]>, vehicle: Vehicle, dt: number): number {
  if (vehicle.kind === "truck" && !vehicle.toDepot && traffic.depot) {
    vehicle.cooldown -= dt;
    if (vehicle.cooldown <= 0) vehicle.toDepot = true;
  }
  if (vehicle.unload > 0) {
    vehicle.unload -= dt;
    vehicle.speed = 0;
    return 0;
  }
  let gap = Infinity, stop = Infinity;
  const lane = traffic.lanes[vehicle.lane];
  if (vehicle.connector) {
    const own = vehicle.connector;
    const myDistance = vehicle.s - own.length;
    for (const item of occupied.get(own.to) ?? []) {
      if (item.vehicle !== vehicle && item.d > myDistance) gap = Math.min(gap, item.d - myDistance - (item.length + vehicle.length) / 2);
    }
  } else {
    for (const item of occupied.get(lane.id) ?? []) {
      if (item.vehicle !== vehicle && item.d > vehicle.s) gap = Math.min(gap, item.d - vehicle.s - (item.length + vehicle.length) / 2);
    }
    const remaining = lane.length - vehicle.s;
    const bay = lane.bay >= 0 ? traffic.depot!.bays[lane.bay] : null;
    // 倉庫卸貨道在讓行點等待；一般車道在停止線等待
    const line = bay ? bay.merge - vehicle.s : remaining - vehicle.length / 2 - 0.3;
    let go = vehicle.committed || line >= 4;
    if (!go) {
      if (vehicle.next >= 0 && mayEnter(traffic, occupied, vehicle)) {
        go = true;
        vehicle.committed = line < 1;
      } else stop = Math.min(stop, line);
    }
    if (go) {
      // 看得到下一條車道上的車
      const own = connector(traffic, lane.id, vehicle.next);
      const myDistance = -(remaining + own.length);
      for (const item of occupied.get(vehicle.next) ?? []) {
        if (item.vehicle !== vehicle && item.d > myDistance) gap = Math.min(gap, item.d - myDistance - (item.length + vehicle.length) / 2);
      }
    }
    if (bay) {
      if (!vehicle.unloaded) stop = Math.min(stop, bay.stop - vehicle.s);
      for (const door of [0, 1] as const) {
        const before = bay.doors[door] - vehicle.s - vehicle.length / 2;
        if (before > -0.5 && before < 2.8 && traffic.doors[lane.bay][door] < 0.9) stop = Math.min(stop, Math.max(0, before - 0.2));
      }
    }
  }
  const max = lane.bay >= 0 && !vehicle.connector ? DEPOT_SPEED : vehicle.max;
  const room = Math.min(gap - SAFE_GAP, stop);
  const target = vehicle.held && !vehicle.connector ? 0 : max * Math.max(0, Math.min(1, room / 4.5));
  vehicle.speed += Math.max(-9 * dt, Math.min(3 * dt, target - vehicle.speed));
  vehicle.speed = Math.max(0, vehicle.speed);
  return Math.max(0, Math.min(vehicle.speed * dt, gap - 0.3, stop));
}

function advance(traffic: Traffic, vehicle: Vehicle, distance: number): void {
  vehicle.s += distance;
  const depot = traffic.depot;
  for (let guard = 0; guard < 4; guard++) {
    if (vehicle.connector) {
      if (vehicle.s < vehicle.connector.length) return;
      vehicle.s -= vehicle.connector.length;
      vehicle.lane = vehicle.connector.to;
      vehicle.connector = null;
      vehicle.committed = false;
      vehicle.next = chooseNext(traffic, vehicle);
      continue;
    }
    const lane = traffic.lanes[vehicle.lane];
    if (lane.bay >= 0 && !vehicle.unloaded && vehicle.s >= depot!.bays[lane.bay].stop - 1e-6) {
      vehicle.unloaded = true;
      vehicle.unload = UNLOAD_SECONDS;
      vehicle.speed = 0;
    }
    if (vehicle.s < lane.length) return;
    if (lane.bay >= 0) {
      // 離開倉庫：釋出卸貨道，過一陣子再回來
      traffic.occupants[lane.bay] = null;
      vehicle.toDepot = false;
      vehicle.unloaded = false;
      vehicle.cooldown = 25 + hash(vehicle.id.length + traffic.time) * 25;
    }
    vehicle.s -= lane.length;
    vehicle.connector = connector(traffic, lane.id, vehicle.next);
  }
}

/** 車輛目前的世界座標與方向。 */
export function vehiclePose(traffic: Traffic, vehicle: Vehicle): Pose {
  if (vehicle.connector) return polylinePose(vehicle.connector.path, vehicle.connector.cumulative, vehicle.s);
  const lane = traffic.lanes[vehicle.lane];
  return polylinePose(lane.path, lane.cumulative, vehicle.s);
}

function polylinePose(path: { x: number; z: number }[], cumulative: number[], distance: number): Pose {
  const total = cumulative[cumulative.length - 1];
  const d = Math.max(0, Math.min(total, distance));
  let k = 1;
  while (k < cumulative.length - 1 && cumulative[k] < d) k++;
  const a = path[k - 1], b = path[k];
  const span = cumulative[k] - cumulative[k - 1] || 1;
  const t = (d - cumulative[k - 1]) / span;
  return { x: a.x + (b.x - a.x) * t, z: a.z + (b.z - a.z) * t, heading: headingOf(b.x - a.x, b.z - a.z) };
}

/** 卸貨道燈號：綠＝空、黃＝進出中、紅＝卸貨中（同 WareTrack 的碼頭燈）。 */
export function bayState(traffic: Traffic, index: number): "free" | "moving" | "unloading" {
  const truck = traffic.occupants[index];
  return !truck ? "free" : truck.unload > 0 ? "unloading" : "moving";
}

/* ---------------- 道路標線與號誌燈 ---------------- */

export interface StreetParts {
  sidewalks: Part[];
  asphalt: Part[];
  /** 中線虛線、斑馬線與停止線 */
  markings: Part[];
  /** 號誌燈桿與燈箱（靜態） */
  poles: Part[];
}

export interface SignalLamp {
  x: number;
  y: number;
  z: number;
  node: number;
  lane: number;
}

export function streetParts(traffic: Traffic): StreetParts {
  const parts: StreetParts = { sidewalks: [], asphalt: [], markings: [], poles: [] };
  const white = "#ffffff";
  for (const segment of traffic.segments) {
    const length = Math.hypot(segment.x2 - segment.x1, segment.z2 - segment.z1);
    const heading = headingOf(segment.x2 - segment.x1, segment.z2 - segment.z1);
    const cx = (segment.x1 + segment.x2) / 2, cz = (segment.z1 + segment.z2) / 2;
    parts.sidewalks.push({ x: cx, y: 0.006, z: cz, sx: length + ROAD_WIDTH + SIDEWALK * 2, sy: 0.012, sz: ROAD_WIDTH + SIDEWALK * 2, rotation: heading, color: "#e4e9f2" });
    parts.asphalt.push({ x: cx, y: 0.014, z: cz, sx: length + ROAD_WIDTH, sy: 0.012, sz: ROAD_WIDTH, rotation: heading, color: "#c7cfe3" });
    const [fx, fz] = forward(heading);
    for (let u = TRIM_SIGNAL + 0.8; u < length - TRIM_SIGNAL - 1.4; u += 2.4) {
      parts.markings.push({ x: segment.x1 + fx * (u + 0.6), y: 0.022, z: segment.z1 + fz * (u + 0.6), sx: 1.2, sy: 0.01, sz: 0.14, rotation: heading, color: white });
    }
  }
  for (const node of traffic.nodes) {
    if (!node.signal) continue;
    for (const laneId of node.incoming) {
      const lane = traffic.lanes[laneId];
      const end = lane.path[lane.path.length - 1];
      const [fx, fz] = forward(lane.heading), [rx, rz] = right(lane.heading);
      // 停止線（本車道）與斑馬線（整條路寬）
      parts.markings.push({ x: end.x - fx * 0.15, y: 0.022, z: end.z - fz * 0.15, sx: 0.18, sy: 0.01, sz: ROAD_WIDTH / 2 - 0.15, rotation: lane.heading, color: white });
      const zebra = { x: node.x - fx * (TRIM_SIGNAL - 0.35) , z: node.z - fz * (TRIM_SIGNAL - 0.35) };
      for (let k = -2; k <= 2; k++) {
        parts.markings.push({ x: zebra.x + rx * k * 0.68, y: 0.022, z: zebra.z + rz * k * 0.68, sx: 0.55, sy: 0.01, sz: 0.36, rotation: lane.heading, color: white });
      }
      const pole = { x: end.x + rx * (LANE_OFFSET + 0.9) + fx * 0.3, z: end.z + rz * (LANE_OFFSET + 0.9) + fz * 0.3 };
      parts.poles.push({ x: pole.x, y: 1.1, z: pole.z, sx: 0.12, sy: 2.2, sz: 0.12, rotation: lane.heading, color: "#3a4256" });
      parts.poles.push({ x: pole.x, y: 2.3, z: pole.z, sx: 0.32, sy: 0.62, sz: 0.32, rotation: lane.heading, color: "#2b3140" });
    }
  }
  return parts;
}

/** 號誌燈位置（每個路口每個進入方向一盞），顏色由 lightState 決定。 */
export function signalLamps(traffic: Traffic): SignalLamp[] {
  const lamps: SignalLamp[] = [];
  for (const node of traffic.nodes) {
    if (!node.signal) continue;
    for (const laneId of node.incoming) {
      const lane = traffic.lanes[laneId];
      const end = lane.path[lane.path.length - 1];
      const [fx, fz] = forward(lane.heading), [rx, rz] = right(lane.heading);
      // 燈面朝向來車
      lamps.push({ x: end.x + rx * (LANE_OFFSET + 0.9) + fx * 0.3 - fx * 0.17, y: 2.3, z: end.z + rz * (LANE_OFFSET + 0.9) + fz * 0.3 - fz * 0.17, node: node.id, lane: laneId });
    }
  }
  return lamps;
}

/* ---------------- 城鎮：住宅街廓沿街建物與公園 ---------------- */

export interface Town {
  boxes: Part[];
  /** 山形屋頂：底部中心在 y，sy 為屋脊高 */
  roofs: Part[];
  lawns: Part[];
  trees: { x: number; z: number; scale: number }[];
  /** 建物外框，用於測試是否與道路、倉庫重疊 */
  footprints: { x: number; z: number; r: number }[];
}

const WALLS = ["#fdf6ec", "#f3f7fb", "#e8f1ff", "#e9f7ef", "#fff1e6", "#f5efff"];
const ROOFS = ["#e06c5a", "#5a6b86", "#b0714a", "#4f7bd9", "#7c8796"];
const ACCENTS = ["#f6a821", "#1fb5a8", "#ec5a5f", "#8b5cf6", "#3563f0"];
const GLASS = "#9ec1ea", DARK = "#33405a";
const LOT_STEP = 5.2, LOT_DEPTH = 2.7, SETBACK = 0.9;

/** 住宅街廓沿四邊臨街蓋房子（正面朝街），中間是後院樹木；分區街廓的空地當公園。 */
export function layoutTown(traffic: Traffic, parks: Rect[] = []): Town {
  const town: Town = { boxes: [], roofs: [], lawns: [], trees: [], footprints: [] };
  const depot = traffic.depot;
  const inDepot = (x: number, z: number, margin: number) => {
    if (!depot) return false;
    const [fx, fz] = forward(depot.base.heading), [rx, rz] = right(depot.base.heading);
    const u = (x - depot.base.x) * fx + (z - depot.base.z) * fz, v = (x - depot.base.x) * rx + (z - depot.base.z) * rz;
    return Math.abs(u) < DEPOT.approach + margin && v > -margin && v < DEPOT.far + 1.5 + margin;
  };
  traffic.cells.forEach((cell, cellIndex) => {
    const inset = ROAD_WIDTH / 2 + SIDEWALK;
    const w = cell.w - inset * 2, d = cell.d - inset * 2;
    if (w < 4 || d < 4) return;
    town.lawns.push({ x: cell.x, y: 0.01, z: cell.z, sx: w, sy: 0.02, sz: d, rotation: 0, color: "#cfe8d3" });
    const edges = [
      { x: cell.x, z: cell.z - d / 2, nx: 0, nz: -1, length: w, end: 0.6 },
      { x: cell.x, z: cell.z + d / 2, nx: 0, nz: 1, length: w, end: 0.6 },
      { x: cell.x - w / 2, z: cell.z, nx: -1, nz: 0, length: d, end: SETBACK + LOT_DEPTH + 1 },
      { x: cell.x + w / 2, z: cell.z, nx: 1, nz: 0, length: d, end: SETBACK + LOT_DEPTH + 1 },
    ];
    edges.forEach((edge, edgeIndex) => {
      const usable = edge.length - edge.end * 2;
      const count = Math.floor(usable / LOT_STEP);
      const tx = -edge.nz, tz = edge.nx;
      for (let k = 0; k < count; k++) {
        const along = -usable / 2 + LOT_STEP * (k + 0.5) + (usable - count * LOT_STEP) / 2;
        const x = edge.x + tx * along - edge.nx * (SETBACK + LOT_DEPTH / 2), z = edge.z + tz * along - edge.nz * (SETBACK + LOT_DEPTH / 2);
        if (inDepot(x, z, 2.5)) continue;
        const seed = cellIndex * 1000 + edgeIndex * 100 + k;
        const kind = hash(seed);
        // 正面（+z）朝向街道
        const rotation = Math.atan2(edge.nx, edge.nz);
        if (kind < 0.1) { town.trees.push({ x, z, scale: 0.9 + hash(seed + 3) * 0.4 }); continue; }
        addBuilding(town, kind < 0.6 ? "house" : kind < 0.82 ? "shop" : "office", x, z, rotation, seed);
      }
    });
    // 後院樹木
    const yard = { w: w - (SETBACK + LOT_DEPTH + 0.6) * 2, d: d - (SETBACK + LOT_DEPTH + 0.6) * 2 };
    if (yard.w > 1.5 && yard.d > 1.5) scatterTrees(town, { x: cell.x, z: cell.z, ...yard }, cellIndex, depot ? (x, z) => inDepot(x, z, 1.5) : null);
  });
  parks.forEach((park, index) => {
    town.lawns.push({ x: park.x, y: 0.02, z: park.z, sx: park.w, sy: 0.02, sz: park.d, rotation: 0, color: "#cfe8d3" });
    // 公園比後院疏朗
    scatterTrees(town, { x: park.x, z: park.z, w: park.w - 1.2, d: park.d - 1.2 }, 500 + index, null, 0.68, 3.6);
  });
  return town;
}

function scatterTrees(town: Town, area: Rect, seed: number, skip: ((x: number, z: number) => boolean) | null, sparse = 0.45, step = 2.8): void {
  const cols = Math.max(1, Math.floor(area.w / step)), rows = Math.max(1, Math.floor(area.d / step));
  for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) {
    const h = hash(seed * 31 + i * 7 + j * 13);
    if (h < sparse) continue;
    const x = area.x - area.w / 2 + (i + 0.5) * (area.w / cols) + (h - 0.7) * 0.8;
    const z = area.z - area.d / 2 + (j + 0.5) * (area.d / rows) + (hash(h * 97) - 0.5) * 0.8;
    if (skip && skip(x, z)) continue;
    town.trees.push({ x, z, scale: 0.75 + h * 0.45 });
  }
}

function addBuilding(town: Town, type: "house" | "shop" | "office", x: number, z: number, rotation: number, seed: number): void {
  const cos = Math.cos(rotation), sin = Math.sin(rotation);
  // 建物座標（lx 往右、lz 往正面）轉世界座標
  const part = (lx: number, y: number, lz: number, sx: number, sy: number, sz: number, color: string): Part => ({
    x: x + lx * cos + lz * sin, y, z: z - lx * sin + lz * cos, sx, sy, sz, rotation, color,
  });
  const wall = WALLS[Math.floor(hash(seed + 1) * WALLS.length)];
  if (type === "house") {
    const w = 2.6 + hash(seed + 2) * 0.6, d = 2.3, h = 1.4 + hash(seed + 4) * 0.3;
    town.boxes.push(
      part(0, h / 2, 0, w, h, d, wall),
      part(-w * 0.18, 0.45, d / 2 + 0.02, 0.5, 0.9, 0.05, ROOFS[Math.floor(hash(seed + 5) * ROOFS.length)]),
      part(w * 0.22, h * 0.6, d / 2 + 0.02, 0.6, 0.45, 0.05, GLASS),
      part(w * 0.3, h + 0.55, -d * 0.2, 0.3, 0.7, 0.3, "#9aa5b8"),
    );
    town.roofs.push(part(0, h, 0, w + 0.3, 0.95, d + 0.35, ROOFS[Math.floor(hash(seed + 6) * ROOFS.length)]));
    town.footprints.push({ x, z, r: Math.hypot(w, d) / 2 });
  } else if (type === "shop") {
    const w = 3.8, d = 2.7, h = 1.7, accent = ACCENTS[Math.floor(hash(seed + 7) * ACCENTS.length)];
    town.boxes.push(
      part(0, h / 2, 0, w, h, d, wall),
      part(0, h + 0.08, 0, w + 0.2, 0.16, d + 0.2, DARK),
      part(0, 0.6, d / 2 + 0.02, w * 0.62, 0.8, 0.05, GLASS),
      part(0, 1.25, d / 2 + 0.4, w * 0.9, 0.1, 0.8, accent),
      part(0, h + 0.45, d / 2 - 0.1, w * 0.5, 0.45, 0.12, accent),
    );
    town.footprints.push({ x, z, r: Math.hypot(w, d) / 2 });
  } else {
    const w = 3, d = 2.7, floors = 2 + Math.floor(hash(seed + 8) * 3), h = floors * 1.05 + 0.3;
    town.boxes.push(part(0, h / 2, 0, w, h, d, "#dfe6f1"), part(0, h + 0.07, 0, w + 0.15, 0.14, d + 0.15, "#9aa5b8"));
    for (let f = 0; f < floors; f++) town.boxes.push(part(0, 0.75 + f * 1.05, 0, w + 0.04, 0.4, d + 0.04, GLASS));
    town.footprints.push({ x, z, r: Math.hypot(w, d) / 2 });
  }
}
