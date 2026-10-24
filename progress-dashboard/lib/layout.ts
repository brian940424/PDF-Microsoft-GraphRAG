import type { PlanItem, TestModule } from "./types";

export interface Rect {
  x: number;
  z: number;
  w: number;
  d: number;
}

export interface Lot {
  item: PlanItem;
  x: number;
  z: number;
}

export interface Zone extends Rect {
  name: string;
  lots: Lot[];
}

export interface ContainerSlot {
  module: TestModule;
  x: number;
  z: number;
}

export interface Yard extends Rect {
  slots: ContainerSlot[];
}

export interface SiteLayout {
  zones: Zone[];
  yard: Yard | null;
  /** 環繞工地的道路中心線矩形 */
  road: Rect;
  trees: { x: number; z: number; scale: number }[];
  /** 涵蓋整個場景的半徑，用於鏡頭縮放與陰影範圍 */
  radius: number;
}

export const LOT_SIZE = 7;
const ZONE_PADDING = 1.5;
const ZONE_LABEL_SPACE = 1.5;
const GAP = 3.5;
const CONTAINER_PITCH_X = 6.2;
const CONTAINER_PITCH_Z = 3;
const YARD_COLUMNS = 4;
const ROAD_OFFSET = 5;

/** 工地版面：每個類別一個分區（地塊網格），測試模組放在貨櫃場，整體以貨架式排列後置中。 */
export function layoutSite(plan: PlanItem[], modules: TestModule[]): SiteLayout {
  const groups = new Map<string, PlanItem[]>();
  for (const item of plan) groups.set(item.district, [...(groups.get(item.district) ?? []), item]);

  const blocks: { kind: "zone" | "yard"; name: string; w: number; d: number; cols: number; count: number }[] = [];
  for (const [name, items] of groups) {
    const cols = Math.ceil(Math.sqrt(items.length));
    const rows = Math.ceil(items.length / cols);
    blocks.push({ kind: "zone", name, cols, count: items.length,
      w: cols * LOT_SIZE + ZONE_PADDING * 2, d: rows * LOT_SIZE + ZONE_PADDING * 2 + ZONE_LABEL_SPACE });
  }
  if (modules.length) {
    const cols = Math.min(YARD_COLUMNS, modules.length);
    const rows = Math.ceil(modules.length / cols);
    blocks.push({ kind: "yard", name: "測試貨櫃場", cols, count: modules.length,
      w: cols * CONTAINER_PITCH_X + ZONE_PADDING * 2, d: rows * CONTAINER_PITCH_Z + ZONE_PADDING * 2 + ZONE_LABEL_SPACE });
  }

  // 貨架式排列：一排放滿再換下一排
  const area = blocks.reduce((sum, block) => sum + (block.w + GAP) * (block.d + GAP), 0);
  const shelfWidth = Math.max(Math.sqrt(area) * 1.35, ...blocks.map((block) => block.w));
  const placed: (Rect & { block: (typeof blocks)[number] })[] = [];
  let cursorX = 0, cursorZ = 0, shelfDepth = 0;
  for (const block of blocks) {
    if (cursorX > 0 && cursorX + block.w > shelfWidth) {
      cursorX = 0;
      cursorZ += shelfDepth + GAP;
      shelfDepth = 0;
    }
    placed.push({ block, x: cursorX + block.w / 2, z: cursorZ + block.d / 2, w: block.w, d: block.d });
    cursorX += block.w + GAP;
    shelfDepth = Math.max(shelfDepth, block.d);
  }
  const maxX = Math.max(0, ...placed.map((rect) => rect.x + rect.w / 2));
  const maxZ = Math.max(0, ...placed.map((rect) => rect.z + rect.d / 2));
  const shiftX = maxX / 2, shiftZ = maxZ / 2;

  const zones: Zone[] = [];
  let yard: Yard | null = null;
  for (const rect of placed) {
    const x = rect.x - shiftX, z = rect.z - shiftZ;
    const left = x - rect.w / 2 + ZONE_PADDING, top = z - rect.d / 2 + ZONE_PADDING;
    if (rect.block.kind === "zone") {
      const lots = groups.get(rect.block.name)!.map((item, index) => ({
        item,
        x: left + (index % rect.block.cols) * LOT_SIZE + LOT_SIZE / 2,
        z: top + Math.floor(index / rect.block.cols) * LOT_SIZE + LOT_SIZE / 2,
      }));
      zones.push({ name: rect.block.name, x, z, w: rect.w, d: rect.d, lots });
    } else {
      const slots = modules.map((module, index) => ({
        module,
        x: left + (index % rect.block.cols) * CONTAINER_PITCH_X + CONTAINER_PITCH_X / 2,
        z: top + Math.floor(index / rect.block.cols) * CONTAINER_PITCH_Z + CONTAINER_PITCH_Z / 2,
      }));
      yard = { x, z, w: rect.w, d: rect.d, slots };
    }
  }

  const road = { x: 0, z: 0, w: maxX + ROAD_OFFSET * 2, d: maxZ + ROAD_OFFSET * 2 };
  return { zones, yard, road, trees: layoutTrees(road), radius: Math.hypot(road.w, road.d) / 2 + 6 };
}

/** 道路外側的行道樹，以座標雜湊決定疏密，結果固定不隨重整變動。 */
function layoutTrees(road: Rect): SiteLayout["trees"] {
  const trees: SiteLayout["trees"] = [];
  const offset = 5.5, step = 4.5;
  const halfW = road.w / 2 + offset, halfD = road.d / 2 + offset;
  const add = (x: number, z: number) => {
    const hash = Math.abs(Math.sin(x * 12.9898 + z * 78.233) * 43758.5453) % 1;
    if (hash > 0.38) trees.push({ x, z, scale: 0.8 + hash * 0.5 });
  };
  for (let x = -halfW; x <= halfW; x += step) { add(x, -halfD); add(x, halfD); }
  for (let z = -halfD + step; z < halfD; z += step) { add(-halfW, z); add(halfW, z); }
  return trees;
}

/** 沿道路矩形周長的參數 t∈[0,1) 取得位置與行進方向。 */
export function roadPoint(road: Rect, t: number): { x: number; z: number; heading: number } {
  const hw = road.w / 2, hd = road.d / 2;
  const perimeter = 2 * (road.w + road.d);
  let s = (((t % 1) + 1) % 1) * perimeter;
  if (s < road.w) return { x: -hw + s, z: hd, heading: 0 };
  s -= road.w;
  if (s < road.d) return { x: hw, z: hd - s, heading: Math.PI / 2 };
  s -= road.d;
  if (s < road.w) return { x: hw - s, z: -hd, heading: Math.PI };
  s -= road.w;
  return { x: -hw, z: -hd + s, heading: -Math.PI / 2 };
}

export const STATUS_ORDER = ["進行中", "待確認", "待辦", "完成", "不處理"];

export function statusRank(status: string): number {
  const index = STATUS_ORDER.indexOf(status);
  return index < 0 ? STATUS_ORDER.length : index;
}

export function priorityRank(priority: string): number {
  return priority === "高" ? 0 : priority === "中" ? 1 : 2;
}
