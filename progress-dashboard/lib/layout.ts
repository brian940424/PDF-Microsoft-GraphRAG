import type { PlanItem, TestModule } from "./types";
import { packBlocks } from "./world";

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
  /** 分區街廓的外框（最外圈街道中心線） */
  road: Rect;
  /** 街道中心線，供路網與城鎮使用 */
  grid: { xs: number[]; zs: number[] };
  /** 街廓內的空地（公園） */
  parks: Rect[];
  /** 涵蓋整個場景的半徑，用於鏡頭縮放與陰影範圍 */
  radius: number;
}

export const LOT_SIZE = 7;
const ZONE_PADDING = 1.5;
const ZONE_LABEL_SPACE = 1.5;
const CONTAINER_PITCH_X = 6.2;
const CONTAINER_PITCH_Z = 3;
const YARD_COLUMNS = 4;

/** 工地版面：每個類別一個分區（地塊網格），測試模組放在貨櫃場，以格狀重劃排進街廓（四周都是街道）。 */
export function layoutSite(plan: PlanItem[], modules: TestModule[]): SiteLayout {
  const groups = new Map<string, PlanItem[]>();
  for (const item of plan) groups.set(item.district, [...(groups.get(item.district) ?? []), item]);

  const blocks: { kind: "zone" | "yard"; name: string; w: number; d: number; cols: number }[] = [];
  for (const [name, items] of groups) {
    const cols = Math.ceil(Math.sqrt(items.length));
    const rows = Math.ceil(items.length / cols);
    blocks.push({ kind: "zone", name, cols, w: cols * LOT_SIZE + ZONE_PADDING * 2, d: rows * LOT_SIZE + ZONE_PADDING * 2 + ZONE_LABEL_SPACE });
  }
  if (modules.length) {
    const cols = Math.min(YARD_COLUMNS, modules.length);
    const rows = Math.ceil(modules.length / cols);
    blocks.push({ kind: "yard", name: "測試貨櫃場", cols,
      w: cols * CONTAINER_PITCH_X + ZONE_PADDING * 2, d: rows * CONTAINER_PITCH_Z + ZONE_PADDING * 2 + ZONE_LABEL_SPACE });
  }

  const packed = packBlocks(blocks);
  const zones: Zone[] = [];
  let yard: Yard | null = null;
  blocks.forEach((block, index) => {
    const { x, z } = packed.centers[index];
    const left = x - block.w / 2 + ZONE_PADDING, top = z - block.d / 2 + ZONE_PADDING;
    if (block.kind === "zone") {
      const lots = groups.get(block.name)!.map((item, k) => ({
        item,
        x: left + (k % block.cols) * LOT_SIZE + LOT_SIZE / 2,
        z: top + Math.floor(k / block.cols) * LOT_SIZE + LOT_SIZE / 2,
      }));
      zones.push({ name: block.name, x, z, w: block.w, d: block.d, lots });
    } else {
      const slots = modules.map((module, k) => ({
        module,
        x: left + (k % block.cols) * CONTAINER_PITCH_X + CONTAINER_PITCH_X / 2,
        z: top + Math.floor(k / block.cols) * CONTAINER_PITCH_Z + CONTAINER_PITCH_Z / 2,
      }));
      yard = { x, z, w: block.w, d: block.d, slots };
    }
  });
  const road = packed.bounds;
  return { zones, yard, road, grid: { xs: packed.xs, zs: packed.zs }, parks: packed.parks, radius: Math.hypot(road.w, road.d) / 2 + 6 };
}

export const STATUS_ORDER = ["進行中", "待確認", "待辦", "完成", "不處理"];

export function statusRank(status: string): number {
  const index = STATUS_ORDER.indexOf(status);
  return index < 0 ? STATUS_ORDER.length : index;
}

export function priorityRank(priority: string): number {
  return priority === "高" ? 0 : priority === "中" ? 1 : 2;
}
