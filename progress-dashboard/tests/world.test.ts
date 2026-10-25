import { describe, expect, it } from "vitest";

import {
  ROAD_WIDTH, createTraffic, layoutTown, lightState, packBlocks, signalLamps, stepTraffic, streetParts, syncTrucks, vehiclePose,
  type Traffic, type Vehicle,
} from "@/lib/world";

const SIZES = [
  { w: 24, d: 26 }, { w: 24, d: 18 }, { w: 10, d: 10 }, { w: 17, d: 18 }, { w: 31, d: 20 }, { w: 17, d: 10 }, { w: 10, d: 10 },
];

function city(trucks = ["t1", "t2", "t3", "t4"]): { traffic: Traffic; grid: ReturnType<typeof packBlocks> } {
  const grid = packBlocks(SIZES);
  return { traffic: createTraffic(grid, trucks), grid };
}

/** 同一條車道上任兩台車的車身不重疊。 */
function overlaps(traffic: Traffic): string[] {
  const problems: string[] = [];
  const byLane = new Map<number, Vehicle[]>();
  for (const v of traffic.vehicles) if (!v.connector) byLane.set(v.lane, [...(byLane.get(v.lane) ?? []), v]);
  for (const [lane, list] of byLane) {
    list.sort((a, b) => a.s - b.s);
    for (let k = 1; k < list.length; k++) {
      const gap = list[k].s - list[k - 1].s - (list[k].length + list[k - 1].length) / 2;
      if (gap < -0.05) problems.push(`lane ${lane}: ${list[k - 1].id}/${list[k].id} gap ${gap.toFixed(2)}`);
    }
  }
  return problems;
}

describe("packBlocks", () => {
  it("arranges blocks on a regular street grid without overlaps", () => {
    const grid = packBlocks(SIZES);
    expect(grid.xs.length).toBeGreaterThan(2);
    expect(grid.zs.length).toBeGreaterThan(2);
    SIZES.forEach((size, i) => {
      const c = grid.centers[i];
      // 每個區塊都完整落在某一格街廓裡，不壓到街道
      const col = grid.xs.findIndex((x, k) => k < grid.xs.length - 1 && c.x > x && c.x < grid.xs[k + 1]);
      const row = grid.zs.findIndex((z, k) => k < grid.zs.length - 1 && c.z > z && c.z < grid.zs[k + 1]);
      expect(c.x - size.w / 2).toBeGreaterThanOrEqual(grid.xs[col] + ROAD_WIDTH / 2);
      expect(c.x + size.w / 2).toBeLessThanOrEqual(grid.xs[col + 1] - ROAD_WIDTH / 2);
      expect(c.z - size.d / 2).toBeGreaterThanOrEqual(grid.zs[row] + ROAD_WIDTH / 2);
      expect(c.z + size.d / 2).toBeLessThanOrEqual(grid.zs[row + 1] - ROAD_WIDTH / 2);
      for (let j = 0; j < i; j++) {
        const o = grid.centers[j];
        expect(Math.abs(c.x - o.x) < (size.w + SIZES[j].w) / 2 && Math.abs(c.z - o.z) < (size.d + SIZES[j].d) / 2).toBe(false);
      }
    });
  });
});

describe("city", () => {
  it("builds a connected two-way street network with signals and a depot block", () => {
    const { traffic } = city();
    expect(traffic.depot).not.toBeNull();
    expect(traffic.nodes.filter((n) => n.signal).length).toBeGreaterThan(8);
    for (const node of traffic.nodes) {
      expect(node.incoming.length).toBeGreaterThan(0);
      expect(node.outgoing.length).toBeGreaterThan(0);
    }
    // 所有節點都能開到倉庫
    expect(traffic.depotDistance.every((d) => Number.isFinite(d))).toBe(true);
    expect(traffic.cells.some((cell) => cell.depot)).toBe(true);
    const parts = streetParts(traffic);
    expect(parts.asphalt.length).toBe(traffic.segments.length);
    expect(signalLamps(traffic).length).toBe(parts.poles.length / 2);
  });

  it("gives exactly one approach a green light at a time", () => {
    const { traffic } = city();
    const node = traffic.nodes.find((n) => n.signal && n.incoming.length === 4)!;
    const seenGreen = new Set<number>();
    for (let t = 0; t < 60; t += 0.25) {
      traffic.time = t;
      const states = node.incoming.map((lane) => lightState(traffic, node.id, lane));
      expect(states.filter((state) => state !== "red").length).toBeLessThanOrEqual(1);
      node.incoming.forEach((lane, k) => { if (states[k] === "green") seenGreen.add(lane); });
    }
    expect(seenGreen.size).toBe(4);
  });

  it("keeps vehicles apart while trucks drive to the depot, unload and leave", () => {
    const { traffic } = city();
    expect(traffic.vehicles.filter((v) => v.kind !== "truck").length).toBeGreaterThan(10);
    let unloaded = 0, turned = 0;
    const lastLane = new Map(traffic.vehicles.map((v) => [v.id, v.lane]));
    for (let step = 0; step < 9000; step++) {
      stepTraffic(traffic, 1 / 30);
      if (step % 5 === 0) expect(overlaps(traffic)).toEqual([]);
      for (const v of traffic.vehicles) {
        if (v.unload > 5 - 1 / 30 - 1e-9) unloaded++;
        if (!v.connector && v.lane !== lastLane.get(v.id)) {
          const before = traffic.lanes[lastLane.get(v.id)!].heading, after = traffic.lanes[v.lane].heading;
          if (Math.abs(Math.sin(after - before)) > 0.9) turned++;
          lastLane.set(v.id, v.lane);
        }
      }
      if (step % 100 === 0) for (const v of traffic.vehicles) expect(Number.isFinite(vehiclePose(traffic, v).x)).toBe(true);
    }
    expect(unloaded).toBeGreaterThanOrEqual(3);
    expect(turned).toBeGreaterThan(20);
    // 車流沒有卡死：大多數車在動
    const moving = traffic.vehicles.filter((v) => v.speed > 0.1).length;
    expect(moving).toBeGreaterThan(traffic.vehicles.length * 0.3);
  }, 60_000);

  it("queues followers behind a held vehicle", () => {
    const { traffic } = city(["t1"]);
    const truck = traffic.vehicles.find((v) => v.kind === "truck")!;
    truck.cooldown = 1e9;
    let behind: Vehicle | undefined;
    for (let step = 0; step < 6000 && !behind; step++) {
      stepTraffic(traffic, 1 / 30);
      truck.held = !truck.connector;
      if (truck.held && truck.speed === 0) {
        behind = traffic.vehicles.find((v) => v !== truck && !v.connector && v.lane === truck.lane && v.s < truck.s && truck.s - v.s < 6 && v.speed < 0.05);
      }
    }
    expect(behind).toBeDefined();
  }, 30_000);

  it("syncs trucks without moving the existing ones", () => {
    const { traffic } = city(["a", "b"]);
    const a = traffic.vehicles.find((v) => v.id === "a")!;
    const where = [a.lane, a.s];
    syncTrucks(traffic, ["a", "c"]);
    expect([a.lane, a.s]).toEqual(where);
    expect(traffic.vehicles.some((v) => v.id === "b")).toBe(false);
    expect(traffic.vehicles.some((v) => v.id === "c")).toBe(true);
  });
});

describe("town", () => {
  it("lines the outer blocks with buildings that stay off the streets and out of the depot", () => {
    const { traffic, grid } = city();
    const town = layoutTown(traffic, grid.parks);
    expect(town.footprints.length).toBeGreaterThan(30);
    const depot = traffic.depot!;
    for (const f of town.footprints) {
      for (const s of traffic.segments) {
        // 到街道中心線的距離大於半個路寬
        const dx = s.x2 - s.x1, dz = s.z2 - s.z1, len2 = dx * dx + dz * dz;
        const t = Math.max(0, Math.min(1, ((f.x - s.x1) * dx + (f.z - s.z1) * dz) / len2));
        const distance = Math.hypot(f.x - (s.x1 + dx * t), f.z - (s.z1 + dz * t));
        expect(distance).toBeGreaterThan(ROAD_WIDTH / 2 + 0.4);
      }
      expect(Math.hypot(f.x - depot.x, f.z - depot.z)).toBeGreaterThan(depot.length / 2);
    }
  });
});
