"""由 progress-dashboard/lib/world.ts 自動產生（cd progress-dashboard && npm run export-world），請勿手動修改。"""

WORLD_JS = r"""
const ROAD_WIDTH = 3.6;
const LANE_OFFSET = 0.9;
const SIDEWALK = 1.0;
const STREET_GAP = 6.4;
const TOWN_DEPTH = 15;
const TRIM_SIGNAL = 2.6, TRIM_CORNER = 2.2, TRIM_MINOR = 0.6;
const SIGNAL = {
    green: 4.5,
    amber: 1,
    allRed: 2.5
};
const forward = (heading)=>[
        Math.cos(heading),
        -Math.sin(heading)
    ];
const right = (heading)=>[
        Math.sin(heading),
        Math.cos(heading)
    ];
const headingOf = (dx, dz)=>Math.atan2(-dz, dx);
const angleDiff = (a, b)=>Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
const hash = (n)=>Math.abs(Math.sin(n * 12.9898 + 78.233) * 43758.5453) % 1;
function packBlocks(sizes) {
    if (!sizes.length) return {
        centers: [],
        xs: [
            -12,
            12
        ],
        zs: [
            -12,
            12
        ],
        parks: [],
        bounds: {
            x: 0,
            z: 0,
            w: 24,
            d: 24
        }
    };
    const order = sizes.map((_, i)=>i).sort((a, b)=>sizes[b].w * sizes[b].d - sizes[a].w * sizes[a].d);
    const used = sizes.reduce((sum, size)=>sum + size.w * size.d, 0);
    let best = null;
    for(let cols = 1; cols <= sizes.length; cols++){
        const rows = Math.ceil(sizes.length / cols);
        const colW = new Array(cols).fill(0), rowD = new Array(rows).fill(0);
        order.forEach((index, k)=>{
            colW[k % cols] = Math.max(colW[k % cols], sizes[index].w);
            rowD[Math.floor(k / cols)] = Math.max(rowD[Math.floor(k / cols)], sizes[index].d);
        });
        const width = sum(colW), depth = sum(rowD);
        const score = Math.abs(Math.log((width + STREET_GAP * cols) / (depth + STREET_GAP * rows) / 1.3)) + (1 - used / (width * depth));
        if (!best || score < best.score) best = {
            cols,
            colW,
            rowD,
            score
        };
    }
    const { cols, colW, rowD } = best;
    const xs = [
        0
    ], zs = [
        0
    ];
    colW.forEach((w)=>xs.push(xs[xs.length - 1] + w + STREET_GAP));
    rowD.forEach((d)=>zs.push(zs[zs.length - 1] + d + STREET_GAP));
    const shiftX = xs[xs.length - 1] / 2, shiftZ = zs[zs.length - 1] / 2;
    for(let i = 0; i < xs.length; i++)xs[i] -= shiftX;
    for(let i = 0; i < zs.length; i++)zs[i] -= shiftZ;
    const centers = new Array(sizes.length);
    const parks = [];
    const cellCount = colW.length * rowD.length;
    for(let k = 0; k < cellCount; k++){
        const c = k % cols, r = Math.floor(k / cols);
        const left = xs[c] + STREET_GAP / 2, top = zs[r] + STREET_GAP / 2;
        const index = order[k];
        if (index === undefined) {
            parks.push({
                x: left + colW[c] / 2,
                z: top + rowD[r] / 2,
                w: colW[c],
                d: rowD[r]
            });
            continue;
        }
        const { w, d } = sizes[index];
        centers[index] = {
            x: left + w / 2,
            z: top + d / 2
        };
        if (colW[c] - w >= 2.5) parks.push({
            x: left + w + (colW[c] - w) / 2,
            z: top + rowD[r] / 2,
            w: colW[c] - w,
            d: rowD[r]
        });
        if (rowD[r] - d >= 2.5) parks.push({
            x: left + w / 2,
            z: top + d + (rowD[r] - d) / 2,
            w,
            d: rowD[r] - d
        });
    }
    return {
        centers,
        xs,
        zs,
        parks,
        bounds: {
            x: 0,
            z: 0,
            w: xs[xs.length - 1] - xs[0],
            d: zs[zs.length - 1] - zs[0]
        }
    };
}
function sum(values) {
    return values.reduce((total, value)=>total + value, 0);
}
const DEPOT = {
    halfLength: 5,
    near: 3.8,
    far: 10.3,
    aisles: [
        5.4,
        8.7
    ],
    approach: 17
};
const DEPOT_SPAN = DEPOT.approach + 3.5;
const TRUCK = {
    length: 4.4,
    max: 3.2
};
const DEPOT_SPEED = 2.4;
const UNLOAD_SECONDS = 5;
const SAFE_GAP = 1.3;
function createTraffic(grid, truckIds) {
    const xs = [
        grid.xs[0] - TOWN_DEPTH,
        ...grid.xs,
        grid.xs[grid.xs.length - 1] + TOWN_DEPTH
    ];
    const zs = [
        grid.zs[0] - TOWN_DEPTH,
        ...grid.zs,
        grid.zs[grid.zs.length - 1] + TOWN_DEPTH
    ];
    const plan = planDepot(xs, zs);
    let segments = [];
    for (const z of zs)for(let i = 0; i < xs.length - 1; i++)segments.push({
        x1: xs[i],
        z1: z,
        x2: xs[i + 1],
        z2: z
    });
    for (const x of xs)for(let i = 0; i < zs.length - 1; i++)segments.push({
        x1: x,
        z1: zs[i],
        x2: x,
        z2: zs[i + 1]
    });
    if (plan) segments = segments.filter((segment)=>!plan.removes(segment));
    const traffic = {
        road: {
            x: 0,
            z: 0,
            w: grid.xs[grid.xs.length - 1] - grid.xs[0],
            d: grid.zs[grid.zs.length - 1] - grid.zs[0]
        },
        nodes: [],
        lanes: [],
        segments,
        cells: townCells(xs, zs, segments, plan),
        depot: null,
        vehicles: [],
        doors: [],
        occupants: [],
        time: 0,
        connectors: new Map(),
        depotDistance: [],
        seed: 20261025
    };
    const graphSegments = plan ? splitAt(splitAt(segments, plan.entry), plan.exit) : segments;
    buildGraph(traffic, graphSegments, plan);
    if (plan) buildDepot(traffic, plan);
    placeVehicles(traffic, truckIds);
    return traffic;
}
function planDepot(xs, zs) {
    const last = (values)=>values.length - 1;
    const sides = [
        {
            horizontal: true,
            line: zs[1],
            heading: Math.PI,
            lo: xs[0],
            hi: xs[last(xs)],
            band: [
                zs[0],
                zs[1]
            ],
            nodes: xs
        },
        {
            horizontal: false,
            line: xs[last(xs) - 1],
            heading: Math.PI / 2,
            lo: zs[0],
            hi: zs[last(zs)],
            band: [
                xs[last(xs) - 1],
                xs[last(xs)]
            ],
            nodes: zs
        },
        {
            horizontal: false,
            line: xs[1],
            heading: Math.PI * 1.5,
            lo: zs[0],
            hi: zs[last(zs)],
            band: [
                xs[0],
                xs[1]
            ],
            nodes: zs
        },
        {
            horizontal: true,
            line: zs[last(zs) - 1],
            heading: 0,
            lo: xs[0],
            hi: xs[last(xs)],
            band: [
                zs[last(zs) - 1],
                zs[last(zs)]
            ],
            nodes: xs
        }
    ];
    for (const side of sides){
        const middle = (side.lo + side.hi) / 2;
        const candidates = [];
        for(let c = side.lo + DEPOT_SPAN + 3; c <= side.hi - DEPOT_SPAN - 3; c += 0.5)candidates.push(c);
        candidates.sort((a, b)=>Math.abs(a - middle) - Math.abs(b - middle));
        const center = candidates.find((c)=>side.nodes.every((n)=>Math.abs(n - (c - DEPOT.approach)) >= 3.5 && Math.abs(n - (c + DEPOT.approach)) >= 3.5));
        if (center === undefined) continue;
        const base = side.horizontal ? {
            x: center,
            z: side.line,
            heading: side.heading
        } : {
            x: side.line,
            z: center,
            heading: side.heading
        };
        const at = (u, v)=>localPoint(base, u, v);
        const [bandLo, bandHi] = side.band;
        return {
            base,
            entry: at(-DEPOT.approach, 0),
            exit: at(DEPOT.approach, 0),
            removes: (segment)=>{
                const perpendicular = side.horizontal ? segment.x1 === segment.x2 : segment.z1 === segment.z2;
                if (!perpendicular) return false;
                const along = side.horizontal ? segment.x1 : segment.z1;
                const across = side.horizontal ? [
                    segment.z1,
                    segment.z2
                ] : [
                    segment.x1,
                    segment.x2
                ];
                return Math.abs(along - center) < DEPOT_SPAN && Math.min(...across) >= bandLo - 1e-6 && Math.max(...across) <= bandHi + 1e-6;
            },
            contains: (x, z, margin)=>{
                const [fx, fz] = forward(base.heading), [rx, rz] = right(base.heading);
                const u = (x - base.x) * fx + (z - base.z) * fz, v = (x - base.x) * rx + (z - base.z) * rz;
                return Math.abs(u) < DEPOT.approach + margin && v > -margin && v < DEPOT.far + 1.5 + margin;
            }
        };
    }
    return null;
}
function localPoint(base, u, v) {
    const [fx, fz] = forward(base.heading), [rx, rz] = right(base.heading);
    return {
        x: base.x + fx * u + rx * v,
        z: base.z + fz * u + rz * v
    };
}
function splitAt(segments, point) {
    const result = [];
    for (const segment of segments){
        const onX = Math.abs(segment.z1 - segment.z2) < 1e-6 && Math.abs(point.z - segment.z1) < 1e-6 && point.x > Math.min(segment.x1, segment.x2) && point.x < Math.max(segment.x1, segment.x2);
        const onZ = Math.abs(segment.x1 - segment.x2) < 1e-6 && Math.abs(point.x - segment.x1) < 1e-6 && point.z > Math.min(segment.z1, segment.z2) && point.z < Math.max(segment.z1, segment.z2);
        if (onX || onZ) result.push({
            ...segment,
            x2: point.x,
            z2: point.z
        }, {
            x1: point.x,
            z1: point.z,
            x2: segment.x2,
            z2: segment.z2
        });
        else result.push(segment);
    }
    return result;
}
function townCells(xs, zs, segments, plan) {
    const has = (x1, z1, x2, z2)=>segments.some((s)=>Math.abs(s.x1 - x1) < 1e-6 && Math.abs(s.z1 - z1) < 1e-6 && Math.abs(s.x2 - x2) < 1e-6 && Math.abs(s.z2 - z2) < 1e-6);
    const cells = [];
    const add = (x1, x2, z1, z2)=>{
        const cell = {
            x: (x1 + x2) / 2,
            z: (z1 + z2) / 2,
            w: x2 - x1,
            d: z2 - z1
        };
        cells.push({
            ...cell,
            depot: plan ? overlaps(cell, plan) : false
        });
    };
    const last = xs.length - 1, lastZ = zs.length - 1;
    for (const [z1, z2] of [
        [
            zs[0],
            zs[1]
        ],
        [
            zs[lastZ - 1],
            zs[lastZ]
        ]
    ]){
        const cuts = xs.filter((x, i)=>i === 0 || i === last || has(x, z1, x, z2));
        for(let i = 0; i < cuts.length - 1; i++)add(cuts[i], cuts[i + 1], z1, z2);
    }
    for (const [x1, x2] of [
        [
            xs[0],
            xs[1]
        ],
        [
            xs[last - 1],
            xs[last]
        ]
    ]){
        const cuts = zs.slice(1, lastZ).filter((z, i, list)=>i === 0 || i === list.length - 1 || has(x1, z, x2, z));
        for(let i = 0; i < cuts.length - 1; i++)add(x1, x2, cuts[i], cuts[i + 1]);
    }
    return cells;
}
function overlaps(cell, plan) {
    for(let fx = -0.45; fx <= 0.45; fx += 0.15)for(let fz = -0.45; fz <= 0.45; fz += 0.15){
        if (plan.contains(cell.x + cell.w * fx, cell.z + cell.d * fz, 0)) return true;
    }
    return false;
}
function buildGraph(traffic, segments, plan) {
    const keyOf = (x, z)=>`${x.toFixed(3)},${z.toFixed(3)}`;
    const index = new Map();
    const nodeAt = (x, z)=>{
        const key = keyOf(x, z);
        if (!index.has(key)) {
            const kind = plan && Math.hypot(x - plan.entry.x, z - plan.entry.z) < 1e-6 ? "depotEntry" : plan && Math.hypot(x - plan.exit.x, z - plan.exit.z) < 1e-6 ? "depotExit" : "grid";
            index.set(key, traffic.nodes.length);
            traffic.nodes.push({
                id: traffic.nodes.length,
                x,
                z,
                kind,
                signal: false,
                incoming: [],
                outgoing: [],
                offset: 0
            });
        }
        return index.get(key);
    };
    const pairs = segments.map((segment)=>[
            nodeAt(segment.x1, segment.z1),
            nodeAt(segment.x2, segment.z2)
        ]);
    const degree = new Map();
    for (const [a, b] of pairs){
        degree.set(a, (degree.get(a) ?? 0) + 1);
        degree.set(b, (degree.get(b) ?? 0) + 1);
    }
    for (const node of traffic.nodes){
        node.signal = node.kind === "grid" && (degree.get(node.id) ?? 0) >= 3;
        node.offset = hash(node.x * 0.37 + node.z * 1.91) * 40;
    }
    const trim = (id)=>{
        const node = traffic.nodes[id];
        return node.kind !== "grid" ? TRIM_MINOR : node.signal ? TRIM_SIGNAL : TRIM_CORNER;
    };
    for (const [a, b] of pairs){
        for (const [from, to] of [
            [
                a,
                b
            ],
            [
                b,
                a
            ]
        ]){
            const p = traffic.nodes[from], q = traffic.nodes[to];
            const length = Math.hypot(q.x - p.x, q.z - p.z);
            const dx = (q.x - p.x) / length, dz = (q.z - p.z) / length;
            const heading = headingOf(dx, dz);
            const [rx, rz] = right(heading);
            const start = {
                x: p.x + dx * trim(from) + rx * LANE_OFFSET,
                z: p.z + dz * trim(from) + rz * LANE_OFFSET
            };
            const end = {
                x: q.x - dx * trim(to) + rx * LANE_OFFSET,
                z: q.z - dz * trim(to) + rz * LANE_OFFSET
            };
            addLane(traffic, from, to, [
                start,
                end
            ], heading, -1);
        }
    }
}
function addLane(traffic, from, to, path, heading, bay) {
    const cumulative = cumulate(path);
    const lane = {
        id: traffic.lanes.length,
        from,
        to,
        path,
        cumulative,
        length: cumulative[cumulative.length - 1],
        heading,
        bay
    };
    traffic.lanes.push(lane);
    traffic.nodes[from].outgoing.push(lane.id);
    traffic.nodes[to].incoming.push(lane.id);
    return lane;
}
function cumulate(path) {
    const cumulative = [
        0
    ];
    for(let k = 1; k < path.length; k++)cumulative.push(cumulative[k - 1] + Math.hypot(path[k].x - path[k - 1].x, path[k].z - path[k - 1].z));
    return cumulative;
}
function buildDepot(traffic, plan) {
    const { base } = plan;
    const entry = traffic.nodes.findIndex((node)=>node.kind === "depotEntry");
    const exit = traffic.nodes.findIndex((node)=>node.kind === "depotExit");
    const along = (laneId)=>angleDiff(traffic.lanes[laneId].heading, base.heading) < 0.1;
    const bays = DEPOT.aisles.map((v, bayIndex)=>{
        const control = [
            [
                -DEPOT.approach + TRIM_MINOR,
                LANE_OFFSET
            ],
            [
                -DEPOT.approach + 4,
                LANE_OFFSET
            ],
            [
                -DEPOT.halfLength - 3,
                v
            ],
            [
                -DEPOT.halfLength,
                v
            ],
            [
                0,
                v
            ],
            [
                DEPOT.halfLength,
                v
            ],
            [
                DEPOT.halfLength + 3,
                v
            ],
            [
                DEPOT.approach - 4,
                LANE_OFFSET
            ],
            [
                DEPOT.approach - TRIM_MINOR,
                LANE_OFFSET
            ]
        ];
        const smooth = chaikin(control, 3);
        const path = smooth.map(([u, w])=>localPoint(base, u, w));
        const lane = addLane(traffic, entry, exit, path, base.heading, bayIndex);
        const distanceAt = (u)=>{
            let best = 0;
            smooth.forEach(([pu, pv], k)=>{
                if (Math.hypot(pu - u, pv - v) < Math.hypot(smooth[best][0] - u, smooth[best][1] - v)) best = k;
            });
            return lane.cumulative[best];
        };
        const offLane = smooth.map(([, w])=>w >= ROAD_WIDTH / 2 + 0.8);
        return {
            lane: lane.id,
            v,
            path,
            length: lane.length,
            stop: distanceAt(0),
            doors: [
                distanceAt(-DEPOT.halfLength),
                distanceAt(DEPOT.halfLength)
            ],
            leave: lane.cumulative[offLane.indexOf(true)],
            merge: lane.cumulative[offLane.lastIndexOf(true)]
        };
    });
    const center = localPoint(base, 0, (DEPOT.near + DEPOT.far) / 2);
    traffic.depot = {
        x: center.x,
        z: center.z,
        rotation: base.heading,
        length: DEPOT.halfLength * 2,
        depth: DEPOT.far - DEPOT.near,
        bays,
        base,
        entry,
        exit,
        entryLane: traffic.nodes[entry].incoming.find((id)=>along(id) && traffic.lanes[id].bay < 0),
        afterEntryLane: traffic.nodes[entry].outgoing.find((id)=>along(id) && traffic.lanes[id].bay < 0),
        beforeExitLane: traffic.nodes[exit].incoming.find((id)=>along(id) && traffic.lanes[id].bay < 0),
        afterExitLane: traffic.nodes[exit].outgoing.find((id)=>along(id) && traffic.lanes[id].bay < 0)
    };
    traffic.doors = bays.map(()=>[
            0,
            0
        ]);
    traffic.occupants = bays.map(()=>null);
    const distance = new Array(traffic.nodes.length).fill(Infinity);
    const entryLane = traffic.lanes[traffic.depot.entryLane];
    distance[entryLane.from] = entryLane.length;
    const queue = [
        entryLane.from
    ];
    while(queue.length){
        queue.sort((a, b)=>distance[a] - distance[b]);
        const node = queue.shift();
        for (const laneId of traffic.nodes[node].incoming){
            const lane = traffic.lanes[laneId];
            if (lane.bay >= 0) continue;
            const candidate = distance[node] + lane.length;
            if (candidate < distance[lane.from]) {
                distance[lane.from] = candidate;
                queue.push(lane.from);
            }
        }
    }
    traffic.depotDistance = distance;
}
function chaikin(points, iterations) {
    let current = points;
    for(let i = 0; i < iterations; i++){
        const next = [
            current[0]
        ];
        for(let k = 0; k < current.length - 1; k++){
            const [ax, az] = current[k], [bx, bz] = current[k + 1];
            next.push([
                ax * 0.75 + bx * 0.25,
                az * 0.75 + bz * 0.25
            ], [
                ax * 0.25 + bx * 0.75,
                az * 0.25 + bz * 0.75
            ]);
        }
        next.push(current[current.length - 1]);
        current = next;
    }
    return current;
}
function connector(traffic, from, to) {
    const key = `${from}>${to}`;
    const cached = traffic.connectors.get(key);
    if (cached) return cached;
    const a = traffic.lanes[from], b = traffic.lanes[to];
    const p0 = a.path[a.path.length - 1], p1 = a.path[a.path.length - 2], q0 = b.path[0], q1 = b.path[1];
    const d1 = norm(p0.x - p1.x, p0.z - p1.z), d2 = norm(q1.x - q0.x, q1.z - q0.z);
    const cross = d1[0] * d2[1] - d1[1] * d2[0];
    let control = {
        x: (p0.x + q0.x) / 2,
        z: (p0.z + q0.z) / 2
    };
    if (Math.abs(cross) > 1e-3) {
        const t = ((q0.x - p0.x) * d2[1] - (q0.z - p0.z) * d2[0]) / cross;
        control = {
            x: p0.x + d1[0] * t,
            z: p0.z + d1[1] * t
        };
    }
    const path = [];
    for(let k = 0; k <= 10; k++){
        const t = k / 10, u = 1 - t;
        path.push({
            x: u * u * p0.x + 2 * u * t * control.x + t * t * q0.x,
            z: u * u * p0.z + 2 * u * t * control.z + t * t * q0.z
        });
    }
    const cumulative = cumulate(path);
    const result = {
        key,
        from,
        to,
        path,
        cumulative,
        length: Math.max(0.01, cumulative[cumulative.length - 1])
    };
    traffic.connectors.set(key, result);
    return result;
}
function norm(x, z) {
    const length = Math.hypot(x, z) || 1;
    return [
        x / length,
        z / length
    ];
}
function lightState(traffic, nodeId, laneId) {
    const node = traffic.nodes[nodeId];
    if (!node.signal) return "green";
    const approaches = [
        ...node.incoming
    ].sort((a, b)=>traffic.lanes[a].heading - traffic.lanes[b].heading);
    const phase = SIGNAL.green + SIGNAL.amber + SIGNAL.allRed;
    const local = (traffic.time + node.offset) % (phase * approaches.length);
    const active = Math.floor(local / phase), within = local - active * phase;
    if (approaches[active] !== laneId) return "red";
    return within < SIGNAL.green ? "green" : within < SIGNAL.green + SIGNAL.amber ? "amber" : "red";
}
function random(traffic) {
    traffic.seed = (traffic.seed * 1664525 + 1013904223) % 4294967296;
    return traffic.seed / 4294967296;
}
function placeVehicles(traffic, truckIds) {
    const total = sum(traffic.lanes.filter((lane)=>lane.bay < 0).map((lane)=>lane.length));
    const cars = Math.max(8, Math.min(44, Math.round(total / 30)));
    truckIds.forEach((id, index)=>addVehicle(traffic, makeTruck(id, index)));
    for(let k = 0; k < cars; k++)addVehicle(traffic, makeCar(`car-${k}`, k));
}
function makeTruck(id, index) {
    return {
        id,
        kind: "truck",
        lane: -1,
        connector: null,
        s: 0,
        speed: 0,
        max: TRUCK.max,
        length: TRUCK.length,
        color: 0,
        held: false,
        next: -1,
        committed: false,
        cooldown: 4 + index * 7,
        toDepot: false,
        unload: 0,
        unloaded: false
    };
}
function makeCar(id, seed) {
    const van = hash(seed + 0.5) < 0.22;
    return {
        id,
        kind: van ? "van" : "car",
        lane: -1,
        connector: null,
        s: 0,
        speed: 0,
        max: 2.9 + hash(seed) * 0.9,
        length: van ? 2.6 : 2.2,
        color: Math.floor(hash(seed + 1.7) * 7),
        held: false,
        next: -1,
        committed: false,
        cooldown: 0,
        toDepot: false,
        unload: 0,
        unloaded: false
    };
}
function addVehicle(traffic, vehicle) {
    const lanes = traffic.lanes.filter((lane)=>lane.bay < 0 && lane.length > vehicle.length + 3);
    for(let attempt = 0; attempt < 200; attempt++){
        const lane = lanes[Math.floor(random(traffic) * lanes.length)];
        const s = vehicle.length / 2 + 0.5 + random(traffic) * (lane.length - vehicle.length - 1);
        const clear = traffic.vehicles.every((other)=>other.connector || other.lane !== lane.id || Math.abs(other.s - s) > (other.length + vehicle.length) / 2 + 2);
        if (!clear) continue;
        Object.assign(vehicle, {
            lane: lane.id,
            s,
            speed: vehicle.max * 0.6
        });
        traffic.vehicles.push(vehicle);
        vehicle.next = chooseNext(traffic, vehicle);
        return;
    }
}
function syncTrucks(traffic, truckIds) {
    const wanted = new Set(truckIds);
    traffic.vehicles = traffic.vehicles.filter((vehicle)=>vehicle.kind !== "truck" || wanted.has(vehicle.id));
    traffic.occupants = traffic.occupants.map((occupant)=>occupant && wanted.has(occupant.id) ? occupant : null);
    const existing = new Set(traffic.vehicles.map((vehicle)=>vehicle.id));
    truckIds.forEach((id, index)=>{
        if (!existing.has(id)) addVehicle(traffic, makeTruck(id, index));
    });
}
function chooseNext(traffic, vehicle) {
    const lane = traffic.lanes[vehicle.lane];
    const node = traffic.nodes[lane.to];
    const depot = traffic.depot;
    if (lane.bay >= 0) return depot.afterExitLane;
    const options = node.outgoing.filter((id)=>traffic.lanes[id].bay < 0 && angleDiff(traffic.lanes[id].heading, lane.heading) < Math.PI * 0.8);
    if (vehicle.kind === "truck" && depot && vehicle.toDepot) {
        if (lane.id === depot.entryLane) {
            const bay = traffic.occupants.indexOf(null);
            if (bay >= 0) {
                traffic.occupants[bay] = vehicle;
                return depot.bays[bay].lane;
            }
            vehicle.toDepot = false;
            vehicle.cooldown = 10;
        } else {
            const score = (id)=>id === depot.entryLane ? 0 : traffic.lanes[id].to === depot.entry ? Infinity : traffic.lanes[id].length + traffic.depotDistance[traffic.lanes[id].to];
            const best = options.sort((a, b)=>score(a) - score(b))[0];
            if (best !== undefined) return best;
        }
    }
    if (!options.length) return node.outgoing.find((id)=>traffic.lanes[id].bay < 0) ?? lane.id;
    const weights = options.map((id)=>angleDiff(traffic.lanes[id].heading, lane.heading) < 0.3 ? 3 : 1);
    let pick = random(traffic) * sum(weights);
    for(let k = 0; k < options.length; k++){
        pick -= weights[k];
        if (pick <= 0) return options[k];
    }
    return options[options.length - 1];
}
function occupancy(traffic) {
    const map = new Map();
    const push = (lane, item)=>{
        const list = map.get(lane);
        if (list) list.push(item);
        else map.set(lane, [
            item
        ]);
    };
    const depot = traffic.depot;
    for (const vehicle of traffic.vehicles){
        if (vehicle.connector) push(vehicle.connector.to, {
            d: vehicle.s - vehicle.connector.length,
            length: vehicle.length,
            vehicle
        });
        else push(vehicle.lane, {
            d: vehicle.s,
            length: vehicle.length,
            vehicle
        });
        if (depot && !vehicle.connector && traffic.lanes[vehicle.lane].bay >= 0) {
            const bay = depot.bays[traffic.lanes[vehicle.lane].bay];
            if (vehicle.s < bay.leave) push(depot.afterEntryLane, {
                d: vehicle.s,
                length: vehicle.length,
                vehicle
            });
            if (vehicle.s > bay.merge) push(depot.beforeExitLane, {
                d: traffic.lanes[depot.beforeExitLane].length - (bay.length - vehicle.s),
                length: vehicle.length,
                vehicle
            });
        }
    }
    return map;
}
function hasRoom(occupied, vehicle, laneId, own) {
    return (occupied.get(laneId) ?? []).every((item)=>{
        if (item.vehicle === vehicle) return true;
        if (item.vehicle.connector) return item.vehicle.connector === own;
        return item.d - item.length / 2 >= vehicle.length + 1.5;
    });
}
function mayEnter(traffic, occupied, vehicle) {
    const lane = traffic.lanes[vehicle.lane];
    const own = connector(traffic, lane.id, vehicle.next);
    if (!hasRoom(occupied, vehicle, vehicle.next, own)) return false;
    const depot = traffic.depot;
    if (lane.bay >= 0 && depot) {
        const before = traffic.lanes[depot.beforeExitLane];
        const bay = depot.bays[lane.bay];
        const busy = (occupied.get(before.id) ?? []).some((item)=>item.vehicle !== vehicle && item.d + item.length / 2 > before.length - (bay.length - bay.merge) - 9);
        const crossing = traffic.vehicles.some((other)=>other.connector?.to === depot.afterExitLane && other !== vehicle);
        return !busy && !crossing;
    }
    return lightState(traffic, lane.to, lane.id) === "green";
}
function stepTraffic(traffic, dt) {
    if (dt <= 0) return;
    traffic.time += dt;
    const occupied = occupancy(traffic);
    const moves = traffic.vehicles.map((vehicle)=>planMove(traffic, occupied, vehicle, dt));
    traffic.vehicles.forEach((vehicle, k)=>advance(traffic, vehicle, moves[k]));
    const depot = traffic.depot;
    if (!depot) return;
    depot.bays.forEach((bay, index)=>{
        const truck = traffic.occupants[index];
        const doors = traffic.doors[index];
        for (const door of [
            0,
            1
        ]){
            const inside = truck && !truck.connector && truck.lane === bay.lane;
            const near = inside ? Math.abs(truck.s - bay.doors[door]) < 4.6 && !(truck.unload > 0) : false;
            doors[door] += Math.max(-1.2 * dt, Math.min(1.2 * dt, (near ? 1 : 0) - doors[door]));
        }
    });
}
function planMove(traffic, occupied, vehicle, dt) {
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
        for (const item of occupied.get(own.to) ?? []){
            if (item.vehicle !== vehicle && item.d > myDistance) gap = Math.min(gap, item.d - myDistance - (item.length + vehicle.length) / 2);
        }
    } else {
        for (const item of occupied.get(lane.id) ?? []){
            if (item.vehicle !== vehicle && item.d > vehicle.s) gap = Math.min(gap, item.d - vehicle.s - (item.length + vehicle.length) / 2);
        }
        const remaining = lane.length - vehicle.s;
        const bay = lane.bay >= 0 ? traffic.depot.bays[lane.bay] : null;
        const line = bay ? bay.merge - vehicle.s : remaining - vehicle.length / 2 - 0.3;
        let go = vehicle.committed || line >= 4;
        if (!go) {
            if (vehicle.next >= 0 && mayEnter(traffic, occupied, vehicle)) {
                go = true;
                vehicle.committed = line < 1;
            } else stop = Math.min(stop, line);
        }
        if (go) {
            const own = connector(traffic, lane.id, vehicle.next);
            const myDistance = -(remaining + own.length);
            for (const item of occupied.get(vehicle.next) ?? []){
                if (item.vehicle !== vehicle && item.d > myDistance) gap = Math.min(gap, item.d - myDistance - (item.length + vehicle.length) / 2);
            }
        }
        if (bay) {
            if (!vehicle.unloaded) stop = Math.min(stop, bay.stop - vehicle.s);
            for (const door of [
                0,
                1
            ]){
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
function advance(traffic, vehicle, distance) {
    vehicle.s += distance;
    const depot = traffic.depot;
    for(let guard = 0; guard < 4; guard++){
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
        if (lane.bay >= 0 && !vehicle.unloaded && vehicle.s >= depot.bays[lane.bay].stop - 1e-6) {
            vehicle.unloaded = true;
            vehicle.unload = UNLOAD_SECONDS;
            vehicle.speed = 0;
        }
        if (vehicle.s < lane.length) return;
        if (lane.bay >= 0) {
            traffic.occupants[lane.bay] = null;
            vehicle.toDepot = false;
            vehicle.unloaded = false;
            vehicle.cooldown = 25 + hash(vehicle.id.length + traffic.time) * 25;
        }
        vehicle.s -= lane.length;
        vehicle.connector = connector(traffic, lane.id, vehicle.next);
    }
}
function vehiclePose(traffic, vehicle) {
    if (vehicle.connector) return polylinePose(vehicle.connector.path, vehicle.connector.cumulative, vehicle.s);
    const lane = traffic.lanes[vehicle.lane];
    return polylinePose(lane.path, lane.cumulative, vehicle.s);
}
function polylinePose(path, cumulative, distance) {
    const total = cumulative[cumulative.length - 1];
    const d = Math.max(0, Math.min(total, distance));
    let k = 1;
    while(k < cumulative.length - 1 && cumulative[k] < d)k++;
    const a = path[k - 1], b = path[k];
    const span = cumulative[k] - cumulative[k - 1] || 1;
    const t = (d - cumulative[k - 1]) / span;
    return {
        x: a.x + (b.x - a.x) * t,
        z: a.z + (b.z - a.z) * t,
        heading: headingOf(b.x - a.x, b.z - a.z)
    };
}
function bayState(traffic, index) {
    const truck = traffic.occupants[index];
    return !truck ? "free" : truck.unload > 0 ? "unloading" : "moving";
}
function streetParts(traffic) {
    const parts = {
        sidewalks: [],
        asphalt: [],
        markings: [],
        poles: []
    };
    const white = "#ffffff";
    for (const segment of traffic.segments){
        const length = Math.hypot(segment.x2 - segment.x1, segment.z2 - segment.z1);
        const heading = headingOf(segment.x2 - segment.x1, segment.z2 - segment.z1);
        const cx = (segment.x1 + segment.x2) / 2, cz = (segment.z1 + segment.z2) / 2;
        parts.sidewalks.push({
            x: cx,
            y: 0.006,
            z: cz,
            sx: length + ROAD_WIDTH + SIDEWALK * 2,
            sy: 0.012,
            sz: ROAD_WIDTH + SIDEWALK * 2,
            rotation: heading,
            color: "#e4e9f2"
        });
        parts.asphalt.push({
            x: cx,
            y: 0.014,
            z: cz,
            sx: length + ROAD_WIDTH,
            sy: 0.012,
            sz: ROAD_WIDTH,
            rotation: heading,
            color: "#c7cfe3"
        });
        const [fx, fz] = forward(heading);
        for(let u = TRIM_SIGNAL + 0.8; u < length - TRIM_SIGNAL - 1.4; u += 2.4){
            parts.markings.push({
                x: segment.x1 + fx * (u + 0.6),
                y: 0.022,
                z: segment.z1 + fz * (u + 0.6),
                sx: 1.2,
                sy: 0.01,
                sz: 0.14,
                rotation: heading,
                color: white
            });
        }
    }
    for (const node of traffic.nodes){
        if (!node.signal) continue;
        for (const laneId of node.incoming){
            const lane = traffic.lanes[laneId];
            const end = lane.path[lane.path.length - 1];
            const [fx, fz] = forward(lane.heading), [rx, rz] = right(lane.heading);
            parts.markings.push({
                x: end.x - fx * 0.15,
                y: 0.022,
                z: end.z - fz * 0.15,
                sx: 0.18,
                sy: 0.01,
                sz: ROAD_WIDTH / 2 - 0.15,
                rotation: lane.heading,
                color: white
            });
            const zebra = {
                x: node.x - fx * (TRIM_SIGNAL - 0.35),
                z: node.z - fz * (TRIM_SIGNAL - 0.35)
            };
            for(let k = -2; k <= 2; k++){
                parts.markings.push({
                    x: zebra.x + rx * k * 0.68,
                    y: 0.022,
                    z: zebra.z + rz * k * 0.68,
                    sx: 0.55,
                    sy: 0.01,
                    sz: 0.36,
                    rotation: lane.heading,
                    color: white
                });
            }
            const pole = {
                x: end.x + rx * (LANE_OFFSET + 0.9) + fx * 0.3,
                z: end.z + rz * (LANE_OFFSET + 0.9) + fz * 0.3
            };
            parts.poles.push({
                x: pole.x,
                y: 1.1,
                z: pole.z,
                sx: 0.12,
                sy: 2.2,
                sz: 0.12,
                rotation: lane.heading,
                color: "#3a4256"
            });
            parts.poles.push({
                x: pole.x,
                y: 2.3,
                z: pole.z,
                sx: 0.32,
                sy: 0.62,
                sz: 0.32,
                rotation: lane.heading,
                color: "#2b3140"
            });
        }
    }
    return parts;
}
function signalLamps(traffic) {
    const lamps = [];
    for (const node of traffic.nodes){
        if (!node.signal) continue;
        for (const laneId of node.incoming){
            const lane = traffic.lanes[laneId];
            const end = lane.path[lane.path.length - 1];
            const [fx, fz] = forward(lane.heading), [rx, rz] = right(lane.heading);
            lamps.push({
                x: end.x + rx * (LANE_OFFSET + 0.9) + fx * 0.3 - fx * 0.17,
                y: 2.3,
                z: end.z + rz * (LANE_OFFSET + 0.9) + fz * 0.3 - fz * 0.17,
                node: node.id,
                lane: laneId
            });
        }
    }
    return lamps;
}
const WALLS = [
    "#fdf6ec",
    "#f3f7fb",
    "#e8f1ff",
    "#e9f7ef",
    "#fff1e6",
    "#f5efff"
];
const ROOFS = [
    "#e06c5a",
    "#5a6b86",
    "#b0714a",
    "#4f7bd9",
    "#7c8796"
];
const ACCENTS = [
    "#f6a821",
    "#1fb5a8",
    "#ec5a5f",
    "#8b5cf6",
    "#3563f0"
];
const GLASS = "#9ec1ea", DARK = "#33405a";
const LOT_STEP = 5.2, LOT_DEPTH = 2.7, SETBACK = 0.9;
function layoutTown(traffic, parks = []) {
    const town = {
        boxes: [],
        roofs: [],
        lawns: [],
        trees: [],
        footprints: []
    };
    const depot = traffic.depot;
    const inDepot = (x, z, margin)=>{
        if (!depot) return false;
        const [fx, fz] = forward(depot.base.heading), [rx, rz] = right(depot.base.heading);
        const u = (x - depot.base.x) * fx + (z - depot.base.z) * fz, v = (x - depot.base.x) * rx + (z - depot.base.z) * rz;
        return Math.abs(u) < DEPOT.approach + margin && v > -margin && v < DEPOT.far + 1.5 + margin;
    };
    traffic.cells.forEach((cell, cellIndex)=>{
        const inset = ROAD_WIDTH / 2 + SIDEWALK;
        const w = cell.w - inset * 2, d = cell.d - inset * 2;
        if (w < 4 || d < 4) return;
        town.lawns.push({
            x: cell.x,
            y: 0.01,
            z: cell.z,
            sx: w,
            sy: 0.02,
            sz: d,
            rotation: 0,
            color: "#cfe8d3"
        });
        const edges = [
            {
                x: cell.x,
                z: cell.z - d / 2,
                nx: 0,
                nz: -1,
                length: w,
                end: 0.6
            },
            {
                x: cell.x,
                z: cell.z + d / 2,
                nx: 0,
                nz: 1,
                length: w,
                end: 0.6
            },
            {
                x: cell.x - w / 2,
                z: cell.z,
                nx: -1,
                nz: 0,
                length: d,
                end: SETBACK + LOT_DEPTH + 1
            },
            {
                x: cell.x + w / 2,
                z: cell.z,
                nx: 1,
                nz: 0,
                length: d,
                end: SETBACK + LOT_DEPTH + 1
            }
        ];
        edges.forEach((edge, edgeIndex)=>{
            const usable = edge.length - edge.end * 2;
            const count = Math.floor(usable / LOT_STEP);
            const tx = -edge.nz, tz = edge.nx;
            for(let k = 0; k < count; k++){
                const along = -usable / 2 + LOT_STEP * (k + 0.5) + (usable - count * LOT_STEP) / 2;
                const x = edge.x + tx * along - edge.nx * (SETBACK + LOT_DEPTH / 2), z = edge.z + tz * along - edge.nz * (SETBACK + LOT_DEPTH / 2);
                if (inDepot(x, z, 2.5)) continue;
                const seed = cellIndex * 1000 + edgeIndex * 100 + k;
                const kind = hash(seed);
                const rotation = Math.atan2(edge.nx, edge.nz);
                if (kind < 0.1) {
                    town.trees.push({
                        x,
                        z,
                        scale: 0.9 + hash(seed + 3) * 0.4
                    });
                    continue;
                }
                addBuilding(town, kind < 0.6 ? "house" : kind < 0.82 ? "shop" : "office", x, z, rotation, seed);
            }
        });
        const yard = {
            w: w - (SETBACK + LOT_DEPTH + 0.6) * 2,
            d: d - (SETBACK + LOT_DEPTH + 0.6) * 2
        };
        if (yard.w > 1.5 && yard.d > 1.5) scatterTrees(town, {
            x: cell.x,
            z: cell.z,
            ...yard
        }, cellIndex, depot ? (x, z)=>inDepot(x, z, 1.5) : null);
    });
    parks.forEach((park, index)=>{
        town.lawns.push({
            x: park.x,
            y: 0.02,
            z: park.z,
            sx: park.w,
            sy: 0.02,
            sz: park.d,
            rotation: 0,
            color: "#cfe8d3"
        });
        scatterTrees(town, {
            x: park.x,
            z: park.z,
            w: park.w - 1.2,
            d: park.d - 1.2
        }, 500 + index, null, 0.68, 3.6);
    });
    return town;
}
function scatterTrees(town, area, seed, skip, sparse = 0.45, step = 2.8) {
    const cols = Math.max(1, Math.floor(area.w / step)), rows = Math.max(1, Math.floor(area.d / step));
    for(let i = 0; i < cols; i++)for(let j = 0; j < rows; j++){
        const h = hash(seed * 31 + i * 7 + j * 13);
        if (h < sparse) continue;
        const x = area.x - area.w / 2 + (i + 0.5) * (area.w / cols) + (h - 0.7) * 0.8;
        const z = area.z - area.d / 2 + (j + 0.5) * (area.d / rows) + (hash(h * 97) - 0.5) * 0.8;
        if (skip && skip(x, z)) continue;
        town.trees.push({
            x,
            z,
            scale: 0.75 + h * 0.45
        });
    }
}
function addBuilding(town, type, x, z, rotation, seed) {
    const cos = Math.cos(rotation), sin = Math.sin(rotation);
    const part = (lx, y, lz, sx, sy, sz, color)=>({
            x: x + lx * cos + lz * sin,
            y,
            z: z - lx * sin + lz * cos,
            sx,
            sy,
            sz,
            rotation,
            color
        });
    const wall = WALLS[Math.floor(hash(seed + 1) * WALLS.length)];
    if (type === "house") {
        const w = 2.6 + hash(seed + 2) * 0.6, d = 2.3, h = 1.4 + hash(seed + 4) * 0.3;
        town.boxes.push(part(0, h / 2, 0, w, h, d, wall), part(-w * 0.18, 0.45, d / 2 + 0.02, 0.5, 0.9, 0.05, ROOFS[Math.floor(hash(seed + 5) * ROOFS.length)]), part(w * 0.22, h * 0.6, d / 2 + 0.02, 0.6, 0.45, 0.05, GLASS), part(w * 0.3, h + 0.55, -d * 0.2, 0.3, 0.7, 0.3, "#9aa5b8"));
        town.roofs.push(part(0, h, 0, w + 0.3, 0.95, d + 0.35, ROOFS[Math.floor(hash(seed + 6) * ROOFS.length)]));
        town.footprints.push({
            x,
            z,
            r: Math.hypot(w, d) / 2
        });
    } else if (type === "shop") {
        const w = 3.8, d = 2.7, h = 1.7, accent = ACCENTS[Math.floor(hash(seed + 7) * ACCENTS.length)];
        town.boxes.push(part(0, h / 2, 0, w, h, d, wall), part(0, h + 0.08, 0, w + 0.2, 0.16, d + 0.2, DARK), part(0, 0.6, d / 2 + 0.02, w * 0.62, 0.8, 0.05, GLASS), part(0, 1.25, d / 2 + 0.4, w * 0.9, 0.1, 0.8, accent), part(0, h + 0.45, d / 2 - 0.1, w * 0.5, 0.45, 0.12, accent));
        town.footprints.push({
            x,
            z,
            r: Math.hypot(w, d) / 2
        });
    } else {
        const w = 3, d = 2.7, floors = 2 + Math.floor(hash(seed + 8) * 3), h = floors * 1.05 + 0.3;
        town.boxes.push(part(0, h / 2, 0, w, h, d, "#dfe6f1"), part(0, h + 0.07, 0, w + 0.15, 0.14, d + 0.15, "#9aa5b8"));
        for(let f = 0; f < floors; f++)town.boxes.push(part(0, 0.75 + f * 1.05, 0, w + 0.04, 0.4, d + 0.04, GLASS));
        town.footprints.push({
            x,
            z,
            r: Math.hypot(w, d) / 2
        });
    }
}
"""
