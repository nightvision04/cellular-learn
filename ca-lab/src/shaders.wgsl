// General GPU kernels for CA Lab (packed version).
//
// Cell states are stored as bytes, 4 cells per u32 word along x, so one
// thread updates 4 neighbouring cells per step. A "simulation" (job) is one
// candidate rule run on one scenario with one random seed; dispatch y = job.
//
// Nothing here is specific to any disease: number of states (<= 8), terrain
// types, grid size (width a multiple of 4), steps and the rule table are all
// data. The model class matches the CPU: next state depends on (current state,
// terrain type, number of "infectious" Moore neighbours 0..8), via per-row
// cumulative thresholds over a 31-bit uniform.

struct P {
    n_cells: u32, width: u32, height: u32, k: u32,
    nt: u32, rows: u32, t: u32, slot: u32,
    n_sims: u32, nd: u32, tl: u32, n_samples: u32,
    mode: u32, steps1: u32, n_slots: u32, n_words: u32,
};

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;            // sims * n_words (packed)
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;      // sims * n_words (packed)
@group(0) @binding(3) var<storage, read> terrain: array<u32>;        // scenarios * n_words (packed)
@group(0) @binding(4) var<storage, read> th: array<u32>;             // rules * rows * (k-1)
@group(0) @binding(5) var<storage, read> jobs: array<vec4<u32>>;     // (rule, scenario, seed_lo, seed_hi)
@group(0) @binding(6) var<storage, read> lut: array<u32>;            // [0,8) infectious, [8,16) tracked index, [16,24) quiet
@group(0) @binding(7) var<storage, read_write> counts: array<atomic<u32>>; // sims * samples * nd * tl
@group(0) @binding(8) var<storage, read> dmap: array<u32>;           // scenarios * n_cells (district or NONE), unpacked
@group(0) @binding(9) var<storage, read> init: array<u32>;           // scenarios * n_words (packed)
@group(0) @binding(10) var<storage, read> frames: array<u32>;        // scenarios * steps1 * n_words (packed truth)
@group(0) @binding(11) var<storage, read_write> acc: array<atomic<u32>>;   // sims * n_slots * 2 (total, correct)

const NONE: u32 = 0xFFFFFFFFu;
const MAXC: u32 = 1024u; // workgroup-local district counters (nd * tl must fit)

var<workgroup> wcount: array<atomic<u32>, 1024>;
var<workgroup> wacc: array<atomic<u32>, 2>;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn byte(w: u32, i: u32) -> u32 { return (w >> (8u * i)) & 0xFFu; }

// word at (row y, word column xw) of the current grid; mode 1 reads the truth frame at time t
fn word(sim: u32, scen: u32, y: i32, xw: i32) -> u32 {
    let ww = i32(p.width / 4u);
    let idx = u32(y * ww + xw);
    if (p.mode == 1u) {
        return frames[(scen * p.steps1 + p.t) * p.n_words + idx];
    }
    return src[sim * p.n_words + idx];
}

@compute @workgroup_size(256)
fn k_init(@builtin(global_invocation_id) g: vec3<u32>) {
    let wi = g.x; let sim = g.y;
    if (wi >= p.n_words || sim >= p.n_sims) { return; }
    let j = jobs[sim];
    dst[sim * p.n_words + wi] = init[j.y * p.n_words + wi];
}

@compute @workgroup_size(256)
fn k_step(@builtin(global_invocation_id) g: vec3<u32>) {
    let wi = g.x; let sim = g.y;
    if (wi >= p.n_words || sim >= p.n_sims) { return; }
    let j = jobs[sim];
    let ww = i32(p.width / 4u);
    let y = i32(wi) / ww;
    let xw = i32(wi) % ww;
    let h = i32(p.height);
    // infectious flags for x = 4xw-1 .. 4xw+4 in rows y-1, y, y+1 (0 outside the grid)
    var colsum: array<u32, 6>;
    for (var q = 0u; q < 6u; q++) { colsum[q] = 0u; }
    var mid: u32 = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        let yy = y + dy;
        if (yy < 0 || yy >= h) { continue; }
        let c = word(sim, j.y, yy, xw);
        if (dy == 0) { mid = c; }
        for (var q = 0u; q < 4u; q++) { colsum[q + 1u] += lut[byte(c, q)]; }
        if (xw > 0) { colsum[0] += lut[byte(word(sim, j.y, yy, xw - 1), 3u)]; }
        if (xw < ww - 1) { colsum[5] += lut[byte(word(sim, j.y, yy, xw + 1), 0u)]; }
    }
    let tw = terrain[j.y * p.n_words + wi];
    let km1 = p.k - 1u;
    var out: u32 = 0u;
    for (var q = 0u; q < 4u; q++) {
        let s = byte(mid, q);
        let n = colsum[q] + colsum[q + 1u] + colsum[q + 2u] - lut[s];
        let tt = byte(tw, q);
        let row = (s * p.nt + tt) * 9u + n;
        let cell = wi * 4u + q;
        let u = pcg(j.z ^ pcg(j.w ^ pcg(p.t ^ pcg(cell)))) >> 1u; // 31-bit uniform
        let tb = (j.x * p.rows + row) * km1;
        var ns = 0u;
        for (var r = 0u; r < km1; r++) {
            if (u >= th[tb + r]) { ns += 1u; }
        }
        out |= ns << (8u * q);
    }
    dst[sim * p.n_words + wi] = out;
}

// District counts: accumulate in workgroup memory, then one global add per counter.
@compute @workgroup_size(256)
fn k_observe(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let wi = g.x; let sim = g.y;
    let nc = p.nd * p.tl;
    for (var i = li; i < nc; i += 256u) { atomicStore(&wcount[i], 0u); }
    workgroupBarrier();
    if (wi < p.n_words && sim < p.n_sims) {
        let j = jobs[sim];
        let c = src[sim * p.n_words + wi];
        for (var q = 0u; q < 4u; q++) {
            let jx = lut[8u + byte(c, q)];
            if (jx == NONE) { continue; }
            let d = dmap[j.y * p.n_cells + wi * 4u + q];
            if (d == NONE) { continue; }
            atomicAdd(&wcount[d * p.tl + jx], 1u);
        }
    }
    workgroupBarrier();
    if (sim < p.n_sims) {
        for (var i = li; i < nc; i += 256u) {
            let v = atomicLoad(&wcount[i]);
            if (v != 0u) {
                atomicAdd(&counts[(sim * p.n_samples + p.slot) * nc + i], v);
            }
        }
    }
}

// Accuracy vs ground-truth frame t: counts cells that are not "quiet" in both.
@compute @workgroup_size(256)
fn k_acc(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let wi = g.x; let sim = g.y;
    if (li < 2u) { atomicStore(&wacc[li], 0u); }
    workgroupBarrier();
    if (wi < p.n_words && sim < p.n_sims) {
        let j = jobs[sim];
        let a = src[sim * p.n_words + wi];
        let b = frames[(j.y * p.steps1 + p.t) * p.n_words + wi];
        var tot = 0u; var ok = 0u;
        for (var q = 0u; q < 4u; q++) {
            let sa = byte(a, q); let sb = byte(b, q);
            if (lut[16u + sa] == 1u && lut[16u + sb] == 1u) { continue; }
            tot += 1u;
            if (sa == sb) { ok += 1u; }
        }
        if (tot > 0u) { atomicAdd(&wacc[0], tot); }
        if (ok > 0u) { atomicAdd(&wacc[1], ok); }
    }
    workgroupBarrier();
    if (li == 0u && sim < p.n_sims) {
        let base = (sim * p.n_slots + p.slot) * 2u;
        let t0 = atomicLoad(&wacc[0]);
        let o0 = atomicLoad(&wacc[1]);
        if (t0 > 0u) { atomicAdd(&acc[base], t0); }
        if (o0 > 0u) { atomicAdd(&acc[base + 1u], o0); }
    }
}
