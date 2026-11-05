//! CA Lab: learn the hidden local rule of a spatial process on a 100x100 grid
//! with a genetic algorithm, and score it on held-out scenarios.
//!
//! The "model" is a rule table: for every (current state, terrain type,
//! number of infectious Moore neighbours 0..8) it stores logits over the next
//! state. Deterministic experiments take the argmax; stochastic experiments
//! sample from the softmax. The genetic algorithm only ever sees simulated
//! observations of the ground-truth process, never the ground-truth rule.
//!
//! Usage: ca-lab <exp1|exp2|exp3|exp4> [site_data_dir]

mod gpu;

use rayon::prelude::*;
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const W: usize = 100;
pub const H: usize = 100;
pub const N: usize = W * H;
const DIST: usize = 10; // districts are DIST x DIST cells -> 10 x 10 districts
const ND: usize = (W / DIST) * (H / DIST);
const WATER: u8 = 2;

// ---------------------------------------------------------------- randomness

#[inline]
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Counter-based uniform in [0,1): the same (seed, t, cell) always gives the
/// same number, so runs are reproducible and parallel-safe.
#[inline]
#[allow(dead_code)]
fn unif(seed: u64, t: u64, i: u64) -> f32 {
    (mix(seed ^ mix(t.wrapping_mul(0x9E37_79B9) ^ mix(i ^ 0xABCD))) >> 40) as f32 / (1u64 << 24) as f32
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.0)
    }
    /// wyrand-style step: one multiply, good enough for simulation noise.
    #[inline]
    fn u32f(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0xA076_1D64_78BD_642F);
        let m = (self.0 as u128).wrapping_mul((self.0 ^ 0xE703_7ED1_A0B4_28DB) as u128);
        (((m >> 64) as u64 ^ m as u64) >> 40) as f32 / (1u64 << 24) as f32
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.f().max(1e-12);
        let u2 = self.f();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------- scenarios

#[derive(Clone)]
struct Scenario {
    init: Vec<u8>,
    terrain: Vec<u8>,
}

fn blobs(rng: &mut Rng, g: &mut [u8], terrain: &[u8], count: usize, rmin: usize, rmax: usize, fill: f64, state: u8) {
    let mut placed = 0;
    let mut tries = 0;
    while placed < count && tries < 10_000 {
        tries += 1;
        let cx = rng.below(W) as i32;
        let cy = rng.below(H) as i32;
        if terrain[cy as usize * W + cx as usize] == WATER {
            continue;
        }
        let r = (rmin + rng.below(rmax - rmin + 1)) as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                let (x, y) = (cx + dx, cy + dy);
                if x < 0 || y < 0 || x >= W as i32 || y >= H as i32 || dx * dx + dy * dy > r * r {
                    continue;
                }
                let i = y as usize * W + x as usize;
                if terrain[i] != WATER && rng.f() < fill {
                    g[i] = state;
                }
            }
        }
        placed += 1;
    }
}

fn scen_exp1(seed: u64) -> Scenario {
    let mut rng = Rng(seed);
    let terrain = vec![0u8; N];
    let mut g = vec![0u8; N];
    let n = 4 + rng.below(5);
    blobs(&mut rng, &mut g, &terrain, n, 2, 4, 0.55, 1);
    Scenario { init: g, terrain }
}

fn scen_exp2(seed: u64) -> Scenario {
    let mut rng = Rng(seed);
    let terrain = vec![0u8; N];
    let mut g = vec![0u8; N];
    let n = 3 + rng.below(5);
    blobs(&mut rng, &mut g, &terrain, n, 0, 1, 1.0, 1);
    Scenario { init: g, terrain }
}

fn scen_exp3(seed: u64) -> Scenario {
    let mut rng = Rng(seed);
    let terrain = vec![0u8; N];
    let mut g = vec![0u8; N];
    let n = 2 + rng.below(4);
    blobs(&mut rng, &mut g, &terrain, n, 1, 2, 0.6, 1);
    Scenario { init: g, terrain }
}

/// A map with towns (urban), countryside (rural) and a river plus lakes
/// (water, nobody lives there). Each seed gives a different map.
fn scen_exp4(seed: u64) -> Scenario {
    let mut rng = Rng(seed);
    let mut terrain = vec![0u8; N];
    // towns
    let towns = 4 + rng.below(4);
    for _ in 0..towns {
        let cx = rng.below(W) as f64;
        let cy = rng.below(H) as f64;
        let r = 4.0 + rng.f() * 8.0;
        for y in 0..H {
            for x in 0..W {
                let d = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
                if d < r * (0.75 + 0.25 * rng.f()) {
                    terrain[y * W + x] = 1;
                }
            }
        }
    }
    // a meandering river from top to bottom
    let mut x = 20.0 + rng.f() * 60.0;
    let width = 2.0 + rng.f() * 2.0;
    let phase = rng.f() * 6.28;
    for y in 0..H {
        x += (y as f64 * 0.08 + phase).sin() * 0.9 + (rng.f() - 0.5) * 0.6;
        for xx in 0..W {
            if (xx as f64 - x).abs() < width {
                terrain[y * W + xx] = WATER;
            }
        }
    }
    // a couple of lakes
    for _ in 0..2 {
        let cx = rng.below(W) as f64;
        let cy = rng.below(H) as f64;
        let r = 3.0 + rng.f() * 5.0;
        for y in 0..H {
            for x in 0..W {
                if ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt() < r {
                    terrain[y * W + x] = WATER;
                }
            }
        }
    }
    let mut g: Vec<u8> = terrain.iter().map(|&t| if t == WATER { 3 } else { 0 }).collect();
    let n = 3 + rng.below(3);
    blobs(&mut rng, &mut g, &terrain, n, 0, 1, 1.0, 1);
    Scenario { init: g, terrain }
}

// ---------------------------------------------------------------- ground truth rules

type Rule = fn(u8, u8, u8, f32) -> u8;

/// Complex contagion: a susceptible cell becomes infected only when at least
/// TWO of its 8 neighbours are infected. Infection is permanent.
fn truth_exp1(s: u8, n: u8, _t: u8, _u: f32) -> u8 {
    if s == 0 && n >= 2 { 1 } else { s }
}
fn base_persist(s: u8, _n: u8, _t: u8, _u: f32) -> u8 {
    s
}
fn base_any_neighbour(s: u8, n: u8, _t: u8, _u: f32) -> u8 {
    if s == 0 && n >= 1 { 1 } else { s }
}

/// SIR with a fixed 3-step infectious period: S -> I1 if any neighbour is
/// infectious, then I1 -> I2 -> I3 -> R. Recovered cells are immune forever.
fn truth_exp2(s: u8, n: u8, _t: u8, _u: f32) -> u8 {
    match s {
        0 => if n >= 1 { 1 } else { 0 },
        1 => 2,
        2 => 3,
        3 => 4,
        _ => 4,
    }
}
fn base_si_no_recovery(s: u8, n: u8, _t: u8, _u: f32) -> u8 {
    if s == 0 && n >= 1 { 1 } else { s }
}

/// Stochastic SIR: each infectious neighbour independently transmits with
/// probability BETA per step; an infectious cell recovers with probability GAMMA.
const B3: f32 = 0.12;
const G3: f32 = 0.08;
fn truth_exp3(s: u8, n: u8, _t: u8, u: f32) -> u8 {
    match s {
        0 => {
            let p = 1.0 - (1.0 - B3).powi(n as i32);
            if u < p { 1 } else { 0 }
        }
        1 => if u < G3 { 2 } else { 1 },
        _ => 2,
    }
}

/// Stochastic SIR on a map: transmission is much higher in towns, water is
/// empty (state 3) and never changes.
const B4_URBAN: f32 = 0.30;
const B4_RURAL: f32 = 0.06;
const G4: f32 = 0.10;
fn truth_exp4(s: u8, n: u8, t: u8, u: f32) -> u8 {
    if t == WATER {
        return 3;
    }
    match s {
        0 => {
            let b = if t == 1 { B4_URBAN } else { B4_RURAL };
            let p = 1.0 - (1.0 - b).powi(n as i32);
            if u < p { 1 } else { 0 }
        }
        1 => if u < G4 { 2 } else { 1 },
        2 => 2,
        _ => 3,
    }
}

// ---------------------------------------------------------------- experiment definitions

struct Exp {
    id: &'static str,
    title: &'static str,
    summary: &'static str,
    blurb: &'static str,
    states: Vec<(&'static str, &'static str)>,
    infectious: Vec<bool>,
    quiet: Vec<bool>,
    tracked: Vec<usize>,
    n_terrain: usize,
    steps: usize,
    stochastic: bool,
    truth: Rule,
    scenario: fn(u64) -> Scenario,
    pop: usize,
    gens: usize,
    n_train: usize,
    n_test: usize,
    rollouts: usize,
    det_baselines: Vec<(&'static str, Rule)>,
    stay_bias: f32,
    monotone: bool,
    smooth: f64,
}

/// More varied training outbreaks: random number and size of clusters, and
/// sometimes a share of the population already immune at the start.
fn varied(rng: &mut Rng, terrain: &[u8], immune_state: u8, max_immune: f64, rmax: usize) -> Vec<u8> {
    let mut g: Vec<u8> = terrain.iter().map(|&t| if t == WATER { 3 } else { 0 }).collect();
    let n = 1 + rng.below(8);
    for _ in 0..n {
        let r = rng.below(rmax + 1);
        let fill = 0.5 + rng.f() * 0.5;
        blobs(rng, &mut g, terrain, 1, r, r, fill, 1);
    }
    let imm = if rng.f() < 0.5 { rng.f() * max_immune } else { 0.0 };
    for i in 0..N {
        if g[i] == 0 && rng.f() < imm {
            g[i] = immune_state;
        }
    }
    g
}
fn scen_exp2b(seed: u64) -> Scenario {
    let mut rng = Rng(seed ^ 0xB2);
    let terrain = vec![0u8; N];
    let g = varied(&mut rng, &terrain, 4, 0.15, 3);
    Scenario { init: g, terrain }
}
fn scen_exp3b(seed: u64) -> Scenario {
    let mut rng = Rng(seed ^ 0xB3);
    let terrain = vec![0u8; N];
    let g = varied(&mut rng, &terrain, 2, 0.15, 4);
    Scenario { init: g, terrain }
}
fn scen_exp4b(seed: u64) -> Scenario {
    let terrain = scen_exp4(seed).terrain;
    let mut rng = Rng(seed ^ 0xB4);
    let g = varied(&mut rng, &terrain, 2, 0.15, 3);
    Scenario { init: g, terrain }
}

fn experiment(id: &str) -> Exp {
    let b_blurb = " <b>Version b:</b> two changes versus the original. (1) Built-in assumptions: more sick neighbours can never lower your chance of getting sick (enforced), and rules that change smoothly with the number of sick neighbours are preferred. (2) Training outbreaks are more varied: different cluster sizes and sometimes up to 15% already immune. The tweaked-start tests are still more extreme than anything in training.";
    match id {
        "exp2b" => {
            let mut e = experiment("exp2");
            e.id = "exp2b";
            e.title = "2b · SIR waves, varied training starts";
            e.summary = "varied starts";
            e.blurb = Box::leak(format!("{} <b>Version b:</b> same problem, but training outbreaks have varied cluster sizes and sometimes up to 15% already immune, so the situations that were never seen in experiment 2 now show up in training.", e.blurb).into_boxed_str());
            e.scenario = scen_exp2b;
            return e;
        }
        "exp3b" => {
            let mut e = experiment("exp3");
            e.id = "exp3b";
            e.title = "3b · Stochastic SIR + assumptions, varied starts";
            e.summary = "constrained, varied";
            e.blurb = Box::leak(format!("{}{}", e.blurb, b_blurb).into_boxed_str());
            e.scenario = scen_exp3b;
            e.monotone = true;
            e.smooth = 0.01;
            e.gens = 3000;
            return e;
        }
        "exp4b" => {
            let mut e = experiment("exp4");
            e.id = "exp4b";
            e.title = "4b · Map epidemic + assumptions, varied starts";
            e.summary = "constrained, varied";
            e.blurb = Box::leak(format!("{}{}", e.blurb, b_blurb).into_boxed_str());
            e.scenario = scen_exp4b;
            e.monotone = true;
            e.smooth = 0.01;
            e.gens = 4000;
            return e;
        }
        _ => {}
    }
    let sir = vec![("Susceptible", "#e9e6dc"), ("Infectious", "#d64545"), ("Recovered", "#2f6fdb")];
    match id {
        "exp1" => Exp {
            id: "exp1",
            title: "1 · Threshold contagion",
            summary: "deterministic, 2 states",
            blurb: "The simplest problem. A rumour (or a fire) spreads on a 100×100 grid, but a cell only catches it when <b>at least two</b> of its 8 neighbours already have it. The genetic algorithm sees only before-and-after grids from 6 training outbreaks and must discover the rule. Score = accuracy on cells that are or should be infected, over a 50-step held-out outbreak it never saw.",
            states: vec![("Not reached", "#e9e6dc"), ("Reached", "#d64545")],
            infectious: vec![false, true],
            quiet: vec![true, false],
            tracked: vec![1],
            n_terrain: 1,
            steps: 50,
            stochastic: false,
            truth: truth_exp1,
            scenario: scen_exp1,
            pop: 64,
            gens: 60,
            n_train: 6,
            n_test: 3,
            rollouts: 1,
            det_baselines: vec![("Persistence (nothing changes)", base_persist), ("Naive rule (1 neighbour is enough)", base_any_neighbour)],
            stay_bias: 0.0,
            monotone: false,
            smooth: 0.0,
        },
        "exp2" => Exp {
            id: "exp2",
            title: "2 · SIR waves (fixed illness length)",
            summary: "deterministic, 5 states",
            blurb: "A disease with a fixed 3-day illness. A healthy cell catches it from any sick neighbour, is sick for exactly 3 steps (shown as three shades of red), then is immune forever. This makes expanding rings that annihilate when they meet. The GA has to discover both the infection rule and the 3-step clock from data. Score = accuracy on active cells over a 60-step held-out outbreak.",
            states: vec![("Susceptible", "#e9e6dc"), ("Sick day 1", "#f08a8a"), ("Sick day 2", "#d64545"), ("Sick day 3", "#8f1f1f"), ("Immune", "#2f6fdb")],
            infectious: vec![false, true, true, true, false],
            quiet: vec![true, false, false, false, false],
            tracked: vec![1, 2, 3, 4],
            n_terrain: 1,
            steps: 60,
            stochastic: false,
            truth: truth_exp2,
            scenario: scen_exp2,
            pop: 96,
            gens: 150,
            n_train: 6,
            n_test: 3,
            rollouts: 1,
            det_baselines: vec![("Persistence (nothing changes)", base_persist), ("Spread, never recover", base_si_no_recovery)],
            stay_bias: 0.0,
            monotone: false,
            smooth: 0.0,
        },
        "exp3" => Exp {
            id: "exp3",
            title: "3 · Stochastic SIR epidemic",
            summary: "random, district counts",
            blurb: "Now chance matters, like a real epidemic: each sick neighbour passes the disease on with some probability per day, and sick cells recover at random. Individual cells can no longer be predicted, so — like real surveillance data — the GA only sees <b>daily sick and recovered counts in each of 100 districts</b> (10×10 cells each) from 4 training outbreaks. Score = error (RMSE) in predicted district-level sick/recovered fractions on held-out outbreaks, averaged over 8 runs. Compared against a classic non-spatial SIR model fitted to the same data, and against the true rule itself (the best anyone can do, because of chance).",
            states: sir.clone(),
            infectious: vec![false, true, false],
            quiet: vec![true, false, false],
            tracked: vec![1, 2],
            n_terrain: 1,
            steps: 80,
            stochastic: true,
            truth: truth_exp3,
            scenario: scen_exp3,
            pop: 64,
            gens: 600,
            n_train: 4,
            n_test: 3,
            rollouts: 4,
            det_baselines: vec![],
            stay_bias: 2.5,
            monotone: false,
            smooth: 0.0,
        },
        "exp4" => Exp {
            id: "exp4",
            title: "4 · Epidemic on a map (towns, countryside, river)",
            summary: "random, terrain-dependent",
            blurb: "The same kind of epidemic, but on a landscape: it spreads fast in towns, slowly in the countryside, and not at all across water. The rule table now also depends on the terrain under each cell. The held-out test uses <b>new maps</b> with different towns and rivers, so the model has to learn how terrain changes spread, not memorise one map. Data are district-level daily counts, as in experiment 3.",
            states: vec![("Susceptible", "#e9e6dc"), ("Infectious", "#d64545"), ("Recovered", "#2f6fdb"), ("Water", "#7fb3c9")],
            infectious: vec![false, true, false, false],
            quiet: vec![true, false, false, true],
            tracked: vec![1, 2],
            n_terrain: 3,
            steps: 90,
            stochastic: true,
            truth: truth_exp4,
            scenario: scen_exp4,
            pop: 64,
            gens: 800,
            n_train: 4,
            n_test: 3,
            rollouts: 4,
            det_baselines: vec![],
            stay_bias: 2.5,
            monotone: false,
            smooth: 0.0,
        },
        _ => panic!("unknown experiment {id}"),
    }
}

// ---------------------------------------------------------------- simulation

struct Traj {
    frames: Vec<Vec<u8>>,  // only the kept frames
    national: Vec<f64>,    // (steps+1) x tracked, fraction of populated cells
    district: Vec<f64>,    // samples x ND x tracked
}

const DT: usize = 4; // district counts are observed every DT steps

fn populated(terrain: &[u8]) -> Vec<bool> {
    terrain.iter().map(|&t| t != WATER).collect()
}

fn observe(g: &[u8], pop: &[bool], k: usize, tracked: &[usize], nat: &mut Vec<f64>, dist: Option<&mut Vec<f64>>) {
    let mut cnt = vec![0usize; k];
    let mut np = 0usize;
    for i in 0..N {
        if pop[i] {
            cnt[g[i] as usize] += 1;
            np += 1;
        }
    }
    for &s in tracked {
        nat.push(cnt[s] as f64 / np.max(1) as f64);
    }
    if let Some(dv) = dist {
        let dw = W / DIST;
        let mut dc = vec![0usize; ND * k];
        let mut dp = vec![0usize; ND];
        for y in 0..H {
            for x in 0..W {
                let i = y * W + x;
                if pop[i] {
                    let d = (y / DIST) * dw + x / DIST;
                    dc[d * k + g[i] as usize] += 1;
                    dp[d] += 1;
                }
            }
        }
        for d in 0..ND {
            for &s in tracked {
                dv.push(if dp[d] == 0 { 0.0 } else { dc[d * k + s] as f64 / dp[d] as f64 });
            }
        }
    }
}

/// Run a rule from a scenario for `steps` steps. `keep` lists the time steps
/// whose full grids should be returned (empty = none, None = all).
fn rollout<F: Fn(u8, u8, u8, f32) -> u8>(
    exp: &Exp,
    scn: &Scenario,
    rule: &F,
    seed: u64,
    keep: Option<&[usize]>,
) -> Traj {
    let k = exp.states.len();
    let pop = populated(&scn.terrain);
    let mut g = scn.init.clone();
    let mut nxt = vec![0u8; N];
    let mut mask = vec![0u8; (W + 2) * (H + 2)];
    let inf: Vec<u8> = exp.infectious.iter().map(|&b| b as u8).collect();
    let mut frames = Vec::new();
    let keep_at = |t: usize| match keep {
        None => true,
        Some(ks) => ks.contains(&t),
    };
    let mut national = Vec::with_capacity((exp.steps + 1) * exp.tracked.len());
    let mut district = Vec::new();
    for t in 0..=exp.steps {
        if keep_at(t) {
            frames.push(g.clone());
        }
        observe(&g, &pop, k, &exp.tracked, &mut national, if t % DT == 0 { Some(&mut district) } else { None });
        if t == exp.steps {
            break;
        }
        for y in 0..H {
            let row = (y + 1) * (W + 2) + 1;
            for x in 0..W {
                mask[row + x] = inf[g[y * W + x] as usize];
            }
        }
        const WP: usize = W + 2;
        // one cheap sequential generator per (rollout, step): reproducible, fast
        let mut r = Rng(mix(seed ^ mix(t as u64 + 1)));
        for y in 0..H {
            for x in 0..W {
                let c = (y + 1) * WP + x + 1;
                let n = mask[c - WP - 1] + mask[c - WP] + mask[c - WP + 1] + mask[c - 1] + mask[c + 1]
                    + mask[c + WP - 1] + mask[c + WP] + mask[c + WP + 1];
                let i = y * W + x;
                let u = if exp.stochastic { r.u32f() } else { 0.0 };
                nxt[i] = rule(g[i], n, scn.terrain[i], u);
            }
        }
        std::mem::swap(&mut g, &mut nxt);
    }
    Traj { frames, national, district }
}

/// One step from a given grid (used for the deterministic one-step score).
fn one_step<F: Fn(u8, u8, u8, f32) -> u8>(exp: &Exp, g: &[u8], terrain: &[u8], rule: &F) -> Vec<u8> {
    let inf: Vec<u8> = exp.infectious.iter().map(|&b| b as u8).collect();
    let mut out = vec![0u8; N];
    for y in 0..H {
        for x in 0..W {
            let mut n = 0u8;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (yy, xx) = (y as i32 + dy, x as i32 + dx);
                    if yy >= 0 && xx >= 0 && yy < H as i32 && xx < W as i32 {
                        n += inf[g[yy as usize * W + xx as usize] as usize];
                    }
                }
            }
            let i = y * W + x;
            out[i] = rule(g[i], n, terrain[i], 0.0);
        }
    }
    out
}

fn active_acc(exp: &Exp, a: &[u8], b: &[u8]) -> f64 {
    let (mut tot, mut ok) = (0usize, 0usize);
    for i in 0..N {
        if !(exp.quiet[a[i] as usize] && exp.quiet[b[i] as usize]) {
            tot += 1;
            if a[i] == b[i] {
                ok += 1;
            }
        }
    }
    if tot == 0 { 1.0 } else { ok as f64 / tot as f64 }
}

fn mean_vec(vs: &[Vec<f64>]) -> Vec<f64> {
    let mut m = vec![0.0; vs[0].len()];
    for v in vs {
        for (a, b) in m.iter_mut().zip(v) {
            *a += b;
        }
    }
    m.iter_mut().for_each(|a| *a /= vs.len() as f64);
    m
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
}

// ---------------------------------------------------------------- the model: a rule table

#[derive(Clone)]
struct Compiled {
    k: usize,
    nt: usize,
    det: Vec<u8>,
    cum: Vec<f32>,
    probs: Vec<f32>,
}

impl Compiled {
    fn from_exp(genes: &[f32], exp: &Exp) -> Self {
        Self::new(genes, exp.states.len(), exp.n_terrain, if exp.monotone { Some(&exp.infectious) } else { None })
    }

    /// `monotone`: if given (the infectious-state mask), the chance of moving
    /// INTO an infectious state is forced to never decrease as the number of
    /// infectious neighbours grows ("more sick neighbours never lowers your risk").
    fn new(genes: &[f32], k: usize, nt: usize, monotone: Option<&[bool]>) -> Self {
        let rows = k * nt * 9;
        let mut probs = vec![0f32; rows * k];
        for r in 0..rows {
            let row = &genes[r * k..(r + 1) * k];
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let e: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
            let s: f32 = e.iter().sum();
            for j in 0..k {
                probs[r * k + j] = e[j] / s;
            }
        }
        if let Some(inf) = monotone {
            for s in 0..k {
                for t in 0..nt {
                    for j in 0..k {
                        if j == s || !inf[j] {
                            continue;
                        }
                        let mut run = 0f32;
                        for n in 0..9 {
                            let r = ((s * nt + t) * 9 + n) * k;
                            let p = probs[r + j];
                            if p < run {
                                // raise this entry to the running max, shrink the others to keep a valid distribution
                                let scale = if 1.0 - p > 1e-9 { (1.0 - run) / (1.0 - p) } else { 0.0 };
                                for jj in 0..k {
                                    if jj != j {
                                        probs[r + jj] *= scale;
                                    }
                                }
                                probs[r + j] = run;
                            } else {
                                run = p;
                            }
                        }
                    }
                }
            }
        }
        let mut det = vec![0u8; rows];
        let mut cum = vec![0f32; rows * k];
        for r in 0..rows {
            let row = &probs[r * k..(r + 1) * k];
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            det[r] = row.iter().position(|&v| v == mx).unwrap() as u8;
            let mut c = 0.0;
            for j in 0..k {
                c += row[j];
                cum[r * k + j] = c;
            }
            cum[r * k + k - 1] = 1.0;
        }
        Compiled { k, nt, det, cum, probs }
    }

    /// How jagged the rule is across neighbour counts: mean over every
    /// (state, terrain, next state) of the summed squared jumps between
    /// consecutive neighbour counts.
    fn roughness(&self) -> f64 {
        let (k, nt) = (self.k, self.nt);
        let mut tot = 0.0;
        let mut groups = 0;
        for s in 0..k {
            for t in 0..nt {
                for j in 0..k {
                    let mut g = 0.0;
                    for n in 0..8 {
                        let a = self.probs[((s * nt + t) * 9 + n) * k + j] as f64;
                        let b = self.probs[((s * nt + t) * 9 + n + 1) * k + j] as f64;
                        g += (b - a).powi(2);
                    }
                    tot += g;
                    groups += 1;
                }
            }
        }
        tot / groups as f64
    }
    #[inline]
    fn row(&self, s: u8, n: u8, t: u8) -> usize {
        (s as usize * self.nt + (t as usize).min(self.nt - 1)) * 9 + n as usize
    }
    #[inline]
    fn det(&self, s: u8, n: u8, t: u8) -> u8 {
        self.det[self.row(s, n, t)]
    }
    #[inline]
    fn sample(&self, s: u8, n: u8, t: u8, u: f32) -> u8 {
        let r = self.row(s, n, t) * self.k;
        let mut j = 0;
        while j + 1 < self.k && u >= self.cum[r + j] {
            j += 1;
        }
        j as u8
    }
}

// ---------------------------------------------------------------- data + scoring

// ---------------------------------------------------------------- fast kernel (training + testing)

const WP: usize = W + 2;
const HP: usize = H + 2;
const NONE: u16 = u16::MAX;

/// Per-scenario layout for the fast kernel: a padded grid (1-cell border that
/// is never updated and never infectious), each cell's terrain offset into the
/// rule table, and its district.
struct Fast {
    g0: Vec<u8>,
    toff: Vec<u16>,
    dist: Vec<u16>,
    inv_pop: Vec<f64>,
}

impl Fast {
    fn new(scn: &Scenario) -> Self {
        let mut g0 = vec![0u8; WP * HP];
        let mut toff = vec![0u16; WP * HP];
        let mut dist = vec![NONE; WP * HP];
        let mut pop = vec![0usize; ND];
        for y in 0..H {
            for x in 0..W {
                let (i, p) = (y * W + x, (y + 1) * WP + x + 1);
                g0[p] = scn.init[i];
                toff[p] = scn.terrain[i] as u16 * 9;
                if scn.terrain[i] != WATER {
                    let d = (y / DIST) * (W / DIST) + x / DIST;
                    dist[p] = d as u16;
                    pop[d] += 1;
                }
            }
        }
        let inv_pop = pop.iter().map(|&n| if n == 0 { 0.0 } else { 1.0 / n as f64 }).collect();
        Fast { g0, toff, dist, inv_pop }
    }
}

const TH_ONE: u32 = 1 << 31; // uniforms are 31-bit, so a threshold of 2^31 is never reached

/// Rule as integer thresholds: next state = how many of the 3 thresholds the
/// cell's 31-bit uniform is >= to. Works for deterministic rules too (0 / 2^31).
fn thresholds(c: &Compiled) -> Vec<u32> {
    let rows = c.det.len();
    let mut th = vec![TH_ONE; rows * 3];
    for r in 0..rows {
        for j in 0..c.k - 1 {
            th[r * 3 + j] = ((c.cum[r * c.k + j] as f64) * TH_ONE as f64).round().min(TH_ONE as f64) as u32;
        }
    }
    th
}

#[inline(always)]
fn wy(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0xA076_1D64_78BD_642F);
    let m = (*s as u128).wrapping_mul((*s ^ 0xE703_7ED1_A0B4_28DB) as u128);
    ((m >> 64) as u64) ^ (m as u64)
}

/// Simulate and return only the district series (what training/testing need).
fn fast_district(exp: &Exp, f: &Fast, th: &[u32], seed: u64) -> Vec<f64> {
    let k = exp.states.len();
    let stride_s = (exp.n_terrain * 9) as usize;
    let mut infl = [0u8; 8];
    let mut lut = [255u8; 8];
    for s in 0..k {
        infl[s] = exp.infectious[s] as u8;
    }
    for (j, &s) in exp.tracked.iter().enumerate() {
        lut[s] = j as u8;
    }
    let tl = exp.tracked.len();
    let mut g = f.g0.clone();
    let mut m: Vec<u8> = g.iter().map(|&s| infl[s as usize]).collect();
    // border cells must not count as infectious
    for x in 0..WP {
        m[x] = 0;
        m[(HP - 1) * WP + x] = 0;
    }
    for y in 0..HP {
        m[y * WP] = 0;
        m[y * WP + WP - 1] = 0;
    }
    let mut ng = g.clone();
    let mut nm = m.clone();
    let mut cs = [0u8; WP];
    let samples = exp.steps / DT + 1;
    let mut out = Vec::with_capacity(samples * ND * tl);
    let mut cnt = vec![0u32; ND * tl];
    for t in 0..=exp.steps {
        if t % DT == 0 {
            cnt.iter_mut().for_each(|c| *c = 0);
            for y in 1..=H {
                for x in 1..=W {
                    let p = y * WP + x;
                    let d = f.dist[p];
                    let j = lut[g[p] as usize];
                    if d != NONE && j != 255 {
                        cnt[d as usize * tl + j as usize] += 1;
                    }
                }
            }
            for d in 0..ND {
                for j in 0..tl {
                    out.push(cnt[d * tl + j] as f64 * f.inv_pop[d]);
                }
            }
        }
        if t == exp.steps {
            break;
        }
        let mut st = mix(seed ^ mix(t as u64 + 1));
        for y in 1..=H {
            let (a, b, c) = ((y - 1) * WP, y * WP, (y + 1) * WP);
            for x in 0..WP {
                cs[x] = m[a + x] + m[b + x] + m[c + x];
            }
            for x in 1..=W {
                let p = b + x;
                let n = cs[x - 1] + cs[x] + cs[x + 1] - m[p];
                let row = g[p] as usize * stride_s + f.toff[p] as usize + n as usize;
                let u = (wy(&mut st) >> 33) as u32;
                let tr = &th[row * 3..row * 3 + 3];
                let ns = (u >= tr[0]) as u8 + (u >= tr[1]) as u8 + (u >= tr[2]) as u8;
                ng[p] = ns;
                nm[p] = infl[ns as usize];
            }
        }
        std::mem::swap(&mut g, &mut ng);
        std::mem::swap(&mut m, &mut nm);
    }
    out
}

fn train_err_fast(exp: &Exp, data: &Data, th: &[u32], gen: usize) -> f64 {
    train_err_fast_n(exp, data, th, gen, exp.rollouts)
}

fn train_err_fast_n(exp: &Exp, data: &Data, th: &[u32], gen: usize, nroll: usize) -> f64 {
    let mut err = 0.0;
    for (i, f) in data.train_fast.iter().enumerate() {
        let runs: Vec<Vec<f64>> = (0..nroll).map(|r| fast_district(exp, f, th, mix((gen * 1000 + i * 10 + r) as u64))).collect();
        err += rmse(&mean_vec(&runs), &data.train_obs[i]);
    }
    err / data.train.len() as f64
}

fn test_metric_fast(exp: &Exp, data: &Data, th: &[u32]) -> f64 {
    let mut err = 0.0;
    for (i, f) in data.test_fast.iter().enumerate() {
        let runs: Vec<Vec<f64>> = (0..TEST_ENS).into_par_iter().map(|r| fast_district(exp, f, th, 31_337 + (i * 100 + r) as u64)).collect();
        err += rmse(&mean_vec(&runs), &data.test_truth_mean[i]);
    }
    err / data.test.len() as f64
}

struct Data {
    train_fast: Vec<Fast>,
    test_fast: Vec<Fast>,
    train: Vec<Scenario>,
    test: Vec<Scenario>,
    train_frames: Vec<Vec<Vec<u8>>>,  // deterministic: all truth frames
    train_obs: Vec<Vec<f64>>,         // stochastic: one observed realization (district series)
    test_frames: Vec<Vec<Vec<u8>>>,   // deterministic
    test_truth_mean: Vec<Vec<f64>>,   // stochastic: ensemble mean district series
    test_truth_nat: Vec<Vec<f64>>,    // stochastic: ensemble mean national series
    gpu_train: Option<gpu::SceneSet>,
    gpu_test: Option<gpu::SceneSet>,
}

/// The GPU, if one is available and --cpu was not given. Set once in main.
static GPU: std::sync::OnceLock<Option<gpu::Gpu>> = std::sync::OnceLock::new();
fn the_gpu() -> Option<&'static gpu::Gpu> {
    GPU.get().and_then(|g| g.as_ref())
}

fn spec_of(exp: &Exp) -> gpu::Spec {
    gpu::Spec {
        width: W as u32, height: H as u32, k: exp.states.len() as u32, nt: exp.n_terrain as u32,
        steps: exp.steps as u32, dt: DT as u32, nd: ND as u32,
        infectious: exp.infectious.clone(), tracked: exp.tracked.clone(), quiet: exp.quiet.clone(),
    }
}

fn dmap_of(scn: &Scenario) -> Vec<u32> {
    (0..N).map(|i| if scn.terrain[i] == WATER { gpu::NONE } else { ((i / W / DIST) * (W / DIST) + (i % W) / DIST) as u32 }).collect()
}

/// Thresholds for the GPU: rows * (k-1), general in k. Deterministic rules
/// use the most likely next state (same as the CPU's `det`).
fn thresholds_k(c: &Compiled, deterministic: bool) -> Vec<u32> {
    let rows = c.det.len();
    let mut th = Vec::with_capacity(rows * (c.k - 1));
    for r in 0..rows {
        for j in 0..c.k - 1 {
            th.push(if deterministic {
                if j < c.det[r] as usize { 0 } else { TH_ONE }
            } else {
                ((c.cum[r * c.k + j] as f64) * TH_ONE as f64).round().min(TH_ONE as f64) as u32
            });
        }
    }
    th
}

/// District series for each job from GPU counts.
fn gpu_series(exp: &Exp, o: &gpu::Out, fast: &[Fast], jobs: &[(u32, u32, u64)]) -> Vec<Vec<f64>> {
    let tl = exp.tracked.len();
    let per = o.n_samples * ND * tl;
    jobs.iter().enumerate().map(|(q, &(_, scen, _))| {
        let ip = &fast[scen as usize].inv_pop;
        o.counts[q * per..(q + 1) * per].iter().enumerate().map(|(i, &c)| c as f64 * ip[(i / tl) % ND]).collect()
    }).collect()
}

/// Accuracy summary for one job: (mean rollout accuracy t=1..steps, mean one-step accuracy).
fn gpu_acc(o: &gpu::Out, q: usize, steps: usize, onestep: usize) -> (f64, f64) {
    let a = &o.acc[q * o.n_slots * 2..(q + 1) * o.n_slots * 2];
    let r = |slot: usize| { let (tot, ok) = (a[slot * 2], a[slot * 2 + 1]); if tot == 0 { 1.0 } else { ok as f64 / tot as f64 } };
    let roll = (1..=steps).map(r).sum::<f64>() / steps as f64;
    let ts: Vec<usize> = (0..steps).step_by(onestep).collect();
    let one = ts.iter().map(|&t| r(steps + 1 + t)).sum::<f64>() / ts.len() as f64;
    (roll, one)
}

/// Score a whole population on the GPU in one batch (higher is better).
fn gpu_scores(exp: &Exp, data: &Data, g: &gpu::Gpu, pop: &[Compiled], gen: usize) -> Vec<f64> {
    let spec = spec_of(exp);
    let th: Vec<u32> = pop.iter().flat_map(|c| thresholds_k(c, !exp.stochastic)).collect();
    let set = data.gpu_train.as_ref().unwrap();
    if exp.stochastic {
        let mut jobs = vec![];
        for gi in 0..pop.len() {
            for i in 0..data.train.len() {
                for r in 0..exp.rollouts {
                    jobs.push((gi as u32, i as u32, mix((gen * 1000 + i * 10 + r) as u64)));
                }
            }
        }
        let o = g.run(&spec, set, &th, &jobs, true, false, 0);
        let series = gpu_series(exp, &o, &data.train_fast, &jobs);
        pop.iter().enumerate().map(|(gi, c)| {
            let mut err = 0.0;
            for i in 0..data.train.len() {
                let b = (gi * data.train.len() + i) * exp.rollouts;
                err += rmse(&mean_vec(&series[b..b + exp.rollouts]), &data.train_obs[i]);
            }
            -err / data.train.len() as f64 - exp.smooth * c.roughness()
        }).collect()
    } else {
        let jobs: Vec<(u32, u32, u64)> = (0..pop.len()).flat_map(|gi| (0..data.train.len()).map(move |i| (gi as u32, i as u32, 0))).collect();
        let o = g.run(&spec, set, &th, &jobs, false, true, 5);
        (0..pop.len()).map(|gi| {
            (0..data.train.len()).map(|i| { let (roll, one) = gpu_acc(&o, gi * data.train.len() + i, exp.steps, 5); 0.5 * roll + 0.5 * one }).sum::<f64>() / data.train.len() as f64
        }).collect()
    }
}

/// Held-out metric of one rule on the GPU (same definition as test_metric).
fn gpu_test(exp: &Exp, data: &Data, g: &gpu::Gpu, c: &Compiled) -> f64 {
    let spec = spec_of(exp);
    let th = thresholds_k(c, !exp.stochastic);
    let set = data.gpu_test.as_ref().unwrap();
    if exp.stochastic {
        let jobs: Vec<(u32, u32, u64)> = (0..data.test.len()).flat_map(|i| (0..TEST_ENS).map(move |r| (0, i as u32, 31_337 + (i * 100 + r) as u64))).collect();
        let o = g.run(&spec, set, &th, &jobs, true, false, 0);
        let series = gpu_series(exp, &o, &data.test_fast, &jobs);
        (0..data.test.len()).map(|i| rmse(&mean_vec(&series[i * TEST_ENS..(i + 1) * TEST_ENS]), &data.test_truth_mean[i])).sum::<f64>() / data.test.len() as f64
    } else {
        let jobs: Vec<(u32, u32, u64)> = (0..data.test.len()).map(|i| (0, i as u32, 0)).collect();
        let o = g.run(&spec, set, &th, &jobs, false, true, 0);
        (0..data.test.len()).map(|i| gpu_acc_roll(&o, i, exp.steps)).sum::<f64>() / data.test.len() as f64
    }
}

fn gpu_acc_roll(o: &gpu::Out, q: usize, steps: usize) -> f64 {
    let a = &o.acc[q * o.n_slots * 2..(q + 1) * o.n_slots * 2];
    (1..=steps).map(|t| { let (tot, ok) = (a[t * 2], a[t * 2 + 1]); if tot == 0 { 1.0 } else { ok as f64 / tot as f64 } }).sum::<f64>() / steps as f64
}

const TEST_ENS: usize = 8;

fn build_data(exp: &Exp) -> Data {
    let train: Vec<Scenario> = (0..exp.n_train).map(|i| (exp.scenario)(1000 + i as u64)).collect();
    let test: Vec<Scenario> = (0..exp.n_test).map(|i| (exp.scenario)(9_000_000 + i as u64)).collect();
    let truth = exp.truth;
    let train_fast = train.iter().map(Fast::new).collect();
    let test_fast = test.iter().map(Fast::new).collect();
    let mut d = Data { train_fast, test_fast, train, test, train_frames: vec![], train_obs: vec![], test_frames: vec![], test_truth_mean: vec![], test_truth_nat: vec![], gpu_train: None, gpu_test: None };
    if exp.stochastic {
        d.train_obs = d.train.iter().enumerate().map(|(i, s)| rollout(exp, s, &truth, 77 + i as u64, Some(&[])).district).collect();
        for (i, s) in d.test.iter().enumerate() {
            let runs: Vec<Traj> = (0..TEST_ENS).into_par_iter().map(|r| rollout(exp, s, &truth, 5_000 + (i * 100 + r) as u64, Some(&[]))).collect();
            d.test_truth_mean.push(mean_vec(&runs.iter().map(|t| t.district.clone()).collect::<Vec<_>>()));
            d.test_truth_nat.push(mean_vec(&runs.iter().map(|t| t.national.clone()).collect::<Vec<_>>()));
        }
    } else {
        d.train_frames = d.train.iter().map(|s| rollout(exp, s, &truth, 0, None).frames).collect();
        d.test_frames = d.test.iter().map(|s| rollout(exp, s, &truth, 0, None).frames).collect();
    }
    if let (Some(g), true) = (the_gpu(), spec_of(exp).gpu_ok()) {
        let up = |sc: &[Scenario], fr: &[Vec<Vec<u8>>]| {
            let init: Vec<Vec<u8>> = sc.iter().map(|s| s.init.clone()).collect();
            let terr: Vec<Vec<u8>> = sc.iter().map(|s| s.terrain.clone()).collect();
            let dm: Vec<Vec<u32>> = sc.iter().map(dmap_of).collect();
            g.scenes(&init, &terr, &dm, if exp.stochastic { None } else { Some(fr) })
        };
        d.gpu_train = Some(up(&d.train, &d.train_frames));
        d.gpu_test = Some(up(&d.test, &d.test_frames));
    }
    d
}

/// Training score (higher is better).
/// Stochastic training error of any rule: RMSE between the mean of a few
/// model runs and the single observed realization, per training outbreak.
fn train_err_stoch<F: Fn(u8, u8, u8, f32) -> u8>(exp: &Exp, data: &Data, rule: &F, gen: usize) -> f64 {
    let mut err = 0.0;
    for (i, s) in data.train.iter().enumerate() {
        let runs: Vec<Vec<f64>> = (0..exp.rollouts)
            .map(|r| rollout(exp, s, rule, mix((gen * 1000 + i * 10 + r) as u64), Some(&[])).district)
            .collect();
        err += rmse(&mean_vec(&runs), &data.train_obs[i]);
    }
    err / data.train.len() as f64
}

fn train_score(exp: &Exp, data: &Data, c: &Compiled, gen: usize) -> f64 {
    if exp.stochastic {
        let th = thresholds(c);
        -train_err_fast(exp, data, &th, gen) - exp.smooth * c.roughness()
    } else {
        let rule = |s, n, t, _u| c.det(s, n, t);
        let mut sc = 0.0;
        for (i, s) in data.train.iter().enumerate() {
            let tf = &data.train_frames[i];
            let model = rollout(exp, s, &rule, 0, None).frames;
            let roll: f64 = (1..tf.len()).map(|t| active_acc(exp, &tf[t], &model[t])).sum::<f64>() / (tf.len() - 1) as f64;
            let mut one = 0.0;
            let stride = 5;
            let mut cnt = 0;
            for t in (0..tf.len() - 1).step_by(stride) {
                one += active_acc(exp, &tf[t + 1], &one_step(exp, &tf[t], &s.terrain, &rule));
                cnt += 1;
            }
            sc += 0.5 * roll + 0.5 * one / cnt as f64;
        }
        sc / data.train.len() as f64
    }
}

/// Held-out metric in the experiment's own units (accuracy or RMSE).
fn test_metric<F: Fn(u8, u8, u8, f32) -> u8 + Sync>(exp: &Exp, data: &Data, rule: &F) -> f64 {
    if exp.stochastic {
        let mut err = 0.0;
        for (i, s) in data.test.iter().enumerate() {
            let runs: Vec<Vec<f64>> = (0..TEST_ENS)
                .into_par_iter()
                .map(|r| rollout(exp, s, rule, 31_337 + (i * 100 + r) as u64, Some(&[])).district)
                .collect();
            err += rmse(&mean_vec(&runs), &data.test_truth_mean[i]);
        }
        err / data.test.len() as f64
    } else {
        let mut acc = 0.0;
        for (i, s) in data.test.iter().enumerate() {
            let tf = &data.test_frames[i];
            let model = rollout(exp, s, rule, 0, None).frames;
            acc += (1..tf.len()).map(|t| active_acc(exp, &tf[t], &model[t])).sum::<f64>() / (tf.len() - 1) as f64;
        }
        acc / data.test.len() as f64
    }
}

// ---------------------------------------------------------------- non-spatial baseline (classic SIR)

/// Well-mixed discrete SIR: everyone in a district meets everyone else.
fn sir_series(s0: f64, i0: f64, r0: f64, beta: f64, gamma: f64, steps: usize) -> Vec<(f64, f64)> {
    let (mut s, mut i, mut r) = (s0, i0, r0);
    let mut out = vec![(i, r)];
    for _ in 0..steps {
        let new_inf = s * (1.0 - (-beta * i).exp());
        let rec = gamma * i;
        s -= new_inf;
        i += new_inf - rec;
        r += rec;
        out.push((i, r));
    }
    out
}

fn district_init(scn: &Scenario) -> Vec<(f64, f64, f64)> {
    let dw = W / DIST;
    let mut c = vec![(0.0, 0.0, 0.0, 0.0); ND];
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            if scn.terrain[i] == WATER {
                continue;
            }
            let d = (y / DIST) * dw + x / DIST;
            c[d].3 += 1.0;
            match scn.init[i] {
                0 => c[d].0 += 1.0,
                1 => c[d].1 += 1.0,
                2 => c[d].2 += 1.0,
                _ => {}
            }
        }
    }
    c.iter().map(|&(s, i, r, n)| if n == 0.0 { (0.0, 0.0, 0.0) } else { (s / n, i / n, r / n) }).collect()
}

/// District-level series predicted by isolated well-mixed SIR per district.
fn sir_district_series(scn: &Scenario, beta: f64, gamma: f64, steps: usize) -> Vec<f64> {
    let init = district_init(scn);
    let per: Vec<Vec<(f64, f64)>> = init.iter().map(|&(s, i, r)| sir_series(s, i, r, beta, gamma, steps)).collect();
    let mut out = Vec::new();
    for t in (0..=steps).step_by(DT) {
        for d in 0..ND {
            out.push(per[d][t].0);
            out.push(per[d][t].1);
        }
    }
    out
}

fn fit_well_mixed(exp: &Exp, data: &Data) -> (f64, f64) {
    // grid search on the training district observations
    let mut best = (f64::MAX, 0.0, 0.0);
    for bi in 1..=60 {
        let beta = bi as f64 * 0.05;
        for gi in 1..=40 {
            let gamma = gi as f64 * 0.005;
            let mut e = 0.0;
            for (k, s) in data.train.iter().enumerate() {
                e += rmse(&sir_district_series(s, beta, gamma, exp.steps), &data.train_obs[k]);
            }
            if e < best.0 {
                best = (e, beta, gamma);
            }
        }
    }
    (best.1, best.2)
}

// ---------------------------------------------------------------- output

#[derive(Serialize, serde::Deserialize, Clone)]
struct Hist {
    gen: usize,
    train: f64,
    test: f64,
}

#[derive(Clone)]
struct Snap {
    gen: usize,
    test: f64,
    truth: Vec<String>,
    model: Vec<String>,
    curves: Map<String, Value>,
}

fn grid_str(g: &[u8]) -> String {
    g.iter().map(|&s| (b'0' + s) as char).collect()
}

struct Writer {
    dir: PathBuf,
}

impl Writer {
    fn write_atomic(&self, name: &str, v: &Value) {
        let tmp = self.dir.join(format!(".{name}.{}.tmp", std::process::id()));
        fs::write(&tmp, serde_json::to_vec(v).unwrap()).unwrap();
        fs::rename(&tmp, self.dir.join(name)).unwrap();
    }
    fn update_manifest(&self, exp: &Exp, status: &str, stamp: &str, summary: &str) {
        let p = self.dir.join("manifest.json");
        let mut m: Value = fs::read(&p).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!({"experiments": []}));
        let entry = json!({"id": exp.id, "title": exp.title, "status": status, "file": format!("{}.json", exp.id), "summary": summary, "stamp": stamp});
        let list = m["experiments"].as_array_mut().unwrap();
        if let Some(e) = list.iter_mut().find(|e| e["id"] == exp.id) {
            *e = entry;
        } else {
            list.push(entry);
        }
        list.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
        let now = std::process::Command::new("date").arg("+%Y-%m-%d %H:%M:%S").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
        m["updated"] = json!(now);
        self.write_atomic("manifest.json", &m);
    }
}

/// Transition probabilities of any rule, row by row: probability that a cell
/// in (state, terrain, n infectious neighbours) moves to each next state.
fn rule_probs<F: Fn(u8, u8, u8, f32) -> u8>(k: usize, nt: usize, f: &F) -> Vec<f64> {
    const S: usize = 2000;
    let mut out = vec![0.0; k * nt * 9 * k];
    for s in 0..k {
        for t in 0..nt {
            for n in 0..9 {
                let r = (s * nt + t) * 9 + n;
                for j in 0..S {
                    let u = (j as f32 + 0.5) / S as f32;
                    out[r * k + f(s as u8, n as u8, t as u8, u) as usize] += 1.0 / S as f64;
                }
            }
        }
    }
    out
}

/// How often each (state, terrain, n) situation occurs in the training data.
fn usage(exp: &Exp, data: &Data) -> Vec<u64> {
    let nt = exp.n_terrain;
    let cnt = std::cell::RefCell::new(vec![0u64; exp.states.len() * nt * 9]);
    let tr = exp.truth;
    let f = |s: u8, n: u8, t: u8, u: f32| {
        cnt.borrow_mut()[(s as usize * nt + (t as usize).min(nt - 1)) * 9 + n as usize] += 1;
        tr(s, n, t, u)
    };
    for (i, s) in data.train.iter().enumerate() {
        rollout(exp, s, &f, 77 + i as u64, Some(&[]));
    }
    cnt.into_inner()
}

fn thin(snaps: &[Snap], max: usize) -> Vec<&Snap> {
    if snaps.len() <= max {
        return snaps.iter().collect();
    }
    let mut idx: Vec<usize> = (0..max).map(|j| j * (snaps.len() - 1) / (max - 1)).collect();
    idx.dedup();
    idx.iter().map(|&i| &snaps[i]).collect()
}

// ---------------------------------------------------------------- main loop

// ---------------------------------------------------------------- tweaked starting conditions

fn base_terrain(exp: &Exp) -> Vec<u8> {
    if exp.n_terrain == 3 { scen_exp4(9_000_000).terrain } else { vec![0u8; N] }
}

fn start_grid(terrain: &[u8]) -> Vec<u8> {
    terrain.iter().map(|&t| if t == WATER { 3 } else { 0 }).collect()
}

/// Starting conditions unlike anything in training. Each returns (name, description, scenario).
fn ood_scenarios(exp: &Exp) -> Vec<(String, String, Scenario)> {
    let terrain = base_terrain(exp);
    let seed_state = 1u8;
    let mut out = vec![];
    let mut rng = Rng(424242);
    let immune_state: Option<u8> = match &exp.id[..4] { "exp2" => Some(4), "exp3" | "exp4" => Some(2), _ => None };

    if exp.n_terrain == 1 {
        // one big outbreak in the centre
        let mut g = start_grid(&terrain);
        for y in 0..H {
            for x in 0..W {
                let d = ((x as f64 - 50.0).powi(2) + (y as f64 - 50.0).powi(2)).sqrt();
                if d < 7.0 {
                    g[y * W + x] = seed_state;
                }
            }
        }
        out.push(("Big central outbreak".into(), "One solid outbreak of radius 7 in the middle (training only had small scattered outbreaks).".into(), Scenario { init: g, terrain: terrain.clone() }));
        // a line across the grid
        let mut g = start_grid(&terrain);
        for x in 5..95 {
            g[50 * W + x] = seed_state;
            if exp.id == "exp1" {
                g[51 * W + x] = seed_state;
            }
        }
        out.push(("Line of infection".into(), "A thin line of infection across the whole grid, like an outbreak along a road.".into(), Scenario { init: g, terrain: terrain.clone() }));
        // many scattered seeds
        let mut g = start_grid(&terrain);
        for i in 0..N {
            if rng.f() < 0.01 {
                g[i] = seed_state;
            }
        }
        if exp.id == "exp1" {
            // threshold rule needs pairs to get going
            let mut g2 = g.clone();
            for i in 0..N {
                if g[i] == 1 && i + 1 < N {
                    g2[i + 1] = 1;
                }
            }
            g = g2;
        }
        out.push(("Many scattered seeds".into(), "About 100 tiny seeds spread all over the grid instead of a few clusters.".into(), Scenario { init: g, terrain: terrain.clone() }));
    } else {
        // start inside the biggest town / out in the countryside
        let pick = |want: u8, rng: &mut Rng| -> usize {
            loop {
                let i = rng.below(N);
                let (x, y) = (i % W, i / W);
                if x < 3 || y < 3 || x > W - 4 || y > H - 4 {
                    continue;
                }
                let ok = (0..9).all(|k| terrain[(y + k / 3 - 1) * W + x + k % 3 - 1] == want);
                if ok {
                    return i;
                }
            }
        };
        let mut g = start_grid(&terrain);
        let c = pick(1, &mut rng);
        for d in [0, 1, W, W + 1] {
            g[c + d] = seed_state;
        }
        out.push(("Starts inside a town".into(), "A single small outbreak in the middle of a town.".into(), Scenario { init: g, terrain: terrain.clone() }));
        let mut g = start_grid(&terrain);
        let c = pick(0, &mut rng);
        for d in [0, 1, W, W + 1] {
            g[c + d] = seed_state;
        }
        out.push(("Starts in the countryside".into(), "A single small outbreak out in the countryside, away from towns.".into(), Scenario { init: g, terrain: terrain.clone() }));
    }
    if let Some(imm) = immune_state {
        let mut base = (exp.scenario)(9_000_000);
        let mut r2 = Rng(777);
        for i in 0..N {
            if base.init[i] == 0 && r2.f() < 0.30 {
                base.init[i] = imm;
            }
        }
        out.push(("30% already immune".into(), "The usual held-out outbreak, but 30% of people start out immune (no training outbreak had any immune people at the start).".into(), base));
    }
    out
}

/// Score the best rule saved in site/data/<id>.json on the tweaked starts.
fn run_ood(id: &str, dir: &Path) {
    let exp = experiment(id);
    let k = exp.states.len();
    let p = dir.join(format!("{id}.json"));
    let mut doc: Value = serde_json::from_slice(&fs::read(&p).expect("run the experiment first")).unwrap();
    let genes: Vec<f32> = if let Some(g) = doc["rule"].get("genes") {
        serde_json::from_value(g.clone()).unwrap()
    } else {
        // older output: rebuild logits from the saved probabilities
        let learned: Vec<f64> = serde_json::from_value(doc["rule"]["learned"].clone()).expect("no saved rule");
        learned.iter().map(|&q| if q <= 0.0 { -10.0 } else { q.ln() as f32 }).collect()
    };
    let c = Compiled::from_exp(&genes, &exp);
    let frame_times: Vec<usize> = (0..=exp.steps).step_by((exp.steps / 15).max(1)).collect();
    let tracked_names: Vec<&str> = exp.tracked.iter().map(|&s| exp.states[s].0).collect();
    let tl = exp.tracked.len();
    let curve = |nat: &[f64], j: usize| -> Vec<f64> { (0..=exp.steps).map(|t| nat[t * tl + j]).collect() };
    let truth = exp.truth;
    let mut wm: Option<(f64, f64)> = doc["wm"].as_array().map(|a| (a[0].as_f64().unwrap(), a[1].as_f64().unwrap()));
    if wm.is_none() && exp.stochastic {
        // older output without the fitted classic-SIR parameters: refit (deterministic)
        wm = Some(fit_well_mixed(&exp, &build_data(&exp)));
    }
    let mut results = vec![];
    for (name, desc, scn) in ood_scenarios(&exp) {
        let mut curves = Map::new();
        let mut baselines: Vec<Value> = vec![];
        let (metric, tf, mf);
        if exp.stochastic {
            let model = |s, n, t, u| c.sample(s, n, t, u);
            let ens = |rule: &(dyn Fn(u8, u8, u8, f32) -> u8 + Sync), base: u64| -> (Vec<f64>, Vec<f64>) {
                let runs: Vec<Traj> = (0..TEST_ENS).into_par_iter().map(|r| rollout(&exp, &scn, &|s, n, t, u| rule(s, n, t, u), base + r as u64, Some(&[]))).collect();
                (mean_vec(&runs.iter().map(|t| t.district.clone()).collect::<Vec<_>>()), mean_vec(&runs.iter().map(|t| t.national.clone()).collect::<Vec<_>>()))
            };
            let (td, tn) = ens(&truth, 5_000);
            let (md, mn) = ens(&model, 31_337);
            let (od, _) = ens(&truth, 88_000);
            metric = rmse(&md, &td);
            baselines.push(json!({"name": "True rule (best possible)", "value": rmse(&od, &td)}));
            if let Some((b, g)) = wm {
                baselines.push(json!({"name": "Classic SIR, fitted", "value": rmse(&sir_district_series(&scn, b, g, exp.steps), &td)}));
            }
            for (j, nm) in tracked_names.iter().enumerate() {
                curves.insert(format!("truth · {nm}"), json!(curve(&tn, j)));
                curves.insert(format!("model · {nm}"), json!(curve(&mn, j)));
            }
            tf = rollout(&exp, &scn, &truth, 4242, Some(&frame_times)).frames;
            mf = rollout(&exp, &scn, &model, 999, Some(&frame_times)).frames;
        } else {
            let model = |s, n, t, _u| c.det(s, n, t);
            let all_t = rollout(&exp, &scn, &truth, 0, None);
            let acc_of = |fr: &[Vec<u8>]| (1..fr.len()).map(|t| active_acc(&exp, &all_t.frames[t], &fr[t])).sum::<f64>() / (fr.len() - 1) as f64;
            let all_m = rollout(&exp, &scn, &model, 0, None);
            metric = acc_of(&all_m.frames);
            for (bn, br) in &exp.det_baselines {
                baselines.push(json!({"name": bn, "value": acc_of(&rollout(&exp, &scn, br, 0, None).frames)}));
            }
            for (j, nm) in tracked_names.iter().enumerate() {
                curves.insert(format!("truth · {nm}"), json!(curve(&all_t.national, j)));
                curves.insert(format!("model · {nm}"), json!(curve(&all_m.national, j)));
            }
            tf = frame_times.iter().map(|&t| all_t.frames[t].clone()).collect::<Vec<_>>();
            mf = frame_times.iter().map(|&t| all_m.frames[t].clone()).collect::<Vec<_>>();
        }
        println!("[{id}] tweak '{name}': model {metric:.4}  baselines {}", baselines.iter().map(|b| format!("{:.4}", b["value"].as_f64().unwrap())).collect::<Vec<_>>().join(" / "));
        results.push(json!({"name": name, "desc": desc, "test": metric, "baselines": baselines,
            "truth": tf.iter().map(|g| grid_str(g)).collect::<Vec<_>>(),
            "model": mf.iter().map(|g| grid_str(g)).collect::<Vec<_>>(), "curves": curves}));
    }
    doc["ood"] = json!(results);
    let w = Writer { dir: dir.to_path_buf() };
    w.write_atomic(&format!("{id}.json"), &doc);
    let summary = doc["history"].as_array().and_then(|h| h.last()).map(|h| format!("gen {} · {:.3}", h["gen"], h["test"].as_f64().unwrap())).unwrap_or_default();
    w.update_manifest(&exp, doc["status"].as_str().unwrap_or("done"), &format!("ood-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()), &summary);
}

/// `ca-lab gpubench <exp>`: time one generation on CPU vs GPU and check they agree.
fn gpubench(id: &str) {
    let g = the_gpu().expect("no GPU available");
    let exp = experiment(id);
    let k = exp.states.len();
    let data = build_data(&exp);
    let mut rng = Rng(12345);
    let rows = k * exp.n_terrain * 9;
    let pop: Vec<Compiled> = (0..exp.pop)
        .map(|_| {
            let mut gs: Vec<f32> = (0..rows * k).map(|_| (rng.normal() * 1.5) as f32).collect();
            for r in 0..rows {
                gs[r * k + r / (exp.n_terrain * 9)] += exp.stay_bias;
            }
            Compiled::from_exp(&gs, &exp)
        })
        .collect();
    println!("[{id}] GPU: {}", g.name);
    println!("[{id}] one generation = {} rules x {} outbreaks x {} runs x {} steps on {}x{}", exp.pop, exp.n_train, exp.rollouts, exp.steps, W, H);
    let _ = gpu_scores(&exp, &data, g, &pop[..2], 0); // warm-up (shader compile)
    let t = Instant::now();
    let cpu: Vec<f64> = pop.par_iter().map(|c| train_score(&exp, &data, c, 0)).collect();
    let tc = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let gpu_s = gpu_scores(&exp, &data, g, &pop, 0);
    let tg = t.elapsed().as_secs_f64();
    println!("  CPU (fast kernel, all cores): {tc:.3}s   GPU: {tg:.3}s   -> {:.1}x", tc / tg);
    let t = Instant::now();
    let _ = gpu_scores(&exp, &data, g, &pop, 1);
    println!("  GPU again (steady state): {:.3}s", t.elapsed().as_secs_f64());
    if !exp.stochastic {
        let md = cpu.iter().zip(&gpu_s).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        println!("  deterministic agreement: max |CPU - GPU| score over {} rules = {md:.2e}  ({})", pop.len(), if md < 1e-9 { "IDENTICAL" } else { "DIFFERENT" });
        let tr = exp.truth;
        let tp = rule_probs(k, exp.n_terrain, &tr);
        let tg: Vec<f32> = tp.iter().map(|&q| if q <= 0.0 { -30.0 } else { q.ln() as f32 }).collect();
        let tcmp = Compiled::new(&tg, k, exp.n_terrain, None);
        println!("  true rule held-out accuracy: CPU {:.4}  GPU {:.4}", test_metric(&exp, &data, &tr), gpu_test(&exp, &data, g, &tcmp));
        return;
    }
    let tr = exp.truth;
    let tp = rule_probs(k, exp.n_terrain, &tr);
    let tgn: Vec<f32> = tp.iter().map(|&q| if q <= 0.0 { -30.0 } else { q.ln() as f32 }).collect();
    let tcmp = Compiled::new(&tgn, k, exp.n_terrain, None);
    let spec = spec_of(&exp);
    let th = thresholds_k(&tcmp, false);
    let jobs: Vec<(u32, u32, u64)> = (0..400).map(|r| (0, 0, 900_000 + r)).collect();
    let o = g.run(&spec, data.gpu_train.as_ref().unwrap(), &th, &jobs, true, false, 0);
    let gs = gpu_series(&exp, &o, &data.train_fast, &jobs);
    let cs: Vec<Vec<f64>> = (0..400).into_par_iter().map(|r| rollout(&exp, &data.train[0], &tr, 900_000 + r, Some(&[])).district).collect();
    let (mg, mc) = (mean_vec(&gs), mean_vec(&cs));
    let tl = exp.tracked.len();
    for (j, &s) in exp.tracked.iter().enumerate() {
        for smp in [5usize, 10, exp.steps / DT] {
            let tot = |m: &[f64]| (0..ND).map(|d| m[smp * ND * tl + d * tl + j]).sum::<f64>() / ND as f64;
            println!("  true rule, mean {} share at t={:>3} (400 runs): CPU {:.4}  GPU {:.4}", exp.states[s].0, smp * DT, tot(&mc), tot(&mg));
        }
    }
    println!("  true rule held-out RMSE: CPU {:.4}  GPU {:.4}", test_metric(&exp, &data, &tr), gpu_test(&exp, &data, g, &tcmp));
}

/// `ca-lab bench <exp>`: time one generation's training evaluation with the
/// original simulation path vs the fast kernel, and check they agree.
fn bench(id: &str) {
    let exp = experiment(id);
    let k = exp.states.len();
    let data = build_data(&exp);
    let mut rng = Rng(12345);
    let rows = k * exp.n_terrain * 9;
    let pop: Vec<Compiled> = (0..exp.pop)
        .map(|_| {
            let mut g: Vec<f32> = (0..rows * k).map(|_| (rng.normal() * 1.5) as f32).collect();
            for r in 0..rows {
                g[r * k + r / (exp.n_terrain * 9)] += exp.stay_bias;
            }
            Compiled::from_exp(&g, &exp)
        })
        .collect();
    let time = |name: &str, f: &dyn Fn() -> f64| -> f64 {
        let t = Instant::now();
        let v = f();
        let s = t.elapsed().as_secs_f64();
        println!("  {name:<44} {s:8.3}s   (checksum {v:.4})");
        s
    };
    println!("[{id}] one generation = {} rules x {} outbreaks x {} runs x {} steps", exp.pop, exp.n_train, exp.rollouts, exp.steps);
    let old_1 = time("original path, 1 thread, 4 rules", &|| pop[..4].iter().map(|c| train_err_stoch(&exp, &data, &|s, n, t, u| c.sample(s, n, t, u), 0)).sum());
    let new_1 = time("fast kernel,   1 thread, 4 rules", &|| pop[..4].iter().map(|c| train_err_fast(&exp, &data, &thresholds(c), 0)).sum());
    let old_p = time("original path, all threads, full generation", &|| pop.par_iter().map(|c| train_err_stoch(&exp, &data, &|s, n, t, u| c.sample(s, n, t, u), 0)).sum());
    let new_p = time("fast kernel,   all threads, full generation", &|| pop.par_iter().map(|c| train_err_fast(&exp, &data, &thresholds(c), 0)).sum());
    println!("  speed-up: {:.1}x single-thread, {:.1}x full generation", old_1 / new_1, old_p / new_p);
    // agreement: the true rule through both paths should give the same training error on average
    let tp = { let tr = exp.truth; rule_probs(k, exp.n_terrain, &tr) };
    let tg: Vec<f32> = tp.iter().map(|&q| if q <= 0.0 { -30.0 } else { q.ln() as f32 }).collect();
    let tc = Compiled::new(&tg, k, exp.n_terrain, None);
    let th = thresholds(&tc);
    let tr = exp.truth;
    let a: Vec<f64> = (0..24).into_par_iter().map(|g| train_err_stoch(&exp, &data, &tr, 7_000_000 + g)).collect();
    let b: Vec<f64> = (0..24).into_par_iter().map(|g| train_err_fast(&exp, &data, &th, 7_000_000 + g)).collect();
    let ms = |v: &[f64]| { let m = v.iter().sum::<f64>() / v.len() as f64; (m, (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()) };
    let (ma, sa) = ms(&a);
    let (mb, sb) = ms(&b);
    println!("  agreement (true rule, 24 seeds): original {ma:.4} ± {sa:.4}   fast {mb:.4} ± {sb:.4}");
    // finer check: average district series (sick share, immune share) over many runs
    let s0 = &data.train[0];
    let f0 = &data.train_fast[0];
    let ra: Vec<Vec<f64>> = (0..200).into_par_iter().map(|r| rollout(&exp, s0, &tr, 900_000 + r, Some(&[])).district).collect();
    let rb: Vec<Vec<f64>> = (0..200).into_par_iter().map(|r| fast_district(&exp, f0, &th, 900_000 + r)).collect();
    let (ma, mb) = (mean_vec(&ra), mean_vec(&rb));
    let tl = exp.tracked.len();
    for (j, &s) in exp.tracked.iter().enumerate() {
        let tot = |m: &[f64], smp: usize| (0..ND).map(|d| m[smp * ND * tl + d * tl + j]).sum::<f64>() / ND as f64;
        let last = exp.steps / DT;
        println!("  mean {} share at t=40 / t=end: original {:.4} / {:.4}   fast {:.4} / {:.4}", exp.states[s].0, tot(&ma, 10), tot(&ma, last), tot(&mb, 10), tot(&mb, last));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let force_cpu = args.iter().any(|a| a == "--cpu");
    let g = if force_cpu || args.get(1).map(String::as_str) == Some("ood") { None } else { gpu::Gpu::new() };
    match &g {
        Some(g) => eprintln!("[ca-lab] GPU backend: {}", g.name),
        None => eprintln!("[ca-lab] CPU backend{}", if force_cpu { " (--cpu)" } else { "" }),
    }
    let _ = GPU.set(g);
    if args.get(1).map(String::as_str) == Some("gpubench") {
        gpubench(args.get(2).map(String::as_str).unwrap_or("exp3"));
        return;
    }
    if args.get(1).map(String::as_str) == Some("bench") {
        bench(args.get(2).map(String::as_str).unwrap_or("exp3"));
        return;
    }
    if args.get(1).map(String::as_str) == Some("ood") {
        let id = args.get(2).expect("usage: ca-lab ood <exp>");
        let dir = PathBuf::from(args.get(3).cloned().unwrap_or_else(|| "site/data".into()));
        run_ood(id, &dir);
        return;
    }
    let id = args.get(1).map(String::as_str).unwrap_or("exp1");
    let dir = PathBuf::from(args.get(2).filter(|a| !a.starts_with("--")).cloned().unwrap_or_else(|| "site/data".into()));
    fs::create_dir_all(&dir).unwrap();
    let flag = |name: &str| args.iter().position(|a| a == name);
    let resume = flag("--resume").is_some();
    let until_converged = flag("--until-converged").is_some();
    let no_screen = flag("--no-screen").is_some();
    let exp = experiment(id);
    let max_gens: usize = flag("--gens").and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(exp.gens);
    let ckpt_dir = PathBuf::from("checkpoints");
    fs::create_dir_all(&ckpt_dir).unwrap();
    let ckpt_path = ckpt_dir.join(format!("{}.json", exp.id));
    let out = Writer { dir: dir.clone() };
    let t0 = Instant::now();
    let k = exp.states.len();
    let rows = k * exp.n_terrain * 9;
    let len = rows * k;
    let frame_times: Vec<usize> = (0..=exp.steps).step_by((exp.steps / 15).max(1)).collect();

    println!("[{}] building data ({} train / {} test scenarios)", exp.id, exp.n_train, exp.n_test);
    let data = build_data(&exp);

    // baselines
    let mut baselines: Vec<(String, f64)> = vec![];
    let mut wm: Option<(f64, f64)> = None;
    for (name, r) in &exp.det_baselines {
        baselines.push((name.to_string(), test_metric(&exp, &data, r)));
    }
    if exp.stochastic {
        let (beta, gamma) = fit_well_mixed(&exp, &data);
        let e: f64 = data.test.iter().enumerate().map(|(i, s)| rmse(&sir_district_series(s, beta, gamma, exp.steps), &data.test_truth_mean[i])).sum::<f64>() / data.test.len() as f64;
        baselines.push((format!("Classic SIR, fitted (β={beta:.2}, γ={gamma:.3})"), e));
        let truth = exp.truth;
        baselines.push(("True rule (best possible)".into(), test_metric(&exp, &data, &truth)));
        wm = Some((beta, gamma));
    }
    for (n, v) in &baselines {
        println!("  baseline {n}: {v:.4}");
    }

    // national-curve helper for the visual panel (test scenario 0)
    let scn0 = &data.test[0];
    let tracked_names: Vec<&str> = exp.tracked.iter().map(|&s| exp.states[s].0).collect();
    let truth_curves: Vec<Vec<f64>> = if exp.stochastic {
        let nat = &data.test_truth_nat[0];
        (0..exp.tracked.len()).map(|j| (0..=exp.steps).map(|t| nat[t * exp.tracked.len() + j]).collect()).collect()
    } else {
        let tr = exp.truth;
        let nat = rollout(&exp, scn0, &tr, 0, Some(&[])).national;
        (0..exp.tracked.len()).map(|j| (0..=exp.steps).map(|t| nat[t * exp.tracked.len() + j]).collect()).collect()
    };
    let truth_frames: Vec<String> = {
        let tr = exp.truth;
        rollout(&exp, scn0, &tr, 4242, Some(&frame_times)).frames.iter().map(|g| grid_str(g)).collect()
    };
    let wm_curve: Option<Vec<f64>> = wm.map(|(b, g)| {
        let pop = populated(&scn0.terrain);
        let np = pop.iter().filter(|&&p| p).count() as f64;
        let i0 = (0..N).filter(|&i| pop[i] && scn0.init[i] == 1).count() as f64 / np;
        sir_series(1.0 - i0, i0, 0.0, b, g, exp.steps).iter().map(|x| x.0).collect()
    });

    let make_snap = |c: &Compiled, gen: usize, test: f64| -> Snap {
        let (frames, nat) = if exp.stochastic {
            let rule = |s, n, t, u| c.sample(s, n, t, u);
            let fr = rollout(&exp, scn0, &rule, 999, Some(&frame_times)).frames;
            let runs: Vec<Vec<f64>> = (0..TEST_ENS).into_par_iter().map(|r| rollout(&exp, scn0, &rule, 31_337 + r as u64, Some(&[])).national).collect();
            (fr, mean_vec(&runs))
        } else {
            let rule = |s, n, t, _u| c.det(s, n, t);
            let tr = rollout(&exp, scn0, &rule, 0, Some(&frame_times));
            (tr.frames, tr.national)
        };
        let mut curves = Map::new();
        for (j, nm) in tracked_names.iter().enumerate() {
            curves.insert(format!("truth · {nm}"), json!(truth_curves[j]));
            let mc: Vec<f64> = (0..=exp.steps).map(|t| nat[t * exp.tracked.len() + j]).collect();
            curves.insert(format!("model · {nm}"), json!(mc));
        }
        if let Some(w) = &wm_curve {
            curves.insert("classic SIR · Infectious".into(), json!(w));
        }
        Snap { gen, test, truth: truth_frames.clone(), model: frames.iter().map(|g| grid_str(g)).collect(), curves }
    };

    // ---------------- genetic algorithm
    let mut rng = Rng(0xC0FFEE ^ mix(exp.id.len() as u64));
    let mut pop: Vec<Vec<f32>> = (0..exp.pop)
        .map(|_| {
            let mut g: Vec<f32> = (0..len).map(|_| (rng.normal() * 1.5) as f32).collect();
            if exp.stay_bias > 0.0 {
                for r in 0..rows {
                    let s = r / (exp.n_terrain * 9);
                    g[r * k + s] += exp.stay_bias;
                }
            }
            g
        })
        .collect();
    let mut history: Vec<Hist> = vec![];
    let mut snaps: Vec<Snap> = vec![];
    let mut best_test_seen = f64::NAN;
    let mut start_gen = 0usize;
    if resume {
        let ck: Value = serde_json::from_slice(&fs::read(&ckpt_path).expect("no checkpoint to resume from")).unwrap();
        pop = serde_json::from_value(ck["pop"].clone()).unwrap();
        history = serde_json::from_value(ck["history"].clone()).unwrap();
        rng.0 = ck["rng"].as_u64().unwrap();
        best_test_seen = ck["best_test_seen"].as_f64().unwrap_or(f64::NAN);
        start_gen = ck["next_gen"].as_u64().unwrap() as usize;
        // bring back the generation snapshots already shown on the site
        if let Ok(b) = fs::read(dir.join(format!("{}.json", exp.id))) {
            if let Ok(prev) = serde_json::from_slice::<Value>(&b) {
                for s in prev["snapshots"].as_array().cloned().unwrap_or_default() {
                    snaps.push(Snap {
                        gen: s["gen"].as_u64().unwrap() as usize,
                        test: s["test"].as_f64().unwrap(),
                        truth: serde_json::from_value(s["truth"].clone()).unwrap(),
                        model: serde_json::from_value(s["model"].clone()).unwrap(),
                        curves: s["curves"].as_object().cloned().unwrap(),
                    });
                }
            }
        }
        println!("[{}] resumed at generation {} ({} snapshots)", exp.id, start_gen, snaps.len());
    }
    // Noise floor: what the TRUE rule scores on the training data with the
    // same noisy procedure. The GA cannot meaningfully go below this.
    let noise_floor = if exp.stochastic {
        let tr = exp.truth;
        let v: Vec<f64> = (0..12).into_par_iter().map(|g| train_err_stoch(&exp, &data, &tr, 7_000_000 + g)).collect();
        let m = v.iter().sum::<f64>() / v.len() as f64;
        println!("  noise floor (true rule on training data): {m:.4}");
        Some(m)
    } else {
        None
    };
    let better = |a: f64, b: f64| if exp.stochastic { a < b } else { a > b };
    let sigma: f64 = if exp.stochastic { 0.6 } else { 1.5 };
    let pm = (3.0 / len as f64).max(0.01);
    let elite = 4;

    let truth_probs = { let tr = exp.truth; rule_probs(k, exp.n_terrain, &tr) };
    let use_counts = usage(&exp, &data);
    let terrain_names: Vec<&str> = if exp.n_terrain == 3 { vec!["countryside", "town", "water"] } else { vec!["everywhere"] };
    let rule_json = |c: &Compiled| -> Value {
        let learned = if exp.stochastic {
            rule_probs(k, exp.n_terrain, &|s, n, t, u| c.sample(s, n, t, u))
        } else {
            rule_probs(k, exp.n_terrain, &|s, n, t, _u| c.det(s, n, t))
        };
        json!({"k": k, "nt": exp.n_terrain, "terrain": terrain_names, "learned": learned, "truth": truth_probs, "usage": use_counts})
    };

    let write_all = |history: &[Hist], snaps: &[Snap], status: &str, best: Option<&Hist>, rule: &Value| {
        let snaps_out: Vec<Value> = thin(snaps, 14)
            .iter()
            .map(|s| json!({"gen": s.gen, "test": s.test, "truth": s.truth, "model": s.model, "curves": s.curves}))
            .collect();
        let (lo, hi) = if exp.stochastic { (Some(0.0), None) } else { (Some(0.0), Some(1.0)) };
        let v = json!({
            "id": exp.id, "title": exp.title, "status": status, "blurb": exp.blurb,
            "width": W, "height": H,
            "states": exp.states.iter().map(|(n, c)| json!({"name": n, "color": c})).collect::<Vec<_>>(),
            "metric_name": if exp.stochastic { "district RMSE" } else { "active-cell accuracy" },
            "higher_is_better": !exp.stochastic,
            "metric_lo": lo, "metric_hi": hi,
            "baselines": baselines.iter().map(|(n, v)| json!({"name": n, "value": v})).collect::<Vec<_>>(),
            "history": history, "snapshots": snaps_out, "frame_times": frame_times,
            "curve_label": "Share of population over time (held-out scenario)",
            "elapsed_s": t0.elapsed().as_secs_f64(),
            "rule": rule,
            "wm": wm,
            "notes": ""
        });
        out.write_atomic(&format!("{}.json", exp.id), &v);
        let summary = match best {
            Some(h) => format!("gen {} · {:.3}", h.gen, h.test),
            None => exp.summary.to_string(),
        };
        out.update_manifest(&exp, status, &format!("{}-{}", status, history.len()), &summary);
    };
    write_all(&history, &snaps, "running", None, &Value::Null);
    let mut last_rule = Value::Null;

    let snap_every = (max_gens / 60).max(10);
    let mut converged_note: Option<String> = None;
    for gen in start_gen..max_gens {
        let compiled: Vec<Compiled> = pop.iter().map(|g| Compiled::from_exp(g, &exp)).collect();
        let scores: Vec<f64> = if let (Some(g), true) = (the_gpu(), data.gpu_train.is_some()) {
            gpu_scores(&exp, &data, g, &compiled, gen)
        } else if exp.stochastic && !no_screen {
            // screen everyone with 1 run per outbreak, then give the top quarter the full evaluation
            let ths: Vec<Vec<u32>> = compiled.iter().map(thresholds).collect();
            let mut sc: Vec<f64> = compiled.par_iter().zip(ths.par_iter()).map(|(c, th)| -train_err_fast_n(&exp, &data, th, gen, 1) - exp.smooth * c.roughness()).collect();
            let mut idx: Vec<usize> = (0..sc.len()).collect();
            idx.sort_by(|&a, &b| sc[b].partial_cmp(&sc[a]).unwrap());
            let top: Vec<usize> = idx[..(exp.pop / 4).max(elite)].to_vec();
            let full: Vec<f64> = top.par_iter().map(|&i| train_score(&exp, &data, &compiled[i], gen)).collect();
            // keep screened-out candidates strictly below every fully evaluated one
            let floor = full.iter().cloned().fold(f64::MAX, f64::min);
            for v in sc.iter_mut() {
                *v = v.min(floor) - 1.0;
            }
            for (j, &i) in top.iter().enumerate() {
                sc[i] = full[j];
            }
            sc
        } else {
            compiled.par_iter().map(|c| train_score(&exp, &data, c, gen)).collect()
        };
        let mut order: Vec<usize> = (0..pop.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        let bc = &compiled[order[0]];
        let test = if let (Some(g), true) = (the_gpu(), data.gpu_test.is_some()) {
            gpu_test(&exp, &data, g, bc)
        } else if exp.stochastic {
            test_metric_fast(&exp, &data, &thresholds(bc))
        } else {
            let r = |s, n, t, _u| bc.det(s, n, t);
            test_metric(&exp, &data, &r)
        };
        let train_val = if exp.stochastic { -scores[order[0]] } else { scores[order[0]] };
        history.push(Hist { gen, train: train_val, test });
        // Converged = the last 25 generations' training error averages within
        // 3% of the true rule's own training error (the noise floor).
        if let (true, Some(f)) = (until_converged, noise_floor) {
            if history.len() >= 25 {
                let w: f64 = history[history.len() - 25..].iter().map(|h| h.train).sum::<f64>() / 25.0;
                if w <= f * 1.03 {
                    converged_note = Some(format!("converged at generation {gen}: 25-generation mean training error {w:.4} vs noise floor {f:.4}"));
                }
            }
        }
        let last = gen == max_gens - 1 || converged_note.is_some();
        let improved = best_test_seen.is_nan() || better(test, best_test_seen) && (test - best_test_seen).abs() > 1e-4;
        if improved || last || gen % snap_every == 0 {
            if improved {
                best_test_seen = test;
            }
            snaps.push(make_snap(bc, gen, test));
        }
        println!("[{}] gen {:3}  train {:.4}  held-out {:.4}  ({:.1}s)", exp.id, gen, train_val, test, t0.elapsed().as_secs_f64());
        if gen % 3 == 0 || improved || last {
            last_rule = rule_json(bc);
            last_rule["genes"] = json!(pop[order[0]]);
            write_all(&history, &snaps, "running", history.last(), &last_rule);
        }
        if last {
            if let Some(n) = &converged_note {
                println!("[{}] {n}", exp.id);
            }
            break;
        }

        // next generation
        let mut next: Vec<Vec<f32>> = order[..elite].iter().map(|&i| pop[i].clone()).collect();
        let tour = |rng: &mut Rng| -> usize {
            let mut b = rng.below(pop.len());
            for _ in 0..2 {
                let c = rng.below(pop.len());
                if scores[c] > scores[b] {
                    b = c;
                }
            }
            b
        };
        while next.len() < exp.pop {
            let (a, b) = (tour(&mut rng), tour(&mut rng));
            let mut child = pop[a].clone();
            for r in 0..rows {
                if rng.f() < 0.5 {
                    child[r * k..(r + 1) * k].copy_from_slice(&pop[b][r * k..(r + 1) * k]);
                }
            }
            // multi-scale mutation: most children take small steps, some big jumps
            let scale = [0.3, 1.0, 1.0, 3.0][rng.below(4)];
            for v in child.iter_mut() {
                if rng.f() < pm {
                    *v = (*v as f64 + rng.normal() * sigma * scale).clamp(-10.0, 10.0) as f32;
                }
            }
            if rng.f() < 0.15 {
                let r = rng.below(rows);
                for j in 0..k {
                    child[r * k + j] = (rng.normal() * 2.0) as f32;
                }
            }
            next.push(child);
        }
        pop = next;
        if gen % 10 == 0 {
            let ck = json!({"next_gen": gen + 1, "pop": pop, "history": history, "rng": rng.0, "best_test_seen": best_test_seen});
            let tmp = ckpt_path.with_extension("tmp");
            fs::write(&tmp, serde_json::to_vec(&ck).unwrap()).unwrap();
            fs::rename(&tmp, &ckpt_path).unwrap();
        }
    }
    let best = history.iter().cloned().reduce(|a, b| if better(b.test, a.test) { b } else { a });
    write_all(&history, &snaps, "done", history.last(), &last_rule);
    if let Some(b) = best {
        println!("[{}] done in {:.1}s. final held-out {:.4} (best seen {:.4} at gen {})", exp.id, t0.elapsed().as_secs_f64(), history.last().unwrap().test, b.test, b.gen);
    }
    let _ = Path::new(".");
}
