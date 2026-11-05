//! GPU backend (wgpu: Metal on macOS, Vulkan/DX12 elsewhere).
//!
//! Runs a whole batch of simulations at once — every (candidate rule,
//! scenario, seed) job of a generation — one GPU thread per cell, and reduces
//! the results on the GPU (district counts for random problems, accuracy
//! tallies for deterministic ones). Only those small totals come back.
//!
//! General by construction: states, terrain types, grid size, steps, and the
//! rule table are all data. Limits: k <= 8 states (lookup-table size).

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

pub const NONE: u32 = u32::MAX;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct P {
    n_cells: u32, width: u32, height: u32, k: u32,
    nt: u32, rows: u32, t: u32, slot: u32,
    n_sims: u32, nd: u32, tl: u32, n_samples: u32,
    mode: u32, steps1: u32, n_slots: u32, n_words: u32,
}

/// Static description of a problem (shared by every batch).
#[derive(Clone)]
pub struct Spec {
    pub width: u32,
    pub height: u32,
    pub k: u32,
    pub nt: u32,
    pub steps: u32,
    pub dt: u32,
    pub nd: u32,
    pub infectious: Vec<bool>,
    pub tracked: Vec<usize>,
    pub quiet: Vec<bool>,
}

impl Spec {
    /// The packed kernels need the width to be a multiple of 4 and <= 8 states.
    pub fn gpu_ok(&self) -> bool { self.width % 4 == 0 && self.k <= 8 && self.nd as usize * self.tracked.len() <= 1024 }
    fn n_words(&self) -> u32 { self.n_cells() / 4 }
    fn n_cells(&self) -> u32 { self.width * self.height }
    fn rows(&self) -> u32 { self.k * self.nt * 9 }
    fn lut(&self) -> Vec<u32> {
        let mut l = vec![0u32; 24];
        for (s, &b) in self.infectious.iter().enumerate() { l[s] = b as u32; }
        for s in 0..8 { l[8 + s] = NONE; }
        for (j, &s) in self.tracked.iter().enumerate() { l[8 + s] = j as u32; }
        for (s, &b) in self.quiet.iter().enumerate() { l[16 + s] = b as u32; }
        l
    }
}

/// Scenarios uploaded once and reused by every batch.
pub struct SceneSet {
    terrain: wgpu::Buffer,
    init: wgpu::Buffer,
    dmap: wgpu::Buffer,
    frames: wgpu::Buffer,
    pub has_frames: bool,
}

pub struct Out {
    pub counts: Vec<u32>, // sims * samples * nd * tl
    pub acc: Vec<u32>,    // sims * n_slots * 2
    pub n_samples: usize,
    pub n_slots: usize,
}

pub struct Gpu {
    dev: wgpu::Device,
    q: wgpu::Queue,
    bgl: wgpu::BindGroupLayout,
    p_init: wgpu::ComputePipeline,
    p_step: wgpu::ComputePipeline,
    p_obs: wgpu::ComputePipeline,
    p_acc: wgpu::ComputePipeline,
    align: u64,
    max_bind: u64,
    pub name: String,
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

impl Gpu {
    pub fn new() -> Option<Gpu> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let (dev, q) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ca-lab"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .ok()?;
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ca-lab kernels"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: None },
            count: None,
        }];
        for (b, ro) in [(1, true), (2, false), (3, true), (4, true), (5, true), (6, true), (7, false), (8, true), (9, true), (10, true), (11, false)] {
            entries.push(storage(b, ro));
        }
        let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("ca-lab bgl"), entries: &entries });
        let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ca-lab layout"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let mk = |entry: &str| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (p_init, p_step, p_obs, p_acc) = (mk("k_init"), mk("k_step"), mk("k_observe"), mk("k_acc"));
        Some(Gpu {
            align: limits.min_uniform_buffer_offset_alignment as u64,
            max_bind: (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size),
            dev,
            q,
            bgl,
            p_init,
            p_step,
            p_obs,
            p_acc,
            name: format!("{} ({:?})", info.name, info.backend),
        })
    }

    fn buf(&self, label: &str, data: &[u32], extra: wgpu::BufferUsages) -> wgpu::Buffer {
        let data: &[u32] = if data.is_empty() { &[0] } else { data };
        self.dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | extra,
        })
    }

    fn empty(&self, label: &str, words: u64, extra: wgpu::BufferUsages) -> wgpu::Buffer {
        self.dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: words.max(1) * 4,
            usage: wgpu::BufferUsages::STORAGE | extra,
            mapped_at_creation: false,
        })
    }

    /// Upload scenarios: per scenario its starting grid, terrain, district map
    /// (NONE = not counted) and optionally all ground-truth frames.
    pub fn scenes(&self, init: &[Vec<u8>], terrain: &[Vec<u8>], dmap: &[Vec<u32>], frames: Option<&[Vec<Vec<u8>>]>) -> SceneSet {
        let pack = |g: &[u8]| g.chunks(4).map(|c| c.iter().enumerate().fold(0u32, |a, (i, &b)| a | (b as u32) << (8 * i))).collect::<Vec<u32>>();
        let cat = |v: &[Vec<u8>]| v.iter().flat_map(|g| pack(g)).collect::<Vec<u32>>();
        let fr: Vec<u32> = frames.map(|f| f.iter().flat_map(|s| s.iter().flat_map(|g| pack(g))).collect()).unwrap_or_default();
        SceneSet {
            terrain: self.buf("terrain", &cat(terrain), wgpu::BufferUsages::empty()),
            init: self.buf("init", &cat(init), wgpu::BufferUsages::empty()),
            dmap: self.buf("dmap", &dmap.concat(), wgpu::BufferUsages::empty()),
            frames: self.buf("frames", &fr, wgpu::BufferUsages::empty()),
            has_frames: frames.is_some(),
        }
    }

    /// Largest number of simulations one batch can hold.
    fn max_sims(&self, spec: &Spec) -> usize {
        ((self.max_bind / (spec.n_words() as u64 * 4)) as usize).clamp(1, 65_535)
    }

    /// Run simulations. `jobs` = (rule index, scenario index, seed).
    /// `th` = thresholds for every rule: rules * rows * (k-1).
    /// district: count tracked states per district every `dt` steps.
    /// acc: accuracy vs ground-truth frames at every step, plus (if
    /// onestep > 0) one-step accuracy from the truth frame every `onestep` steps.
    pub fn run(&self, spec: &Spec, set: &SceneSet, th: &[u32], jobs: &[(u32, u32, u64)], district: bool, acc: bool, onestep: u32) -> Out {
        let chunk = self.max_sims(spec);
        let mut out = Out { counts: vec![], acc: vec![], n_samples: 0, n_slots: 0 };
        for part in jobs.chunks(chunk) {
            let o = self.run_chunk(spec, set, th, part, district, acc, onestep);
            out.counts.extend(o.counts);
            out.acc.extend(o.acc);
            out.n_samples = o.n_samples;
            out.n_slots = o.n_slots;
        }
        out
    }

    fn run_chunk(&self, spec: &Spec, set: &SceneSet, th: &[u32], jobs: &[(u32, u32, u64)], district: bool, acc: bool, onestep: u32) -> Out {
        assert!(spec.gpu_ok(), "problem not supported by the GPU kernels (needs width % 4 == 0, <= 8 states)");
        assert!(!acc || set.has_frames, "accuracy needs ground-truth frames");
        let prof = std::env::var("CA_PROFILE").is_ok();
        let t0 = std::time::Instant::now();
        let n = spec.n_cells();
        let sims = jobs.len() as u32;
        let steps1 = spec.steps + 1;
        let n_samples = if district { (spec.steps / spec.dt + 1) as usize } else { 0 };
        let n_slots = if acc { 2 * steps1 as usize } else { 0 };
        let tl = spec.tracked.len() as u32;
        let jw: Vec<u32> = jobs.iter().flat_map(|&(r, s, seed)| [r, s, seed as u32, (seed >> 32) as u32]).collect();
        let b_th = self.buf("th", th, wgpu::BufferUsages::empty());
        let b_jobs = self.buf("jobs", &jw, wgpu::BufferUsages::empty());
        let b_lut = self.buf("lut", &spec.lut(), wgpu::BufferUsages::empty());
        let nw = spec.n_words();
        let cells = sims as u64 * nw as u64;
        let ga = self.empty("grid A", cells, wgpu::BufferUsages::empty());
        let gb = self.empty("grid B", cells, wgpu::BufferUsages::empty());
        let gc = self.empty("grid C", if acc && onestep > 0 { cells } else { 1 }, wgpu::BufferUsages::empty());
        let cwords = sims as u64 * n_samples as u64 * spec.nd as u64 * tl as u64;
        let awords = sims as u64 * n_slots as u64 * 2;
        let b_counts = self.empty("counts", cwords, wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
        let b_acc = self.empty("acc", awords, wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);

        // every dispatch gets its own uniform slot (dynamic offset)
        let base = P { n_cells: n, width: spec.width, height: spec.height, k: spec.k, nt: spec.nt, rows: spec.rows(), n_sims: sims, nd: spec.nd, tl, n_samples: n_samples as u32, steps1, n_slots: n_slots as u32, n_words: nw, ..Default::default() };
        #[derive(Clone, Copy)]
        enum K { Init, Step, Obs, Acc }
        #[derive(Clone, Copy, PartialEq)]
        enum G { AB, BA, AC, CA }
        let mut cmds: Vec<(K, G, P)> = vec![(K::Init, G::BA, base)];
        let mut cur_a = true;
        let src_of = |a: bool| if a { G::AB } else { G::BA };
        for t in 0..=spec.steps {
            if district && t % spec.dt == 0 {
                cmds.push((K::Obs, src_of(cur_a), P { slot: t / spec.dt, ..base }));
            }
            if acc {
                cmds.push((K::Acc, src_of(cur_a), P { t, slot: t, ..base }));
            }
            if t == spec.steps {
                break;
            }
            if acc && onestep > 0 && t % onestep == 0 {
                cmds.push((K::Step, G::AC, P { t, mode: 1, ..base }));
                cmds.push((K::Acc, G::CA, P { t: t + 1, slot: steps1 + t, ..base }));
            }
            cmds.push((K::Step, src_of(cur_a), P { t, ..base }));
            cur_a = !cur_a;
        }
        let stride = (std::mem::size_of::<P>() as u64).div_ceil(self.align) * self.align;
        let mut ubytes = vec![0u8; stride as usize * cmds.len()];
        for (i, (_, _, pp)) in cmds.iter().enumerate() {
            let o = i * stride as usize;
            ubytes[o..o + std::mem::size_of::<P>()].copy_from_slice(bytemuck::bytes_of(pp));
        }
        let b_uni = self.dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("uniforms"), contents: &ubytes, usage: wgpu::BufferUsages::UNIFORM });

        let bg = |s: &wgpu::Buffer, d: &wgpu::Buffer| {
            fn e(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> { wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() } }
            self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &b_uni, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<P>() as u64) }) },
                    e(1, s), e(2, d), e(3, &set.terrain), e(4, &b_th), e(5, &b_jobs), e(6, &b_lut),
                    e(7, &b_counts), e(8, &set.dmap), e(9, &set.init), e(10, &set.frames), e(11, &b_acc),
                ],
            })
        };
        let groups = [bg(&ga, &gb), bg(&gb, &ga), bg(&ga, &gc), bg(&gc, &ga)];
        let gi = |g: G| match g { G::AB => 0, G::BA => 1, G::AC => 2, G::CA => 3 };

        let mut enc = self.dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("batch") });
        enc.clear_buffer(&b_counts, 0, None);
        enc.clear_buffer(&b_acc, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sim"), timestamp_writes: None });
            let wx = nw.div_ceil(256);
            for (i, (k, g, _)) in cmds.iter().enumerate() {
                pass.set_pipeline(match k { K::Init => &self.p_init, K::Step => &self.p_step, K::Obs => &self.p_obs, K::Acc => &self.p_acc });
                pass.set_bind_group(0, &groups[gi(*g)], &[(i as u64 * stride) as u32]);
                pass.dispatch_workgroups(wx, sims, 1);
            }
        }
        let rd = |b: &wgpu::Buffer, words: u64, enc: &mut wgpu::CommandEncoder| -> Option<wgpu::Buffer> {
            if words == 0 {
                return None;
            }
            let st = self.dev.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: words * 4, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            enc.copy_buffer_to_buffer(b, 0, &st, 0, words * 4);
            Some(st)
        };
        let s_counts = rd(&b_counts, cwords, &mut enc);
        let s_acc = rd(&b_acc, awords, &mut enc);
        let t_setup = t0.elapsed().as_secs_f64();
        self.q.submit(Some(enc.finish()));
        if prof {
            self.dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            eprintln!("    gpu: setup {:.1} ms, kernels {:.1} ms ({} dispatches, {} sims)", t_setup * 1e3, (t0.elapsed().as_secs_f64() - t_setup) * 1e3, cmds.len(), sims);
        }
        let t_rd = std::time::Instant::now();
        let read = |b: Option<wgpu::Buffer>| -> Vec<u32> {
            match b {
                None => vec![],
                Some(b) => {
                    b.map_async(wgpu::MapMode::Read, .., |r| r.expect("map failed"));
                    self.dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("poll failed");
                    let v = bytemuck::cast_slice::<u8, u32>(&b.get_mapped_range(..).expect("mapped range")).to_vec();
                    b.unmap();
                    v
                }
            }
        };
        let o = Out { counts: read(s_counts), acc: read(s_acc), n_samples, n_slots };
        if prof {
            eprintln!("    gpu: readback {:.1} ms", t_rd.elapsed().as_secs_f64() * 1e3);
        }
        o
    }
}
