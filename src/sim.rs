use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use rand::distributions::{Distribution, Uniform};
use rmds::{Buffer, Engine, Shader};
use std::fs::read;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct Boid {
    pub pos: [f32; 3],
    pub level: u32,
    pub heading: [f32; 3],
    pub mask: u32,
}

unsafe impl Zeroable for Boid {}
unsafe impl Pod for Boid {}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct AccumulatorHalf {
    pub pos: [f32; 3],
    pub count: u32,
    pub heading: [f32; 3],
    pub _filler: u32,
}

unsafe impl Zeroable for AccumulatorHalf {}
unsafe impl Pod for AccumulatorHalf {}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct Accumulator {
    pub left: AccumulatorHalf,
    pub right: AccumulatorHalf,
}

unsafe impl Zeroable for Accumulator {}
unsafe impl Pod for Accumulator {}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct SelectParams {
    pub plane_pos: [f32; 3],
    pub mask: u32,
    pub plane_normal: [f32; 3],
    pub level: u32,
}

unsafe impl Zeroable for SelectParams {}
unsafe impl Pod for SelectParams {}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct MotionParams {
    pub n_groups: u32,
    pub speed: f32,
    pub dist_thresh: f32,
    pub cohere: f32,
    pub steer: f32,
    pub parallel: f32,
}

unsafe impl Zeroable for MotionParams {}
unsafe impl Pod for MotionParams {}


#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct Group {
    pub center: [f32; 3],
    pub _filler0: u32,
    pub heading: [f32; 3],
    pub _filler1: u32,
}

unsafe impl Zeroable for Group {}
unsafe impl Pod for Group {}

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct Settings {
    pub work_groups: u32,
    pub speed: f32,
    pub dist_thresh: f32,
    pub cohere: f32,
    pub steer: f32,
    pub parallel: f32,
    pub tree_depth: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            work_groups: 40,
            tree_depth: 6,
            speed: 0.04,
            dist_thresh: 8.,
            cohere: 0.5,
            steer: 0.12,
            parallel: 0.12,
        }
    }
}

fn motion_params_from_settings(settings: &Settings) -> MotionParams {
    MotionParams {
        n_groups: LOCAL_X * settings.work_groups,
        dist_thresh: settings.dist_thresh,
        cohere: settings.cohere,
        speed: settings.speed,
        steer: settings.steer,
        parallel: settings.parallel,
    }
}

pub struct Simulation {
    engine: Engine,
    boids: Vec<Boid>,
    settings: Settings,
    n_boids: u32,
    boids_gpu: Buffer,
    groups_gpu: Buffer,
    acc_gpu: Buffer,
    setup: Shader,
    reduce: Shader,
    motion: Shader,
    select: Shader,
    boids_dirty: bool,
}

pub const LOCAL_X: u32 = 16;
impl Simulation {
    pub fn new(settings: Settings) -> Result<Self> {
        assert!(settings.work_groups > 0);
        assert!(settings.tree_depth > 0);
        let mut engine = rmds::Engine::new(true)?;
        let n_boids = settings.work_groups * LOCAL_X;

        let setup = engine.spirv(&read("kernels/setup.comp.spv")?)?;
        let reduce = engine.spirv(&read("kernels/reduce.comp.spv")?)?;
        let motion = engine.spirv(&read("kernels/motion.comp.spv")?)?;
        let select = engine.spirv(&read("kernels/select.comp.spv")?)?;

        let acc_gpu = engine.buffer::<Accumulator>(n_boids as _)?;
        let boids_gpu = engine.buffer::<Boid>(n_boids as _)?;
        let groups_gpu = engine.buffer::<Group>(1 << settings.tree_depth)?;

        let boids = random_boids(n_boids as _, 10.);
        engine.write(boids_gpu, &boids)?;

        Ok(Self {
            n_boids,
            groups_gpu,
            settings,
            acc_gpu,
            boids_gpu,
            engine,
            setup,
            reduce,
            boids,
            motion,
            select,
            boids_dirty: false,
        })
    }

    pub fn boids(&mut self) -> Result<&[Boid]> {
        if self.boids_dirty {
            self.engine.read(self.boids_gpu, &mut self.boids)?;
            self.boids_dirty = false;
        }
        Ok(&self.boids)
    }

    pub fn step(&mut self) -> Result<Vec<Group>> {
        // Setup
        self.boids_dirty = true;
        self.engine.run(
            self.setup,
            self.acc_gpu,
            self.boids_gpu,
            self.settings.work_groups,
            1,
            1,
            &[],
        )?;
        let acc = self.reduce()?;
        let mut partitions = vec![acc_to_group(acc.left)];

        // Build acceleration tree
        let mut total = 0;
        // Tree depth
        for level in 0..self.settings.tree_depth {
            // Mask for each leaf node
            //let mut level_count = 0;
            for mask in 0..(1 << level) {
                // Parent node idx
                let plane_idx = total; 

                /*eprintln!(
                    "Level: {}, Mask: {:b}, Plane idx: {}",
                    level, mask, plane_idx
                );*/

                if let Some(plane) = partitions[plane_idx as usize] {
                    self.select(level, mask, plane)?;
                    let acc = self.reduce()?;
                    //level_count += dbg!(acc.left.count) + dbg!(acc.right.count);
                    //level_count += acc.left.count + acc.right.count;
                    partitions.push(acc_to_group(acc.left));
                    partitions.push(acc_to_group(acc.right));
                } else {
                    partitions.push(None);
                    partitions.push(None);
                }

                total += 1;
            }
            //dbg!((level, level_count));
            //eprintln!();
        }

        // Simulation
        let leaves = (1 << (self.settings.tree_depth)) as usize - 1;
        let groups: Vec<Group> = partitions[leaves..].iter().filter_map(|a| *a).collect();
        self.engine.write(self.groups_gpu, &groups)?;

        let motion_params = motion_params_from_settings(&self.settings);

        self.engine.run(
            self.motion,
            self.groups_gpu,
            self.boids_gpu,
            self.settings.work_groups,
            1,
            1,
            bytemuck::cast_slice(&[motion_params]),
        )?;

        Ok(groups)
    }

    fn select(&mut self, level: u32, mask: u32, plane: Group) -> Result<()> {
        let select_params = SelectParams {
            level,
            mask,
            plane_pos: plane.center,
            plane_normal: plane.heading,
        };
        self.engine.run(
            self.select,
            self.acc_gpu,
            self.boids_gpu,
            self.settings.work_groups,
            1,
            1,
            bytemuck::cast_slice(&[select_params]),
        )
    }

    fn reduce(&mut self) -> Result<Accumulator> {
        let mut stride = 1u32;
        while stride < self.n_boids {
            self.engine.run(
                self.reduce,
                self.acc_gpu,
                self.acc_gpu,
                self.settings.work_groups,
                1,
                1,
                &stride.to_le_bytes(),
            )?;
            stride <<= 1;
        }
        let mut acc = [Accumulator::default()];
        self.engine.read(self.acc_gpu, &mut acc)?;
        Ok(acc[0])
    }
}

fn acc_to_group(acc: AccumulatorHalf) -> Option<Group> {
    (acc.count > 0).then(|| {
        let c = acc.count as f32;
        let [x, y, z] = acc.pos;

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let hx = rng.gen_range(-1.0..1.0);
        let hy = rng.gen_range(-1.0..1.0);
        let hz = rng.gen_range(-1.0..1.0);

        //let [hx, hy, hz] = acc.heading;
        Group {
            center: [x / c, y / c, z / c],
            heading: [hx / c, hy / c, hz / c],
            _filler0: 0,
            _filler1: 0,
        }
    })
}

fn random_boids(n: usize, scale: f32) -> Vec<Boid> {
    let mut rng = rand::thread_rng();
    let unit = Uniform::new(-1., 1.);
    let cube = Uniform::new(-scale, scale);
    (0..n)
        .map(|_| Boid {
            pos: [
                cube.sample(&mut rng),
                cube.sample(&mut rng),
                cube.sample(&mut rng),
            ],
            heading: [
                unit.sample(&mut rng),
                unit.sample(&mut rng),
                unit.sample(&mut rng),
            ],
            mask: 0,
            level: 0,
        })
        .collect()
}
