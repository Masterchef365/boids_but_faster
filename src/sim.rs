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

pub struct Simulation {
    engine: Engine,
    boids: Vec<Boid>,
    work_groups: u32,
    n_boids: u32,
    boids_gpu: Buffer,
    groups_gpu: Buffer,
    tree_depth: u32,
    acc_gpu: Buffer,
    setup: Shader,
    reduce: Shader,
    motion: Shader,
    select: Shader,
    boids_dirty: bool,
}

pub const LOCAL_X: u32 = 16;
impl Simulation {
    pub fn new(work_groups: u32, tree_depth: u32) -> Result<Self> {
        assert!(work_groups > 0);
        assert!(tree_depth > 0);
        let mut engine = rmds::Engine::new(true)?;
        let n_boids = work_groups * LOCAL_X;

        let setup = engine.spirv(&read("kernels/setup.comp.spv")?)?;
        let reduce = engine.spirv(&read("kernels/reduce.comp.spv")?)?;
        let motion = engine.spirv(&read("kernels/motion.comp.spv")?)?;
        let select = engine.spirv(&read("kernels/select.comp.spv")?)?;

        let acc_gpu = engine.buffer::<Accumulator>(n_boids as _)?;
        let boids_gpu = engine.buffer::<Boid>(n_boids as _)?;
        let groups_gpu = engine.buffer::<Group>((1 << tree_depth) - 1)?;

        let boids = random_boids(n_boids as _, 10.);
        engine.write(boids_gpu, &boids)?;

        Ok(Self {
            n_boids,
            groups_gpu,
            tree_depth,
            acc_gpu,
            boids_gpu,
            engine,
            work_groups,
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

    pub fn step(&mut self) -> Result<()> {
        // Setup
        self.boids_dirty = true;
        self.engine.run(
            self.setup,
            self.acc_gpu,
            self.boids_gpu,
            self.work_groups,
            1,
            1,
            &[],
        )?;
        let acc = self.reduce()?;
        let mut partitions = vec![acc_to_group(acc.left)];

        // Build acceleration tree
        let mut total = 0;
        // Tree depth
        for level in 0..self.tree_depth {
            // Mask for each leaf node
            for mask in 0..(1 << level) {
                // Parent node idx
                let plane_idx = total; 

                eprintln!(
                    "Level: {}, Mask: {:b}, Plane idx: {}",
                    level, mask, plane_idx
                );

                if let Some(plane) = partitions[plane_idx as usize] {
                    self.select(level, mask, plane)?;
                    let acc = self.reduce()?;
                    dbg!(acc);
                    partitions.push(acc_to_group(acc.left));
                    partitions.push(acc_to_group(acc.right));
                } else {
                    partitions.push(None);
                    partitions.push(None);
                }

                total += 1;
            }
            eprintln!();
        }

        // Simulation
        let leaves = (1 << (self.tree_depth)) as usize - 1;
        let groups: Vec<Group> = partitions[leaves..].iter().filter_map(|a| *a).collect();
        self.engine.write(self.groups_gpu, &groups)?;

        let motion_params = MotionParams {
            n_groups: groups.len() as _,
            speed: 0.04,
            dist_thresh: 5.,
            cohere: 0.5,
            steer: 0.12,
            parallel: 0.12,
        };

        self.engine.run(
            self.motion,
            self.groups_gpu,
            self.boids_gpu,
            self.work_groups,
            1,
            1,
            bytemuck::cast_slice(&[motion_params]),
        )?;


        Ok(())
    }

    fn select(&mut self, level: u32, mask: u32, plane: Group) -> Result<()> {
        let select_params = SelectParams {
            level,
            mask,
            plane_pos: plane.center,
            plane_normal: plane.heading,
        };
        self.engine.run(
            self.setup,
            self.acc_gpu,
            self.boids_gpu,
            self.work_groups,
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
                self.work_groups,
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
        let [hx, hy, hz] = acc.heading;
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
