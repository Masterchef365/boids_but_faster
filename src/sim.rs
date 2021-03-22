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

pub struct Simulation {
    engine: Engine,
    boids: Vec<Boid>,
    work_groups: u32,
    n_boids: u32,
    boids_gpu: Buffer,
    acc_gpu: Buffer,
    setup: Shader,
    reduce: Shader,
    boids_dirty: bool,
}

pub const LOCAL_X: u32 = 16;
impl Simulation {
    pub fn new(work_groups: u32) -> Result<Self> {
        let mut engine = rmds::Engine::new(true)?;
        let n_boids = work_groups * LOCAL_X;

        let setup = engine.spirv(&read("kernels/setup.comp.spv")?)?;
        let reduce = engine.spirv(&read("kernels/reduce.comp.spv")?)?;

        let acc_gpu = engine.buffer::<Accumulator>(n_boids as _)?;
        let boids_gpu = engine.buffer::<Boid>(n_boids as _)?;

        let boids = random_boids(n_boids as _, 10.);
        engine.write(boids_gpu, &boids)?;

        Ok(Self {
            n_boids,
            acc_gpu,
            boids_gpu,
            engine,
            work_groups,
            setup,
            reduce,
            boids,
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
        self.boids_dirty = true;
        self.engine.run(self.setup, self.acc_gpu, self.boids_gpu, self.work_groups, 1, 1, &[])?;
        dbg!(self.reduce()?);
        Ok(())
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

fn main() -> Result<()> {
    let mut sim = Simulation::new(16)?;
    sim.step()?;
    Ok(())
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
