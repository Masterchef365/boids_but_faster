use rand::distributions::{Distribution, Uniform};
use bytemuck::{Zeroable, Pod};
use std::fs::read;
use anyhow::Result;

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

fn main() -> Result<()> {
    let mut engine = rmds::Engine::new(true)?;
    const LOCAL_X: usize = 16;
    const WORK_GROUPS: usize = 40;
    const N_BOIDS: usize = LOCAL_X * WORK_GROUPS;

    let setup = engine.spirv(&read("kernels/setup.comp.spv")?)?;
    let reduce = engine.spirv(&read("kernels/reduce.comp.spv")?)?;

    let mut acc = vec![Accumulator::default(); N_BOIDS];
    let boids = random_boids(N_BOIDS, 10.);

    let acc_gpu = engine.buffer::<Accumulator>(N_BOIDS)?;
    let boids_gpu = engine.buffer::<Boid>(N_BOIDS)?;

    engine.write(boids_gpu, &boids)?;
    engine.run(setup, acc_gpu, boids_gpu, WORK_GROUPS as _, 1, 1, &[])?;
    let mut stride = 1u32;
    while stride < N_BOIDS as u32 {
        engine.run(reduce, acc_gpu, acc_gpu, WORK_GROUPS as _, 1, 1, &stride.to_le_bytes())?;
        stride <<= 1;
    }
    engine.read(acc_gpu, &mut acc)?;

    dbg!(acc[0]);

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
