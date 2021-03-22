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
pub struct MotionParams {
    pub downsample: u32,
    pub speed: f32,
    pub dist_thresh: f32,
    pub cohere: f32,
    pub steer: f32,
    pub parallel: f32,
}

unsafe impl Zeroable for MotionParams {}
unsafe impl Pod for MotionParams {}

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct Settings {
    pub downsample: u32,
    pub work_groups: u32,
    pub speed: f32,
    pub dist_thresh: f32,
    pub cohere: f32,
    pub steer: f32,
    pub parallel: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            downsample: 1,
            work_groups: 40,
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
        downsample: settings.downsample,
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
    boids_gpu_a: Buffer,
    boids_gpu_b: Buffer,
    motion: Shader,
    boids_dirty: bool,
    buf_select: bool,
}

pub const LOCAL_X: u32 = 16;
impl Simulation {
    pub fn new(settings: Settings) -> Result<Self> {
        assert!(settings.work_groups > 0);
        let mut engine = rmds::Engine::new(true)?;
        let n_boids = settings.work_groups * LOCAL_X;

        let motion = engine.spirv(&read("kernels/motion.comp.spv")?)?;

        let boids_gpu_a = engine.buffer::<Boid>(n_boids as _)?;
        let boids_gpu_b = engine.buffer::<Boid>(n_boids as _)?;

        let boids = random_boids(n_boids as _, 10.);
        engine.write(boids_gpu_a, &boids)?;

        Ok(Self {
            n_boids,
            settings,
            engine,
            boids_gpu_a,
            boids_gpu_b,
            boids,
            motion,
            buf_select: false,
            boids_dirty: false,
        })
    }

    pub fn boids(&mut self) -> Result<&[Boid]> {
        if self.boids_dirty {
            self.engine.read(
                if self.buf_select {
                    self.boids_gpu_b
                } else {
                    self.boids_gpu_a
                },
                &mut self.boids,
            )?;
            self.boids_dirty = false;
        }
        Ok(&self.boids)
    }

    pub fn step(&mut self) -> Result<()> {
        self.boids_dirty = true;

        let motion_params = motion_params_from_settings(&self.settings);

        let (read_buf, write_buf) = match self.buf_select {
            false => (self.boids_gpu_a, self.boids_gpu_b),
            true => (self.boids_gpu_b, self.boids_gpu_a),
        };

        self.engine.run(
            self.motion,
            read_buf,
            write_buf,
            self.settings.work_groups,
            1,
            1,
            bytemuck::cast_slice(&[motion_params]),
        )?;

        self.buf_select = !self.buf_select;

        Ok(())
    }
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
