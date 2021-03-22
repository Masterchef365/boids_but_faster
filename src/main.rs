mod sim;
use sim::{Simulation, Group, Settings};

use anyhow::{Result, Context};
use klystron::{
    runtime_3d::{launch, App},
    DrawType, Engine, FramePacket, Material, Matrix4, Mesh, Object, Vertex, UNLIT_FRAG, UNLIT_VERT,
};
use nalgebra::Vector3;

pub fn main() -> Result<()> {
    let vr = std::env::args().skip(1).next().is_some();
    let settings_path = "settings.yml";
    let settings = match std::fs::File::open(settings_path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Could not {}, writing defualts and exiting", settings_path);
            let file = std::fs::File::create(settings_path)?;
            serde_yaml::to_writer(file, &Settings::default())?;
            return Ok(());
        }
        Ok(f) => {
            serde_yaml::from_reader(f).context("Parsing failed")?
        },
        e => {
            return e.context("Failed to read settings.yml").map(|_| ());
        },
    };

    launch::<MyApp>(vr, settings)
}

struct MyApp {
    lines_material: Material,
    sim: Simulation,
    boid_mesh: Mesh,
    plane_mesh: Mesh,
    planes: Vec<Group>,
    frame: u32,
}

fn point_towards(vec: Vector3<f32>) -> Matrix4<f32> {
    let y = vec.normalize();
    let x = y.cross(&Vector3::y()).normalize();
    let z = x.cross(&y).normalize();
    nalgebra::Matrix3::from_columns(&[x, y, z]).to_homogeneous()
}

impl App for MyApp {
    const NAME: &'static str = "Boids";

    type Args = Settings;

    fn new(engine: &mut dyn Engine, settings: Self::Args) -> Result<Self> {
        let sim = Simulation::new(settings)?;

        let lines_material = engine.add_material(UNLIT_VERT, UNLIT_FRAG, DrawType::Lines)?;

        let (vertices, indices) = boid();
        let boid_mesh = engine.add_mesh(&vertices, &indices)?;

        let (vertices, indices) = plane(10.);
        let plane_mesh = engine.add_mesh(&vertices, &indices)?;

        Ok(Self {
            plane_mesh,
            sim,
            planes: Vec::new(),
            boid_mesh,
            lines_material,
            frame: 0,
        })
    }

    fn next_frame(&mut self, engine: &mut dyn Engine) -> Result<FramePacket> {
        let mut objects = Vec::new();

        //if self.frame % 10 == 0 {
            let start = std::time::Instant::now();
            self.planes = self.sim.step()?;
            dbg!(self.planes.len());
            let elap = start.elapsed();
            println!("{} boid sim took {} ms", self.sim.boids()?.len(), elap.as_secs_f32() * 1000.);
        //}

        for plane in &self.planes {
            objects.push(Object {
                material: self.lines_material,
                mesh: self.plane_mesh,
                transform: Matrix4::new_translation(&Vector3::from(plane.center)) 
                    // * point_towards(plane.normal),
                    * point_towards(Vector3::from(plane.heading)),
            });
        }

        for boid in self.sim.boids()? {
            objects.push(Object {
                material: self.lines_material,
                mesh: self.boid_mesh,
                transform: Matrix4::new_translation(&Vector3::from(boid.pos)) * point_towards(Vector3::from(boid.heading)),
            });
        }

        engine.update_time_value(self.frame as f32 / 100.)?;
        self.frame += 1;

        Ok(FramePacket { objects })
    }
}

fn boid() -> (Vec<Vertex>, Vec<u16>) {
    let color = [1.; 3];
    let vertices = vec![
        Vertex::new([0.0, 0.0, 0.0], color),
        Vertex::new([0.0, 1.0, 0.0], color),
    ];

    let indices = vec![0, 1];

    (vertices, indices)
}

fn plane(size: f32) -> (Vec<Vertex>, Vec<u16>) {
    let color = [1., 0.3, 0.];
    let vertices = vec![
        Vertex::new([size, size, 0.], color),
        Vertex::new([size, -size, 0.], color),
        Vertex::new([-size, -size, 0.], color),
        Vertex::new([-size, size, 0.], color),
    ];

    let indices = vec![0, 1, 1, 2, 2, 3, 3, 0];

    (vertices, indices)
}

