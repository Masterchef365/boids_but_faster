mod sim;
use sim::Simulation;

use anyhow::Result;
use klystron::{
    runtime_3d::{launch, App},
    DrawType, Engine, FramePacket, Material, Matrix4, Mesh, Object, Vertex, UNLIT_FRAG, UNLIT_VERT,
};
use nalgebra::Vector3;

pub fn main() -> Result<()> {
    let vr = std::env::args().skip(1).next().is_some();
    launch::<MyApp>(vr, ())
}

struct MyApp {
    lines_material: Material,
    sim: Simulation,
    boid_mesh: Mesh,
    plane_mesh: Mesh,
    //planes: Vec<Plane>,
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

    type Args = ();

    fn new(engine: &mut dyn Engine, _args: Self::Args) -> Result<Self> {
        let sim = Simulation::new(16, 3)?;

        let lines_material = engine.add_material(UNLIT_VERT, UNLIT_FRAG, DrawType::Lines)?;

        let (vertices, indices) = boid();
        let boid_mesh = engine.add_mesh(&vertices, &indices)?;

        let (vertices, indices) = plane(10.);
        let plane_mesh = engine.add_mesh(&vertices, &indices)?;

        Ok(Self {
            plane_mesh,
            sim,
            //planes: Vec::new(),
            boid_mesh,
            lines_material,
            frame: 0,
        })
    }

    fn next_frame(&mut self, engine: &mut dyn Engine) -> Result<FramePacket> {
        let mut objects = Vec::new();

        //if self.frame % 60 == 0 {
        let start = std::time::Instant::now();
        //self.planes = self.sim.step(0.04);
        self.sim.step()?;
        let elap = start.elapsed();
        println!("{} boid sim took {} ms", self.sim.boids()?.len(), elap.as_secs_f32() * 1000.);

        /*
        for plane in &self.planes {
            objects.push(Object {
                material: self.lines_material,
                mesh: self.plane_mesh,
                transform: Matrix4::new_translation(&plane.pos) 
                    // * point_towards(plane.normal),
                    * point_towards(plane.heading),
            });
        }
        */

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

