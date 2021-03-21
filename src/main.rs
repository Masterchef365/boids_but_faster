use rand::distributions::{Distribution, Uniform};
type Vec3 = nalgebra::Vector3<f32>;

fn main() {
    let n = 1 << 4;
    let mut acc = vec![BoidAccumulator::default(); n];
    let mut boids = random_boids(n, 10.);
    let accel = build_accelerator(&mut boids, &mut acc);
}

fn random_boids(n: usize, scale: f32) -> Vec<Boid> {
    let mut rng = rand::thread_rng();
    let unit = Uniform::new(-1., 1.);
    let cube = Uniform::new(-scale, scale);
    (0..n)
        .map(|_| Boid {
            pos: Vec3::new(
                cube.sample(&mut rng),
                cube.sample(&mut rng),
                cube.sample(&mut rng),
            ),
            heading: Vec3::new(
                unit.sample(&mut rng),
                unit.sample(&mut rng),
                unit.sample(&mut rng),
            ),
            mask: 0,
            level: 0,
        })
        .collect()
}

#[derive(Debug, Copy, Clone)]
struct Boid {
    pos: Vec3,
    heading: Vec3,
    mask: u32,
    level: u32,
}

#[derive(Debug, Copy, Clone)]
struct BoidAccumulator {
    pos: Vec3,
    heading: Vec3,
    count: u32,
}

impl Default for BoidAccumulator {
    fn default() -> Self {
        Self {
            pos: Vec3::zeros(),
            heading: Vec3::zeros(),
            count: 0,
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct Plane {
    pos: Vec3,
    normal: Vec3,
}

fn plane_from_acc0(acc: &[BoidAccumulator]) -> Option<Plane> {
    let acc0 = &acc[0];
    if acc0.count == 0 {
        return None;
    }

    //dbg!(acc0.count);
    let n = acc0.count as f32;
    Some(Plane {
        pos: acc0.pos / n,
        normal: acc0.heading / n,
    })
}

fn build_accelerator(boids: &mut [Boid], acc: &mut [BoidAccumulator]) -> Vec<Option<Plane>> {
    boids.iter_mut().for_each(|b| {
        b.level = 0;
        b.mask = 0;
    });
    select(boids, acc, 0, 0, None);
    bubble(acc);
    let mut partitions = vec![plane_from_acc0(acc)];

    let levels = 3;
    let mut total = 0;
    // Tree depth
    for level in 0..levels {
        // Mask for each leaf node
        for mask in 0..(2 << level) {
            // Parent node idx
            let plane_idx = total / 2; 

            println!(
                "Level: {}, Mask: {:b}, Plane idx: {}",
                level, mask, plane_idx
            );

            if let Some(plane) = &partitions[plane_idx as usize] {
                select(boids, acc, level, mask, Some(plane));
                bubble(acc);
                partitions.push(plane_from_acc0(acc));
            } else {
                partitions.push(None);
            }

            total += 1;
        }
        println!();
    }

    //dbg!(&partitions);
    partitions
}

fn plane_side(pt: Vec3, plane: &Plane) -> bool {
    (pt - plane.pos).dot(&plane.normal) > 0.
}

fn select(
    boids: &mut [Boid],
    acc: &mut [BoidAccumulator],
    level: u32,
    mask: u32,
    plane: Option<&Plane>,
) {
    for (boid, acc) in boids.iter_mut().zip(acc.iter_mut()) {
        if boid.mask == mask && boid.level == level {
            acc.pos = boid.pos;
            acc.heading = boid.heading;
            acc.count = 1;
            if let Some(plane) = plane {
                let plane_face = plane_side(boid.pos, plane);
                println!("\t{}", plane_face);
                let new_bit = if plane_face { 1 << level } else { 0 };
                boid.mask |= new_bit;
                boid.level = level + 1;
            }
        } else {
            acc.pos = Vec3::zeros();
            acc.heading = Vec3::zeros();
            acc.count = 0;
        }
    }
}

fn bubble(acc: &mut [BoidAccumulator]) {
    let mut stride = 2;
    while stride <= acc.len() {
        bubble_step(acc, stride);
        stride <<= 1;
    }
}

fn bubble_step(acc: &mut [BoidAccumulator], stride: usize) {
    for invoke_idx in 0..acc.len() / stride {
        let base_idx = invoke_idx * stride;
        let other_idx = base_idx + stride / 2;
        acc[base_idx].pos += acc[other_idx].pos;
        acc[base_idx].heading += acc[other_idx].heading;
        acc[base_idx].count += acc[other_idx].count;
    }
}
