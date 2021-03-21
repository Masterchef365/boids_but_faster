use rand::distributions::{Distribution, Uniform};
type Vec3 = nalgebra::Vector3<f32>;

fn main() {
    let n = 1 << 8;
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
struct BoidAccumulatorHalf {
    pos: Vec3,
    heading: Vec3,
    count: u32,
}

#[derive(Default, Debug, Copy, Clone)]
struct BoidAccumulator {
    left: BoidAccumulatorHalf,
    right: BoidAccumulatorHalf,
}

impl Default for BoidAccumulatorHalf {
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

fn plane_from_acc_half(half: &BoidAccumulatorHalf) -> Option<Plane> {
    if half.count == 0 {
        return None;
    }

    //dbg!(acc0.count);
    let n = half.count as f32;
    Some(Plane {
        pos: half.pos / n,
        normal: half.heading / n,
    })
}

fn plane_from_acc0(acc: &[BoidAccumulator]) -> (Option<Plane>, Option<Plane>) {
    let left = acc[0].left.count;
    let right = acc[0].right.count;
    dbg!((left, right, left + right));
    (
        plane_from_acc_half(&acc[0].left),
        plane_from_acc_half(&acc[0].right)
    )
}

fn build_accelerator(boids: &mut [Boid], acc: &mut [BoidAccumulator]) -> Vec<Option<Plane>> {
    boids.iter_mut().for_each(|b| {
        b.level = 0;
        b.mask = 0;
    });
    root_select(boids, acc);
    bubble(acc);
    let mut partitions = vec![plane_from_acc0(acc).0];

    let levels = 3;
    let mut total = 0;
    // Tree depth
    for level in 0..levels {
        // Mask for each leaf node
        for mask in 0..(1 << level) {
            // Parent node idx
            let plane_idx = total; 

            eprintln!(
                "Level: {}, Mask: {:b}, Plane idx: {}",
                level, mask, plane_idx
            );

            if let Some(plane) = &partitions[plane_idx as usize] {
                select(boids, acc, level, mask, plane);
                bubble(acc);
                let (left, right) = plane_from_acc0(acc);
                partitions.push(left);
                partitions.push(right);
            } else {
                partitions.push(None);
                partitions.push(None);
            }

            total += 1;
        }
        eprintln!();
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
    plane: &Plane,
) {
    for (boid, acc) in boids.iter_mut().zip(acc.iter_mut()) {
        for zero in [&mut acc.left, &mut acc.right].iter_mut() {
            zero.pos = Vec3::zeros();
            zero.heading = Vec3::zeros();
            zero.count = 0;
        }

        if boid.mask == mask && boid.level == level {
            let plane_face = plane_side(boid.pos, plane);

            // Basically push to a bit vec
            let new_bit = if plane_face { 1 << level } else { 0 };
            boid.mask |= new_bit;
            boid.level = level + 1;

            // Set and zero opposite planes
            let set = match plane_face { 
                true => &mut acc.left,
                false => &mut acc.right,
            };

            set.pos = boid.pos;
            set.heading = boid.heading;
            set.count = 1;
        }
    }
}

fn root_select(boids: &[Boid], acc: &mut [BoidAccumulator]) {
    for (boid, acc) in boids.iter().zip(acc.iter_mut()) {
        acc.left.pos = boid.pos;
        acc.left.heading = boid.heading;
        acc.left.count = 1;
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
        acc[base_idx].left.pos += acc[other_idx].left.pos;
        acc[base_idx].left.heading += acc[other_idx].left.heading;
        acc[base_idx].left.count += acc[other_idx].left.count;

        acc[base_idx].right.pos += acc[other_idx].right.pos;
        acc[base_idx].right.heading += acc[other_idx].right.heading;
        acc[base_idx].right.count += acc[other_idx].right.count;
    }
}
