use rand::distributions::{Distribution, Uniform};
type Vec3 = nalgebra::Vector3<f32>;

pub struct Simulation {
    acc: Vec<BoidAccumulator>,
    boids: Vec<Boid>,
    tree_depth: u32,
}

impl Simulation {
    pub fn new(n: usize, tree_depth: u32) -> Self {
        Self {
            acc: vec![BoidAccumulator::default(); n],
            boids: random_boids(n, 10.),
            tree_depth,
        }
    }

    pub fn step(&mut self, speed: f32) -> Vec<Plane> {
        let accel = build_accelerator(&mut self.boids, &mut self.acc, self.tree_depth);
        let leaves = (1 << (self.tree_depth)) as usize - 1;
        motion(&mut self.boids, dbg!(&accel[leaves..]), speed);
        accel.into_iter().filter_map(|a| a).collect()
    }

    pub fn boids(&self) -> &[Boid] {
        &self.boids
    }
}

fn motion(boids: &mut [Boid], tree: &[Option<Plane>], speed: f32) {
    dbg!(tree.len());
    for boid in boids {
        // Averaging
        let mut avg_neighbor_direction = Vec3::zeros();
        let mut avg_neighbor_offset = Vec3::zeros();
        let mut avg_dist = 0.;
        let mut total_neighbors = 0;

        for plane in tree.iter().filter_map(|p| *p) {
            let offset = plane.pos - boid.pos;
            avg_neighbor_direction += plane.heading.normalize();
            avg_neighbor_offset += offset.normalize();
            avg_dist += offset.magnitude();
            total_neighbors += 1;
        }

        if total_neighbors != 0 {
            avg_neighbor_direction.normalize_mut();
            avg_neighbor_offset.normalize_mut();
            avg_dist /= total_neighbors as f32;

            // Behaviour
            let cohere = (0.5 - avg_dist).max(0.).min(1.);
            let away = boid.heading.cross(&avg_neighbor_offset);
            let closeavoid = away.lerp(&avg_neighbor_offset, cohere);

            let new_heading = (boid.heading + //.
                closeavoid * 0.12 +  //.
                avg_neighbor_direction * 0.12)
                .normalize();
            if new_heading[0].is_nan() || new_heading[1].is_nan() || new_heading[2].is_nan() {
                println!("{}", new_heading);
            } else {
                boid.heading = new_heading;
            }
        }

        boid.pos += boid.heading * speed;
    }
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
pub struct Boid {
    pub pos: Vec3,
    pub heading: Vec3,
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
pub struct Plane {
    pub pos: Vec3,
    pub normal: Vec3,
    pub heading: Vec3,
    // TODO: Weight param?
}

fn plane_from_acc_half(half: &BoidAccumulatorHalf) -> Option<Plane> {
    if half.count == 0 {
        return None;
    }

    let n = half.count as f32;

    // Pick an arbitrary axis to cross with
    let axis = if half.heading.dot(&Vec3::y()) > 0. {
        if half.heading.dot(&Vec3::z()) > 0. {
            Vec3::z()
        } else {
            Vec3::y()
        }
    } else {
        Vec3::x()
    };
    let normal = half.heading.cross(&axis).normalize();

    Some(Plane {
        pos: half.pos / n,
        heading: half.heading / n,
        normal,
    })
}

fn plane_from_acc0(acc: &[BoidAccumulator]) -> (Option<Plane>, Option<Plane>) {
    //let left = acc[0].left.count;
    //let right = acc[0].right.count;
    //dbg!((left, right, left + right));
    (
        plane_from_acc_half(&acc[0].left),
        plane_from_acc_half(&acc[0].right)
    )
}

fn build_accelerator(boids: &mut [Boid], acc: &mut [BoidAccumulator], tree_depth: u32) -> Vec<Option<Plane>> {
    // Reset
    boids.iter_mut().for_each(|b| {
        b.level = 0;
        b.mask = 0;
    });

    // Make initial partition
    root_select(boids, acc);
    bubble(acc);
    let mut partitions = vec![plane_from_acc0(acc).0];

    let mut total = 0;
    // Tree depth
    for level in 0..tree_depth {
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
        //eprintln!();
    }

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

