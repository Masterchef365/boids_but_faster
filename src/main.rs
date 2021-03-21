mod sim;
use sim::Simulation;
fn main() {
    let mut sim = Simulation::new(1 << 10, 3);
    let planes = sim.step();
    dbg!(planes);
}
