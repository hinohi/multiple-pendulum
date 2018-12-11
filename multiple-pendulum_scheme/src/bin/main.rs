use multiple_pendulum_scheme::MultiUniformPendulum2D;
use std::io::{stdout, BufWriter};

fn main() {
    let mut cout = BufWriter::new(stdout());
    let mut p = MultiUniformPendulum2D::new(2, 3.0);
    let dt = 2.0_f64.powi(-10);
    let out_interval = 2.0_f64.powi(-7);
    while p.t < 100.0 {
        p.update_rk44(dt).unwrap();
        if p.t % out_interval == 0.0 {
            p.dump_quantities(&mut cout).unwrap();
        }
    }
}
