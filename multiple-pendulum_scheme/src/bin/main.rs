use multiple_pendulum_scheme::MultiUniformPendulum2D;
use std::io::{stdout, BufWriter};

fn main() {
    let mut cout = BufWriter::new(stdout());
    let mut p = MultiUniformPendulum2D::new(2, 3.0);
    let dt = 2.0_f64.powi(-12);
    let out_interval = 2.0_f64.powi(-7);
    while p.t < 100.0 {
        p.update_irk24(dt).unwrap();
        if p.t % out_interval == 0.0 {
            p.dump_quantities(&mut cout).unwrap();
        }
    }
}
