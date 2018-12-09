use multiple_pendulum_scheme::MultiUniformPendulum2D;

fn main() {
    let mut p = MultiUniformPendulum2D::new(2, 3.0);
    let dt = 2.0_f64.powi(-16);
    let out_interval = 2.0_f64.powi(-7);
    let mut t = 0.0;
    while t < 100.0 {
        p.tick_rk44(dt).unwrap();
        t += dt;
        if t % out_interval == 0.0 {
            let u = p.position_energy();
            let k = p.physical_energy();
            let v = p.get_pos();
            print!("{} {} {} {} {}\n\n\n", t, u, k, v[0], v[1]);
        }
    }
}
