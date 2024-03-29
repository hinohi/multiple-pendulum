use std::f64::consts::PI;
use std::io;

use ndarray::prelude::*;
use ndarray_linalg::{error::LinalgError, SolveH};

type Float = f64;

pub struct MultiUniformPendulum2D {
    n: usize,
    g: Float,
    pub t: Float,
    x: Array1<Float>,
    v: Array1<Float>,
}

impl MultiUniformPendulum2D {
    pub fn new(n: usize, g: Float) -> MultiUniformPendulum2D {
        MultiUniformPendulum2D {
            n,
            g,
            t: 0.0,
            x: Array1::from_vec(vec![2.0; n]),
            v: Array1::from_vec(vec![0.0; n]),
        }
    }

    pub fn dump_quantities<W: io::Write>(&self, w: &mut W) -> io::Result<()> {
        writeln!(
            w,
            "{} {} {} {}",
            self.t,
            self.position_energy(),
            self.physical_energy(),
            self.x
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(" "),
        )?;
        Ok(())
    }

    pub fn position_energy(&self) -> Float {
        let mut u = 0.0;
        for i in 0..self.n {
            u -= (self.n - i) as Float * self.x[i].cos();
        }
        u * self.g
    }

    pub fn physical_energy(&self) -> Float {
        let mut t = 0.0;
        for i in 0..self.n {
            let mi = (self.n - i) as Float;
            t += mi * self.v[i] * self.v[i];
            for j in 0..i {
                t += mi * self.v[i] * self.v[j] * (self.x[i] - self.x[j]).cos();
            }
            for j in i + 1..self.n {
                let mj = (self.n - j) as Float;
                t += mj * self.v[i] * self.v[j] * (self.x[i] - self.x[j]).cos();
            }
        }
        t * 0.5
    }

    fn calc_force(
        &self,
        x: &Array1<Float>,
        v: &Array1<Float>,
    ) -> Result<Array1<Float>, LinalgError> {
        let mut force = v.clone() * v;

        let mut array = vec![0.0; self.n * self.n];
        for i in 0..self.n {
            for j in i + 1..self.n {
                let m = (self.n - j) as Float;
                let a = m * (x[j] - x[i]).sin();
                array[i * self.n + j] = a;
                array[j * self.n + i] = -a;
            }
        }
        let a = Array2::from_shape_vec((self.n, self.n), array).unwrap();
        force = a.dot(&force);
        for i in 0..self.n {
            force[i] -= (self.n - i) as Float * self.g * x[i].sin();
        }

        let mut array = vec![0.0; self.n * self.n];
        for i in 0..self.n {
            array[i * self.n + i] = (self.n - i) as Float;
            for j in i + 1..self.n {
                let m = (self.n - j) as Float;
                let a = m * (x[j] - x[i]).cos();
                array[i * self.n + j] = a;
                array[j * self.n + i] = a;
            }
        }
        let a = Array2::from_shape_vec((self.n, self.n), array).unwrap();
        a.solveh_into(force)
    }

    fn x_mod_pi(&mut self) {
        self.x
            .iter_mut()
            .map(|x| {
                if x.is_sign_negative() {
                    *x += PI as Float;
                } else if *x > PI as Float {
                    *x -= PI as Float;
                }
            })
            .last()
            .unwrap();
    }

    pub fn update_euler(&mut self, dt: Float) -> Result<(), LinalgError> {
        let f = self.calc_force(&self.x, &self.v)?;
        self.x.zip_mut_with(&self.v, |x, v| *x += v * dt);
        self.v.zip_mut_with(&f, |v, f| *v += f * dt);
        self.t += dt;
        self.x_mod_pi();
        Ok(())
    }

    pub fn update_rk44(&mut self, dt: Float) -> Result<(), LinalgError> {
        let f1 = self.calc_force(&self.x, &self.v)?;

        let mut x1 = self.x.clone();
        let mut v1 = self.v.clone();
        x1.zip_mut_with(&self.v, |x, v| *x += v * dt * 0.5);
        v1.zip_mut_with(&f1, |v, f| *v += f * dt * 0.5);
        let f2 = self.calc_force(&x1, &v1)?;

        let mut x2 = self.x.clone();
        let mut v2 = self.v.clone();
        x2.zip_mut_with(&v1, |x, v| *x += v * dt * 0.5);
        v2.zip_mut_with(&f2, |v, f| *v += f * dt * 0.5);
        let f3 = self.calc_force(&x2, &v2)?;

        let mut x3 = self.x.clone();
        let mut v3 = self.v.clone();
        x3.zip_mut_with(&v2, |x, v| *x += v * dt);
        v3.zip_mut_with(&f3, |v, f| *v += f * dt);
        let f4 = self.calc_force(&x3, &v3)?;

        self.x.zip_mut_with(&self.v, |x, v| *x += v * dt / 6.0);
        self.x.zip_mut_with(&v1, |x, v| *x += v * dt / 3.0);
        self.x.zip_mut_with(&v2, |x, v| *x += v * dt / 3.0);
        self.x.zip_mut_with(&v3, |x, v| *x += v * dt / 6.0);
        self.v.zip_mut_with(&f1, |v, f| *v += f * dt / 6.0);
        self.v.zip_mut_with(&f2, |v, f| *v += f * dt / 3.0);
        self.v.zip_mut_with(&f3, |v, f| *v += f * dt / 3.0);
        self.v.zip_mut_with(&f4, |v, f| *v += f * dt / 6.0);
        self.t += dt;
        self.x_mod_pi();
        Ok(())
    }

    pub fn update_irk24(&mut self, dt: Float) -> Result<(), LinalgError> {
        let a11 = 0.25 * dt;
        let a12 = (0.25 - 3_f64.sqrt() / 6.0) * dt;
        let a21 =(0.25 + 3_f64.sqrt() / 6.0) * dt;
        let a22 = 0.25 * dt;
        let b1 = 0.5 * dt;
        let b2 = 0.5 * dt;

        let mut v1 = self.v.clone();
        let mut v2 = self.v.clone();
        let mut f1 = self.calc_force(&self.x, &self.v)?;
        let mut f2 = f1.clone();
        for _ in 0..100 {
            let mut x1 = self.x.clone();
            x1.zip_mut_with(&v1, |x, v| *x += v * a11);
            x1.zip_mut_with(&v2, |x, v| *x += v * a12);
            let mut x2 = self.x.clone();
            x2.zip_mut_with(&v1, |x, v| *x += v * a21);
            x2.zip_mut_with(&v2, |x, v| *x += v * a22);

            v1 = self.v.clone();
            v1.zip_mut_with(&f1, |v, f| *v += f * a11);
            v1.zip_mut_with(&f2, |v, f| *v += f * a12);
            v2 = self.v.clone();
            v2.zip_mut_with(&f1, |v, f| *v += f * a21);
            v2.zip_mut_with(&f2, |v, f| *v += f * a22);

            f1 = self.calc_force(&x1, &v1)?;
            f2 = self.calc_force(&x2, &v2)?;
        }
        self.x.zip_mut_with(&v1, |x, v| *x += v * b1);
        self.x.zip_mut_with(&v2, |x, v| *x += v * b2);
        self.v.zip_mut_with(&f1, |v, f| *v += f * b1);
        self.v.zip_mut_with(&f2, |v, f| *v += f * b2);
        self.t += dt;
        self.x_mod_pi();
        Ok(())
    }
}
