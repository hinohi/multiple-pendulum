#![feature(test)]
extern crate test;
use test::Bencher;

use ndarray::prelude::*;
use ndarray_linalg::{error::LinalgError, SolveH};

type Float = f64;

pub struct A {
    n: usize,
    g: Float,
    x: Array1<Float>,
    v: Array1<Float>,
}

impl A {
    pub fn new(n: usize, g: Float) -> A {
        A {
            n,
            g,
            x: Array1::from_vec((0..n).map(|i| i as Float * 0.2).collect()),
            v: Array1::from_vec((0..n).map(|i| i as Float * 0.1).collect()),
        }
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

    pub fn tick_rk11(&mut self, dt: Float) -> Result<(), LinalgError> {
        let force = self.calc_force(&self.x, &self.v)?;
        self.x.zip_mut_with(&self.v, |x, v| *x += v * dt);
        self.v.zip_mut_with(&force, |v, f| *v += f * dt);
        Ok(())
    }
}

pub struct B {
    n: usize,
    g: Float,
    x: Vec<Float>,
    v: Vec<Float>,
}

impl B {
    pub fn new(n: usize, g: Float) -> B {
        B {
            n,
            g,
            x: (0..n).map(|i| i as Float * 0.2).collect(),
            v: (0..n).map(|i| i as Float * 0.1).collect(),
        }
    }

    fn calc_force(&self, x: &Vec<Float>, v: &Vec<Float>) -> Result<Array1<Float>, LinalgError> {
        let mut f = vec![0.0; self.n];
        let v2 = v.iter().map(|i| i * i).collect::<Vec<Float>>();
        for i in 0..self.n {
            f[i] -= (self.n - i) as Float * self.g * x[i].cos();
            for j in 0..i {
                let s = (x[i] - x[j]).sin();
                f[i] += (self.n - i) as Float * s * v2[j];
                f[j] -= (self.n - j) as Float * s * v2[i];
            }
        }

        let mut array = vec![0.0; self.n * self.n];
        for i in 0..self.n {
            array[i * self.n + i] = (self.n - i) as Float;
            for j in 0..i {
                let m = (self.n - i) as Float;
                let a = m * (x[j] - x[i]).cos();
                array[i * self.n + j] = a;
                array[j * self.n + i] = a;
            }
        }
        let force = Array1::from_vec(f);
        let a = Array2::from_shape_vec((self.n, self.n), array).unwrap();
        a.solveh_into(force)
    }

    pub fn tick_rk11(&mut self, dt: Float) -> Result<(), LinalgError> {
        let force = self.calc_force(&self.x, &self.v)?;
        self.x
            .iter_mut()
            .zip(self.v.iter())
            .map(|(x, v)| *x += v * dt)
            .last()
            .unwrap();
        self.v
            .iter_mut()
            .zip(force.iter())
            .map(|(v, f)| *v += f * dt)
            .last()
            .unwrap();
        Ok(())
    }
}

#[bench]
fn tick_a(b: &mut Bencher) {
    let mut p = A::new(10, 2.0);
    b.iter(|| {
        p.tick_rk11(0.001).unwrap();
    })
}

#[bench]
fn tick_b(b: &mut Bencher) {
    let mut p = B::new(10, 2.0);
    b.iter(|| {
        p.tick_rk11(0.001).unwrap();
    })
}
