use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

#[derive(Clone)]
struct Ising2D {
    rows: usize,
    cols: usize,
    spins: Vec<i8>, // ±1
}

impl Ising2D {
    fn new(rows: usize, cols: usize, rng: &mut SmallRng) -> Self {
        let mut spins = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            spins.push(if rng.gen_bool(0.5) { 1 } else { -1 });
        }
        Self { rows, cols, spins }
    }

    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        i * self.cols + j
    }

    #[inline]
    fn spin(&self, i: usize, j: usize) -> i32 {
        self.spins[self.idx(i, j)] as i32
    }

    #[inline]
    fn nn_sum(&self, i: usize, j: usize) -> i32 {
        let up = (i + self.rows - 1) % self.rows;
        let down = (i + 1) % self.rows;
        let left = (j + self.cols - 1) % self.cols;
        let right = (j + 1) % self.cols;

        self.spin(up, j) + self.spin(down, j) + self.spin(i, left) + self.spin(i, right)
    }

    // J=1, kB=1; ΔE = 2 s_i Σ_nn s_j
    #[inline]
    fn delta_e_flip(&self, i: usize, j: usize) -> i32 {
        2 * self.spin(i, j) * self.nn_sum(i, j)
    }

    fn metropolis_sweep(&mut self, beta: f64, rng: &mut SmallRng) -> usize {
        let n = self.rows * self.cols;
        let mut accepted = 0usize;

        for _ in 0..n {
            let i = rng.gen_range(0..self.rows);
            let j = rng.gen_range(0..self.cols);

            let de = self.delta_e_flip(i, j) as f64;
            if de <= 0.0 || rng.gen::<f64>() < (-beta * de).exp() {
                let k = self.idx(i, j);
                self.spins[k] = -self.spins[k];
                accepted += 1;
            }
        }
        accepted
    }

    // Energy with each bond counted once (right + down), J=1
    fn total_energy(&self) -> i32 {
        let mut e = 0i32;
        for i in 0..self.rows {
            for j in 0..self.cols {
                let s = self.spin(i, j);
                let right = self.spin(i, (j + 1) % self.cols);
                let down = self.spin((i + 1) % self.rows, j);
                e -= s * (right + down);
            }
        }
        e
    }

    fn magnetization(&self) -> i32 {
        self.spins.iter().map(|&x| x as i32).sum()
    }
}

fn simulate(
    rows: usize,
    cols: usize,
    temperature: f64,
    n_therm: usize,
    n_sweeps: usize,
    seed: Option<u64>,
) -> (Ising2D, f64, f64, f64) {
    let mut rng = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_entropy(),
    };

    let beta = 1.0 / temperature;
    let mut model = Ising2D::new(rows, cols, &mut rng);

    // Not needed
    //for _ in 0..n_therm {
    //    model.metropolis_sweep(beta, &mut rng);
    //}

    let n_sites = (rows * cols) as f64;
    let mut e_sum = 0.0;
    let mut mabs_sum = 0.0;
    let mut acc_sum = 0.0;

    for i in 0..n_sweeps {
        // print progress every 1%
        if i % (n_sweeps / 100).max(1) == 0 {
            println!("\rProgress: {:.1}%", 100.0 * (i as f64) / (n_sweeps as f64));
        }
        let acc = model.metropolis_sweep(beta, &mut rng) as f64;
        acc_sum += acc / n_sites;

        let e = model.total_energy() as f64;
        let m = model.magnetization() as f64;

        e_sum += e / n_sites;
        mabs_sum += (m / n_sites).abs();
    }

    (
        model,
        e_sum / (n_sweeps as f64),
        mabs_sum / (n_sweeps as f64),
        acc_sum / (n_sweeps as f64),
    )
}


#[pyfunction]
#[pyo3(signature = (rows=64, cols=64, temperature=2.269, n_therm=2000, n_sweeps=10000, seed=None))]
fn ising_sim(
    rows: usize,
    cols: usize,
    temperature: f64,
    n_therm: usize,
    n_sweeps: usize,
    seed: Option<u64>,
) -> PyResult<Vec<i8>> {
    let (model, _e_avg, _mabs_avg, _acc) =
        simulate(rows, cols, temperature, n_therm, n_sweeps, seed);
    Ok(model.spins)
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ising_sim, m)?)?;
    Ok(())
}
