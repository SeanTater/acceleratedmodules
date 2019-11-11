use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::cmp::max;
use rand::distributions::Distribution;


#[pyfunction]
/// Formats the sum of two numbers as string
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyclass(module = "rustsim")]
struct Simulation {
    safety_stock: usize,
    lead_time: usize,
    order_quantity: usize,
    job_lot_zipf: f64,
    itemwise_traffic_zipf: f64,
}



#[pymethods]
impl Simulation {
    #[new]
    fn new(obj: &PyRawObject, safety_stock: usize, lead_time: usize, order_quantity: usize, job_lot_zipf: Option<f64>, itemwise_traffic_zipf: Option<f64>) {
        obj.init(Simulation {
            safety_stock,
            lead_time,
            order_quantity,
            job_lot_zipf: job_lot_zipf.unwrap_or(2.75),
            itemwise_traffic_zipf: itemwise_traffic_zipf.unwrap_or(4.0),
        });
    }



    /// Do exactly the same search Python does
    fn simulate_demand_inner(&self, starting_quantity: usize) -> (usize, usize, usize, usize, f64, f64) {
        let mut successful_transactions = 0;
        let mut successful_sales = 0;
        let mut failed_transactions = 0;
        let mut failed_sales = 0;
        let mut stock = starting_quantity;
        let mut trucks = vec![0; self.lead_time];
        let mut rng = rand::thread_rng();
        let jl_zipf = zipf::ZipfDistribution::new(1000, self.job_lot_zipf).unwrap();
        let it_zipf = zipf::ZipfDistribution::new(1000, self.itemwise_traffic_zipf).unwrap();

        for day in 0..365 {
            // A truck arrived
            stock += trucks[day % self.lead_time];
            // This many customers arrive
            for _customer in 0..it_zipf.sample(&mut rng) {
                // This customer wants this many
                let request = jl_zipf.sample(&mut rng);
                if stock >= request {
                    // There are enough.
                    successful_transactions += 1;
                    successful_sales += request;
                    stock -= request;
                } else {
                    // There are not enough
                    failed_transactions += 1;
                    failed_sales += request;
                }
            }
            // The day is over. Start making orders.
            if stock < self.safety_stock {
                let short = max(self.safety_stock - stock, 0);
                let orders = (short + self.order_quantity - 1) / self.order_quantity;
                trucks[(day+self.lead_time-1) % self.lead_time] = orders * self.order_quantity;
            }
        }
        (
            successful_transactions, successful_sales,
            failed_transactions, failed_sales,
            successful_transactions as f64 / (successful_transactions as f64 + failed_transactions as f64),
            successful_sales as f64 / (successful_sales as f64 + failed_sales as f64)
        )
    }

    fn simulate_demand(&self, py: Python<'_>, starting_quantity: usize) -> PyResult<PyObject> {
        Ok(self.simulate_demand_inner(starting_quantity).into_py(py))
    }

    fn repeat_simulate_demand(&self, py: Python<'_>, starting_quantity: usize, count: usize) -> PyResult<PyObject> {
        let (mut st, mut ss, mut ft, mut fs) = (0, 0, 0, 0);
        for _ in 0..count {
            let (xst, xss, xft, xfs, _, _) = self.simulate_demand_inner(starting_quantity);
            st += xst;
            ss += xss;
            ft += xft;
            fs += xfs;
        }
        Ok((st, ss, ft, fs, st/(st+ft), ss/(ss+fs)).into_py(py))
    }
}


/// This module is a python module implemented in Rust.
#[pymodule]
fn rustsim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sum_as_string))?;
    m.add_class::<Simulation>()?;

    Ok(())
}