use pyo3::prelude::*;
use rand::distributions::Distribution;
use std::convert::TryInto;
use ocl::ProQue;

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
    fn new(
        obj: &PyRawObject,
        safety_stock: usize,
        lead_time: usize,
        order_quantity: usize,
        job_lot_zipf: Option<f64>,
        itemwise_traffic_zipf: Option<f64>,
    ) {
        obj.init(Simulation {
            safety_stock,
            lead_time,
            order_quantity,
            job_lot_zipf: job_lot_zipf.unwrap_or(2.75),
            itemwise_traffic_zipf: itemwise_traffic_zipf.unwrap_or(4.0),
        });
    }

    fn repeat_simulate_demand(&self, starting_quantity: usize, count: usize) -> (usize, usize, usize, usize, f64, f64) {
        self.ocl_repeat_simulate_demand(starting_quantity, count).unwrap()
    }

}

// NB: These methods are not accessible from Python
impl Simulation {
    /// OpenCL implementation of repeat_simulate_demand
    /// 
    /// Includes a rewritten version of simulate_demand.
    /// There are several differences:
    /// 
    /// 1. I don't want the bulk of the computation to be generating a perfect zipf distribution
    ///    when I know we got that by eyeballing the curve anyway. So instead I generate a
    ///    pretty large sample and put up with a small period (of like 16M elements)
    /// 
    fn ocl_repeat_simulate_demand(&self, starting_quantity: usize, simulation_samples: usize) -> ocl::Result<(usize, usize, usize, usize, f64, f64)> {

        // This is the embedded opencl program we're going to run inside our app.
        // It may be easier to understand if you read this last

        // Think of this program queue as your connection to the device
        let pro_que = ProQue::builder()
            .src(include_str!("simulation.cl"))
            .dims(simulation_samples)
            .build()?;

        // These two are precomputed zipf distributions, to make sampling from these distributions
        // faster and simpler to implement. But it does spend CPU time before you start.
        let precomp_size = 16<<20;
        let job_lot_zipf_precomp         = precompute_zipf_buffer(&pro_que, 1000, self.job_lot_zipf,          precomp_size)?;
        let itemise_traffic_zipf_precomp = precompute_zipf_buffer(&pro_que, 1000, self.itemwise_traffic_zipf, precomp_size)?;

        // We also need to seed the simple uniform random number generator on ocl because it has no randomness of its own
        // So first we compute it on the CPU (the Host)
        let seed : Vec<u32> = (0..simulation_samples).into_iter().map(|_| rand::random()).collect();
        // Then send it to the device
        let seed = pro_que.buffer_builder::<u32>()
            .len(simulation_samples)
            .copy_host_slice(&seed[..])
            .build()?;

        // These four are the resulting statistics, to be filled in by the device
        let successful_transactions = pro_que.create_buffer::<u64>()?;
        let successful_sales        = pro_que.create_buffer::<u64>()?;
        let failed_transactions     = pro_que.create_buffer::<u64>()?;
        let failed_sales            = pro_que.create_buffer::<u64>()?;


        let kernel = pro_que.kernel_builder("ocl_simulate_demand")
            .arg(&seed)
            .arg(&job_lot_zipf_precomp)
            .arg(&itemise_traffic_zipf_precomp)
            .arg(&successful_transactions)
            .arg(&successful_sales)
            .arg(&failed_transactions)
            .arg(&failed_sales)
            .arg(starting_quantity)
            .arg(self.lead_time.min(10))
            .arg(self.safety_stock as i32)
            .arg(self.order_quantity as i32)
            .arg(precomp_size)
            .build()?;

        unsafe { kernel.enq()?; }

        // Copy the statistics back. It doesn't have to be this hard.
        // But I want to explain it all in detail because I figure you'll spend a lot of your time
        // doing exactly this.
        
        // I did it by making a single vector, which the closure will take control of (hence "move")
        let mut vec = vec![0u64; simulation_samples];
        let mut get_sum = move |buffer: &ocl::Buffer<u64>| -> ocl::Result<usize> {
            // This copies the device buffer into our host vector.
            buffer.read(&mut vec).enq()?;
            // This iterates over it and sums it into a u64.
            // It would be a good idea to keep it as u64 because - who knows - maybe we want to
            // sell more than 4 billion widgets. But they are purposely inconvenient to work with
            // because they are also inconvenient for some computers to work with and they will
            // slow you down on the GPU. Usize, however, is whichever size numbers your computer
            // naturally uses. So we convert it to that and ignore the possible tragedy. We'll
            // just show the max we can if we are limited. Good? No. But easy and maybe good enough
            Ok(vec.iter().copied().sum::<u64>().try_into().unwrap_or(::std::usize::MAX))
        };
        let st = get_sum(&successful_transactions)?;
        let ss = get_sum(&successful_sales)?;
        let ft = get_sum(&failed_transactions)?;
        let fs = get_sum(&failed_sales)?;

        Ok((st, ss, ft, fs,
            st as f64 / (st as f64 + ft as f64),
            ss as f64 / (ss as f64 + fs as f64)))
    }

}

/// Precompute some values for a zipf distribution
/// Used by Simuation but not intended to be visible to Python.
fn precompute_zipf_buffer(pro_que: &ProQue, num_elements: usize, exponent: f64, size: usize) -> ocl::Result<ocl::Buffer<u32>> {
    let z = zipf::ZipfDistribution::new(num_elements, exponent).unwrap();
    let mut rng = rand::thread_rng();
    let v: Vec<u32> = (0..size).into_iter().map(|_| z.sample(&mut rng) as u32).collect();
    pro_que.buffer_builder()
            .len(size)
            .copy_host_slice(&v[..])
            .build()
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn rustoclsim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Simulation>()?;

    Ok(())
}

#[test]
fn test_ocl() {
    let sim = Simulation {
        safety_stock: 10,
        lead_time: 10,
        order_quantity: 7,
        job_lot_zipf: 2.75,
        itemwise_traffic_zipf: 4.0,
    };
    sim.ocl_repeat_simulate_demand(10, 10000).expect("OCL Failed");
}