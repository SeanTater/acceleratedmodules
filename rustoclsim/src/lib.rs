use pyo3::prelude::*;
use rand::distributions::Distribution;
use std::convert::TryInto;
use ocl::ProQue;
use failure::Fallible;

/// Simulation parameters
/// 
/// The idea is these are things that would stay the same across invocations
/// 
/// Simulation has two natures. It lives in the Python world and has an impl accessible there.
/// It also lives in the Rust world. Different methods are used here too. We need that so that
/// it is easier to test it.
#[pyclass(module = "rustsim")]
struct Simulation {
    safety_stock: usize,
    lead_time: usize,
    order_quantity: usize,
    job_lot_zipf_precomp: Vec<u32>,
    itemwise_traffic_zipf_precomp: Vec<u32>,
}

/// Simulation implementation
/// 
/// The following methods are all available from Python
#[pymethods]
impl Simulation {
    /// Implementation of python Simulation.__init__() (just wraps rust Simulation::new())
    #[new]
    fn init(
        obj: &PyRawObject,
        safety_stock: usize,
        lead_time: usize,
        order_quantity: usize,
        job_lot_zipf: Option<f64>,
        itemwise_traffic_zipf: Option<f64>,
    ) {
        obj.init(Simulation::new(
            safety_stock,
            lead_time,
            order_quantity,
            job_lot_zipf,
            itemwise_traffic_zipf
        ));
    }

    /// Calls the appropriate OpenCL function
    fn repeat_simulate_demand(&self, starting_quantity: usize, count: usize) -> (usize, usize, usize, usize, f64, f64) {
        self.ocl_repeat_simulate_demand(starting_quantity, count).unwrap()
    }

}

/// Simulation Implementation, continued
/// 
/// This group doesn't mention pymethods, and isn't visible from Python
impl Simulation {
    fn new(
        safety_stock: usize,
        lead_time: usize,
        order_quantity: usize,
        job_lot_zipf: Option<f64>,
        itemwise_traffic_zipf: Option<f64>,
    ) -> Simulation {
        let job_lot_zipf = job_lot_zipf.unwrap_or(2.75);
        let itemwise_traffic_zipf = itemwise_traffic_zipf.unwrap_or(4.0);
        Simulation {
            safety_stock,
            lead_time,
            order_quantity,
            job_lot_zipf_precomp: precompute_zipf_buffer(1000, job_lot_zipf),
            itemwise_traffic_zipf_precomp: precompute_zipf_buffer(1000, itemwise_traffic_zipf)
        }
    }

    /// OpenCL implementation of repeat_simulate_demand
    /// There are several differences:
    /// 
    /// 1. I don't want the bulk of the computation to be generating a perfect zipf distribution
    ///    when I know we got that by eyeballing the curve anyway. So instead I generate a
    ///    pretty large sample and put up with a small period (of like 16M elements)
    /// 
    /// 2. The source code for the inner simulation in OpenCL is in simulation.cl. We read it
    ///    into this program at compile time. using include_str!(filename)
    /// 
    fn ocl_repeat_simulate_demand(&self, starting_quantity: usize, simulation_samples: usize) -> Fallible<(usize, usize, usize, usize, f64, f64)> {
        let chunk_size = simulation_samples / 1000;
        let chunk_count = 1000;

        // Think of this program queue as your connection to the device
        let pro_que = ProQue::builder()
            .src(include_str!("simulation.cl"))
            .dims(chunk_count)
            .build()?;

        // These two are precomputed zipf distributions, to make sampling from these distributions
        // faster and simpler to implement. A lot of the latency comes from precomputing these
        // so in an ideal world you may do this in opencl too.
        let job_lot_zipf_precomp = pro_que.buffer_builder()
            .len(self.job_lot_zipf_precomp.len())
            .copy_host_slice(&self.job_lot_zipf_precomp[..])
            .build()?;
        let itemwise_traffic_zipf_precomp = pro_que.buffer_builder()
            .len(self.itemwise_traffic_zipf_precomp.len())
            .copy_host_slice(&self.itemwise_traffic_zipf_precomp[..])
            .build()?;

        // We also need to seed the simple uniform random number generator on ocl because it has no randomness of its own
        // So first we compute it on the CPU (the Host)
        let seed : Vec<u32> = (0..chunk_count).into_iter().map(|_| rand::random()).collect();
        // Then send it to the device
        let seed = pro_que.buffer_builder::<u32>()
            .len(chunk_count)
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
            .arg(&itemwise_traffic_zipf_precomp)
            .arg(&successful_transactions)
            .arg(&successful_sales)
            .arg(&failed_transactions)
            .arg(&failed_sales)
            .arg(starting_quantity)
            .arg(self.lead_time.min(10))
            .arg(self.safety_stock as i32)
            .arg(self.order_quantity as i32)
            .arg(self.itemwise_traffic_zipf_precomp.len())
            .arg(chunk_size)
            .build()?;

        unsafe { kernel.enq()?; }

        // Copy the statistics back. It doesn't have to be this hard.
        // But I want to explain it all in detail because I figure you'll spend a lot of your time
        // doing exactly this.
        
        // I did it by making a single vector, which the closure will take control of (hence "move")
        let mut vec = vec![0u64; chunk_count];
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
/// Used by Simulation but not intended to be visible to Python.
fn precompute_zipf_buffer(num_elements: usize, exponent: f64) -> Vec<u32> {
    let z = zipf::ZipfDistribution::new(num_elements, exponent).unwrap();
    let mut rng = rand::thread_rng();
    (0..(16 << 20)).into_iter().map(|_| z.sample(&mut rng) as u32).collect()
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