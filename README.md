Stock Simulation: An example of Python native modules
=====================================================

Basic Implementation
--------------------

The first method is written exactly as it sounds in the definition, with no
flair or pizazz to improve performance, and mostly for legibility.

It boils down to just a few parts:
```py
class Simulation:
    def __init__(self, ...):
        # Several parameters about stocking levels, customer demand, and supply chain
        pass

    def simulate_demand(self, starting_quantity):
        # Some counters

        for day in range(365):
            # A truck arrived
            # This many customers arrive
            for customer in ...:
                # This customer wants this many
                request = ...
                if stock >= request:
                    # There are enough (+ tick some counters)
                    stock -= request
                else:
                    # There are not enough (also tick counters)
            # The day is over. Start making orders.
            if stock < self.safety_stock:
                # Send off an order
        return counters
    
    def repeat_simulate_demand(self, starting_quantity, count=10000):
        # More counters
        for _ in range(count):
            yada_yada = self.simulate_demand(starting_quantity)
            # tick counters
        return counters
    

if __name__ == "__main__":
    # kick off the process
```


Cython Stock Simulation
-----------------------
Like several other versions, this is the lest modified version that would still run.
In the case of Cython, that is byte-for-byte identical so there are no changes at all. This is
really not the best case for Cython, because it can be improved a lot with some type restrictions.
That said, it's easy to get started. You include a build script for `setup.py`, which is minimal
in this case but to build correctly, you will need a name, dependencies, and other metadata.
We'll skip those things for this example, and skip right to building a basic package:

```sh
python setup.py build_ext --inplace
```

Then to use it, you import it like nothing really changed.

```py
import simulation
simulation.Simulation(2, 3, 2).repeat_simulate_demand(10, 10000)
```

Sadly, it's no improvement over bare Python as it stands. To get more out of it, you need to bend
a little. If you look at the resulting `simulation.c`, you can imagine why the result is not faster. There are tons of corner cases Python handles for you that make it convenient for you as a developer and a user but waste a ton of CPU cycles.

Rust Stock Simulation
---------------------
Unlike Cython, Rust does require some rewriting to get started. In most cases the differences
are essentially just syntactical, but zipf distributions are not built-in or part of any grand
numpy-like package so that did require importing two more packages, `rand` and `zipf`.

It's not terribly difficult to get started. Here are the highlights

### Cargo.toml
This build configuation file is similar to `setup.py` in Cython
After some metadata about the compiler version and names, we specify what packages we want

```toml
[dependencies]
rand = "^0.7"
zipf = "^6.1"
```

Then we're asking it to build a library by this name, and we want a C-compatible library.
(The options were a rust-only library or an independent binary. You don't need to specify
anything for those cases, they are inferred.)

```toml
[lib]
name = "rustsim"
crate-type = ["cdylib"]
```

This is the extended form of dependency specification, used when you want to specify options.
Here you're asking for an optional feature from Pyo3.

```
[dependencies.pyo3]
version = "0.8.2"
features = ["extension-module"]
```

### lib.rs
Let's walk through the whole library. It starts with uses, analogous to imports in Python.
Everything in `std` and `core` come for free. The others are dependencies.

```rs
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::cmp::max;
use rand::distributions::Distribution;
```

In rust, classes are split into two parts: the data structure and the methods.
This is the data structure half, and you define each type you want here.
The first line tells Pyo3 to expose this as a Python class as part of the module `rustsim`.

```rs
#[pyclass(module = "rustsim")]
struct Simulation {
    safety_stock: usize,
    lead_time: usize,
    order_quantity: usize,
    job_lot_zipf: f64,
    itemwise_traffic_zipf: f64,
}
```

This is the other half of the class: the methods. There's good reason, but the simplest is that
in Rust you can define methods once against many mostly-unrelated structures.
Also, in the first line here We're telling Pyo3 to expose everything inside as methods.

```rs
#[pymethods]
impl Simulation {
    ...
}
```

This is your constructor. Usually Rust constructors are super simple but in this case we have an
extra empty Python object that we fill with stuff. `obj` here is the same object as `self` is in
`def __init__(self):`

```rs
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
```

Basically nothing changed in the algorithm but some semicolons and `let mut`. So skipping to the highlights:

```rs
fn simulate_demand_inner(&self, starting_quantity: usize) -> (usize, usize, usize, usize, f64, f64) {
    // yada yada

    // We create a random number generator and then use it several times.
    let mut rng = rand::thread_rng();
    // These two distributions represent static formulas that will later be passed uniform random numbers
    // so they don't need to be mutable. (The formula doesn't change when you use it.)
    // Unwrap here ignores the possibility of an error. (It will crash in that case).
    // It's not hard to handle that case but it adds some extra noise.
    let jl_zipf = zipf::ZipfDistribution::new(1000, self.job_lot_zipf).unwrap();
    let it_zipf = zipf::ZipfDistribution::new(1000, self.itemwise_traffic_zipf).unwrap();

    for day in 0..365 {
        // bla bla
        if stock < self.safety_stock {
            // yada yada

            // OK here's something interesting
            // We had to add here rather than subtract 1.
            //     vvvvvvvvvvvvvvvvvvvvvv
            trucks[(day+self.lead_time-1) % self.lead_time] = orders * self.order_quantity;
            // That's because we asked for a unsigned integer and it can underflow.
        }
    }
    (   // return here is implicit.
        
        // Rust enforces that floats and integers stay separate, so we have to tediously convert to floats
        successful_transactions as f64 / (successful_transactions as f64 + failed_transactions as f64),
        successful_sales as f64 / (successful_sales as f64 + failed_sales as f64)
    )
}
```

If you were going to do this the right way, you should use a method similar to this one, where
you actually expose the possiblity for failure (a PyResult). In this case this method would crash
anyway because the inner method didn't handle it right. But if you went this route you wouldn't
do that; you'd implement it from the start with error handling.

We accept a structure containing a reference to the global Python object `py: Python<'_>`,
this is handled entirely by Pyo3, but it means you will only be able to call this function from
python. In exchange, you can do lots of things with the interpreter. We use it here just to convert
some stuff using `into_py()`

The `-> PyResult<PyObject>` part means the function will return a `PyResult` that contains a
`PyObject`. It may or may not have an exception inside it, and any `PyObject` is possible.
Compare that with `-> (usize, usize, usize, usize, f64, f64)` where we committed to exactly what
we will return.

At the end, we use `.into_py(py)` to convert our object into a `PyObject`. Then we wrap it in
`Ok( ... )`, which is the success version of `Result()` (and `PyResult` is a subtype of `Result`).
If it had failed, we would have returned `Err( ... )` instead. In most cases you'll see this done
implicitly using the `?` operator, which bails the current function if the left-side is an `Err`.

```rs
/// You can also perform the conversions manually, and you can get access to the Python GIL, which necessary in many cases
fn simulate_demand(&self, py: Python<'_>, starting_quantity: usize) -> PyResult<PyObject> {
    Ok(self.simulate_demand_inner(starting_quantity).into_py(py))
}
```

Almost there, I promise! `fn repeat_simulate_demand` is pretty boring so we'll skip to the module.
Here we pull it together and create a module around our class. The most important part here is that
the name of this function is the name of your module, and **it must match your library name**.
This requirement is because Python looks for a function by that name in the file.

```rs
/// This module is a python module implemented in Rust.
#[pymodule]
fn rustsim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Simulation>()?;

    Ok(())
}
```

Done.

### Compile your new module
Usually you'd want to do this with `maturin`, which is easy:

```sh
maturin develop # Compiles and installs in the current virtual env
maturin develop --release # Slower compile, 10x faster result
```

