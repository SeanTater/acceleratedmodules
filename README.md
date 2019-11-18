Stock Simulation: An example of Python native modules
=====================================================

- [Stock Simulation: An example of Python native modules](#stock-simulation-an-example-of-python-native-modules)
- [Basic Implementation](#basic-implementation)
- [Cython Stock Simulation](#cython-stock-simulation)
- [Better Cython Simulation](#better-cython-simulation)
- [Rust Stock Simulation](#rust-stock-simulation)
    - [Cargo.toml](#cargotoml)
    - [lib.rs](#librs)
    - [Compile your new module](#compile-your-new-module)
- [Rust OpenCL Implementation](#rust-opencl-implementation)
  - [Overview: writing for separate devices](#overview-writing-for-separate-devices)
  - [Differences in the code](#differences-in-the-code)
    - [Precomputed zipf distributions](#precomputed-zipf-distributions)
    - [Other Differences](#other-differences)
  - [Highlights](#highlights)
  - [Installing it for production](#installing-it-for-production)
  - [Installing it for debugging](#installing-it-for-debugging)

Basic Implementation
====================

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
=======================
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

Better Cython Simulation
========================
If it's worth writing for Cython at all, then it's worth improving it to use more of Cython's power.

For starters, we should define the types of all our variables in each function we're going to use, so that Cython doesn't revert to interpreting:

```py
cpdef simulate_demand(self, int starting_quantity):
    cdef int successful_transactions = 0
    cdef int successful_sales = 0
    cdef int failed_transactions = 0
    cdef int failed_sales = 0
    cdef int stock = starting_quantity
    cdef list trucks = [0] * self.lead_time

    cdef unsigned int state = np.random.randint(1<<32)

    # Everything else defined later
    cdef int day, customer, request, short, orders

    for day in range(365):
            # A truck arrived
            stock += trucks[day % self.lead_time]
            # The rest is as normal
```

The class should be defined similarly, giving the types of everything we plan to use.
In order to do this we also have to include `cimport` for some types too. Namely, because we want to use `np.ndarray` as a type. 

```py
#!/usr/bin/env python3
import numpy as np
cimport numpy as np

cdef class Simulation:
    ''' Simulates the demand of one product over time ...
    '''
    cdef int safety_stock
    cdef int lead_time
    cdef int order_quantity
    cdef double job_lot_zipf
    cdef double itemwise_traffic_zipf
    cdef np.ndarray job_lot_zipf_precomp
    cdef np.ndarray itemwise_traffic_zipf_precomp
```

Similarly, you'll need to let the compiler know where numpy's headers are so it can import them. Numpy makes that pretty painless to do since it's the most popular library for Cython to compile against.

```py
#!/usr/bin/env python3
# setup.py
from distutils.core import setup
from Cython.Build import cythonize
import numpy # <- You need this in order to find the includes

setup(
    ext_modules = cythonize(
        "simulation.pyx",
        language_level=3  # <- This is a matter of taste I guess
    ),
    include_dirs=[numpy.get_include()] # <- This is the include you want
)
```

We also do the same precomputing we will later do (and explain more) in OpenCL. Note that it is possible to send pointers to integers in Cython like it is in C. You're required to treat it as an array however. So `*state = x` doesn't work.

```py
# Completely by-the-book reference implementation of xorshift
cdef unsigned int xorshift32(unsigned int* state):
    # Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs"
    cdef unsigned int x = state[0]
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    state[0] = x
    return x

# Select an item at random from a buffer
cdef unsigned int random_select(unsigned int* state, np.ndarray[np.int64_t, ndim=1] precomp):
    return precomp[xorshift32(state) % precomp.shape[0]]
```

The result is better than Python now. But contain your excitement a bit; it gets much better later.

Rust Stock Simulation
=====================

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


Rust OpenCL Implementation
==========================

You might think that since this derives from the Rust implementation, this would
look similar. But it couldn't be further from the truth. Still, to get it running
you should with all the same things you had for the bare Rust implementation.
But in this case you should also have a GPU available.
I'm not a world-class OpenCL expert so don't be too surprised that the speed is not
a huge improvement over the Rust implementation.

Overview: writing for separate devices
--------------------------------------
```
$ tree src
src
├── lib.rs
└── simulation.cl
```
You notice from the start that there are two files where we had one before.
This is because OpenCL, or any device language like this, works - differently.
Compiling and using applications like this look like the following:

1. Write your code (easy! 😜)
2. Compile it (using `maturin develop` or `maturin build` here)
3. Use it in your Python program, which is interpreted on-the-fly
4. When you run `Simulation::ocl_repeat_simulate_demand()`, that rust method
   will use a few C libraries to find the OpenCL implementation on your computer.
5. The same function passes the source code for `simulation.cl` to the OpenCL
   implementation and compiles it now
6. That finishes and gives a reference to the compiled OpenCL kernel.
7. Run it by giving it arguments and putting it in a queue.
8. When it completes, you `.enq()?` will return.

It's actually not that difficult to handle, you just need to remember what parts
are getting checked when you first write your code, and which are not checked
until it's first run. Everything on the device is not checked until runtime.

There's one more wrinkle. There are now more than one processor and more than one
memory on your computer. Each processor can only access it's own memory. So using
a device almost always looks like:

1. Format the data in the simplest format you can (like a few arrays of floats)
   Make sure there are no links to anything outside of that data.
2. Choose an `N` to count to. The device function (the "kernel") will be called once
   for each `i in 0..N`, and the only thing to differentiate the kernel is its `i`
3. Copy all the input arrays to buffers on the device.
4. Create output arrays for the buffer to fill with any results.
   Keep in mind you can't change the size afterward.
5. Call the kernel, passing it pointers to the copied arrays in it's own memory,
   and passing any scalars verbatim (which is easy).
6. The kernel does all its work in memory, modifying its own arrays.
   It's called once for each element in the work dimension you specify.
7. Repeat steps 4-5 as much as possible to get your mileage out of the copies
8. Copy all the output buffers

```
Host                          Device
┌───────────────────────────┰─────────────────────
│ Input                     ┃ 
│                           ┃ 
│ Format ───────────> Copy arrays ──┐
│                           ┃       │ 
│ Call kernel               ┃       │ Create output arrays
│                           ┃       │
├────────> Send pointers & scalars ─┤
│                           ┃       │
│ wait..                    ┃       │ Run kernel
│                           ┃       │
│ Done.  <── Notify ────────╂───────┤
│                           ┃       │
│<─── Copy output arrays ───╂───────┘
│                           ┃       
│ Profit?                   ┃
```

Differences in the code
-----------------------

There are many places things look different, and usually for good reason

### Precomputed zipf distributions

```rs
struct Simulation {
    ...
    job_lot_zipf_precomp: Vec<u32>,
    itemwise_traffic_zipf_precomp: Vec<u32>,
}

/// Precompute some values for a zipf distribution
/// Used by Simulation but not intended to be visible to Python.
fn precompute_zipf_buffer(num_elements: usize, exponent: f64) -> Vec<u32> {
    ...
}

/// Inside `Simulation::ocl_repeat_simulate_demand()`, we copy those buffers
let job_lot_zipf_precomp = pro_que.buffer_builder()
    .len(self.job_lot_zipf_precomp.len())
    .copy_host_slice(&self.job_lot_zipf_precomp[..])
    .build()?;
let itemwise_traffic_zipf_precomp = ... ditto;

```
And later in OpenCL, we uniformly sample elements from these distributions.

```c
// Completely by-the-book reference implementation of xorshift
uint xorshift32(uint* state)
{
    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Select an item at random from a buffer
uint random_select(uint* state, __global uint* precomp, uint len) {
    return precomp[xorshift32(state) % len];
}
```
But in turn it so happens that the device doesn't have random sources
so we have to pass it something to get it started.
```rs
// We also need to seed the simple uniform random number generator on ocl because it has no randomness of its own
// So first we compute it on the CPU (the Host)
let seed : Vec<u32> = (0..chunk_count).into_iter().map(|_| rand::random()).collect();
// Then send it to the device
let seed = ... same old copy;
```

### Other Differences

- I made a sorta-arbitrary limitation that we will only track 10 trucks so that way we can keep it entirely in the kernel. You can put in any (smallish) number here. I just don't want to create a buffer and send it to the kernel just for it's intermediate scratch space.
- It also made sense to have CL run multiple simulations at a time since then there's even less to copy
- But you still want to have at least a thousand or a few thousand separate iterations

 OpenCL's part                | Rust's part                     | Why
------------------------------|---------------------------------|-----
 `uint trucks[10];`           | `.arg(self.lead_time.min(10))`  | Limit excess copying
 `for (uint sample=0; ...)`   | `chunk_size = samples / 1000;`  | Reduce copying to/from device
 `int me = get_global_id(0);` | `let chunk_count = 1000;`       | Balance workload across many cores

Highlights
----------
Running an OpenCL kernel is always considered unsafe because you're essentially compiling and running *anything* there. For that reason the API developers marked it as unsafe, and rust emphasizes this to you by making you separate it from the rest in a block like so:
```rs
unsafe { kernel.enq()?; }
```

Installing it for production
----------------------------
Technically speaking, there's nothing else you need to do to compile it - sorta. But the issue is that you probably want to run it, and to do that you need an OpenCL environment. On Macbooks it generally just works. For Linux and Windows you need to install the appropriate driver for your hardware.

Installing it for debugging
---------------------------
Thankfully, there are already several drivers available that use only the CPU, and will be able to test whether your code works without driving you nuts install drivers. [POCL] is probably your first choice, and you can find most of your options under [IWOCL].
For Debian Linux we should be able to install it using:
```sh
apt install pocl-opencl-icd
```
and you'll find that's already done if you use a recent Docker image. 😉

[POCL]: http://portablecl.org/
[IWOCL]: https://www.iwocl.org/resources/opencl-implementations/