FROM debian
RUN apt-get update \
    && apt-get install -y git wget python3-pip python3-venv pocl-opencl-icd opencl-headers ocl-icd-opencl-dev python3-dev cpio \
    && python3 -m venv ~/venv \
    && (echo source ~/venv/bin/activate >> ~/.bashrc ) \
    && source ~/venv/bin/activate \
    && pip3 install maturin cython numpy \
    && wget https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init \
    && chmod +x ./rustup-init \
    && ./rustup-init -y --default-toolchain nightly
CMD echo Great, now this will stay open for a while and you can poke around in VSCode. ; sleep 365d
