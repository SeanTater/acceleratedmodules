FROM debian
RUN apt-get update \
    && apt-get install -y git wget python3-pip python3-venv  \
    && python3 -m venv ~/venv \
    && (echo source ~/venv/bin/activate >> ~/.bashrc ) \
    && pip3 install maturin cython numpy \
    && wget https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init \
    && chmod +x ./rustup-init \
    && ./rustup-init -y --default-toolchain nightly
