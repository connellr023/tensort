# Dockerfile for tensort continuous integration
# Author: Connell Reffo

# Use a Rust base image with Cargo pre-installed
FROM rust:latest

# Set the working directory inside the container
WORKDIR /usr/src/tensort

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    unzip

# Download and extract libtorch
RUN wget --timeout=600 https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcu121.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121.zip -d /usr/lib/ && \
    rm libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121.zip

# Download ResNet34 training weights
RUN wget --timeout=300 https://github.com/LaurentMazare/tch-rs/releases/download/mw/resnet34.ot

# Set environment variables for libtorch
ENV LIBTORCH /usr/lib/libtorch
ENV LD_LIBRARY_PATH $LIBTORCH/lib:$LD_LIBRARY_PATH
ENV LIBTORCH_BYPASS_VERSION_CHECK=1

# Copy project files to the container
COPY src ./src
COPY Cargo.toml Cargo.lock ./

# Run tests
RUN cargo test
