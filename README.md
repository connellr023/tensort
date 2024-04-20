# clipsort
> A CLI tool that utilizes the ResNet18 AI model to recognize content in images and sort them.

![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

<br />

#### Development Environment
The pretrained model used for this project can be found <a href="https://github.com/LaurentMazare/tch-rs/releases/download/mw/resnet18.ot">here</a>.

Setting up the development environment can be done by following the **README** from the **tch-rs** repository <a href="https://github.com/LaurentMazare/tch-rs">here</a>

When installing *libtorch*, ensure that the version that supports **CUDA** is used.

On Linux, `.bashrc` should contain the following (at least for my setup)
```
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=/path/to/libtorch:$LD_LIBRARY_PATH
```