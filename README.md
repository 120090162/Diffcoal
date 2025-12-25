# Diffcoal

# Environment Setup
- verfied at platform x64 ubuntu20/22
```bash
conda create -n diffcoal python=3.10 -c conda-forge -y
conda activate diffcoal
# require the gcc/g++ version is equal to the host compilers
# this we assume the cuda is compiled with gcc/g++ 11.4
conda install -c conda-forge coal eigen boost eigenpy numpy gcc_linux-64=11 gxx_linux-64=11 cmake make git pkg-config pytorch=2.4.0 open3d -y

git clone --recursive https://github.com/120090162/Diffcoal.git

# 安装diffcoal库
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DTorch_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_INTERFACE=OFF \
    -DPYTHON_EXECUTABLE=$(which python)

make -j4
make install

# [optinal] boost test
conda install -c conda-forge fmt
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_INTERFACE=OFF \
    -DBUILD_TEST_CASES=ON \
    -DPYTHON_EXECUTABLE=$(which python)

make -j4
ctest
```

# 测试例子

```bash

```