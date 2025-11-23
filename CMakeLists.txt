cmake_minimum_required(VERSION 3.10)
project(ParallelAutoencoder)

set(CMAKE_CXX_STANDARD 14)

# Include directories
include_directories(include)

# Source files
set(COMMON_SOURCES
    src/common/cifar10_loader.cpp
)

set(CPU_SOURCES
    src/cpu/layers.cpp
    src/cpu/autoencoder.cpp
    src/cpu/main_cpu.cpp
)

# Executable for CPU Phase
add_executable(cpu_train ${COMMON_SOURCES} ${CPU_SOURCES})
