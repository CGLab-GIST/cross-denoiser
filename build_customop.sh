#!/bin/bash

# Check if no arguments were provided
if [ $# -eq 0 ]; then
    echo "No build target specified. Build all targets: 'temporal', 'reproject', 'regression'"
    args=("temporal" "reproject" "regression")  # Default values
else
    args=("$@")  # If arguments are provided, use them instead
fi

CUDA_INCLUDE_PATH=/usr/local/cuda/include

# Read TensorFlow compile and link flags from Python output
while IFS= read -r line; do
    # First line will be compile flags, second line will be link flags
    if [[ -z "${TF_CFLAGS+x}" ]]; then
        TF_CFLAGS=($line)  # Assign first line to TF_CFLAGS
    else
        TF_LFLAGS=($line)  # Assign second line to TF_LFLAGS
    fi
done < <(python -c 'import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags())); print(" ".join(tf.sysconfig.get_link_flags()))')

log_and_check_errors() {
    local log_file=$1
    if grep -iE "error|failed|fatal" "$log_file"; then
        echo "Errors detected during compilation of :" $log_file
    else
        echo "Compilation successful, remove log file: $log_file"
        # rm "$log_file"
    fi
}

build_module() {
    local base_name=$1
    local extra_nvcc_flags=$2
    local extra_gpp_flags=$3
    local additional_files=$4  # Additional source files for g++

    # Construct file paths based on the base name
    local source_file="ops/${base_name}.cu.cc"
    local object_file="ops/${base_name}.cu.o"
    local output_file="ops/${base_name}.so"
    local log_file="${source_file}_build.log"

    echo "Building ${source_file}..."

    {
        # Pre-compilation command (e.g., modifying CUDA flags in source files)
        sed -i '/#define GOOGLE_CUDA/c\/\/ #define GOOGLE_CUDA' "$source_file"
        sed -i '/#define __CUDACC__/c\/\/ #define __CUDACC__' "$source_file"

        # Compilation commands
        nvcc -std=c++14 -c -o "$object_file" "$source_file" ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -expt-relaxed-constexpr ${extra_nvcc_flags} -arch=native
        g++ -std=c++14 -shared -o "$output_file" $additional_files "$object_file" ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} ${extra_gpp_flags}

        # Post-compilation command (e.g., reverting CUDA flags modifications)
        sed -i 's~// #define GOOGLE_CUDA$~#define GOOGLE_CUDA~' "$source_file"
        sed -i 's~// #define __CUDACC__$~#define __CUDACC__~' "$source_file"
    } &> "$log_file"

    log_and_check_errors "$log_file"
}

# Function to build a target in parallel
build_in_background() {
    local target=$1
    case $target in
        temporal)
            build_module "temporal" "" "" &
            ;;
        reproject)
            build_module "reproject" "" "-I${CUDA_INCLUDE_PATH}" "ops/reproject.cc" &
            ;;
        regression)
            build_module "regression" "" "-I${CUDA_INCLUDE_PATH}" "ops/regression.cc" &
            ;;
        # Add more cases for other arguments as needed
        *)
            echo "Warning: Unknown build option '$target'"
            ;;
    esac
}

# Function to handle Ctrl+C (SIGINT)
handle_interrupt() {
    # Kill all background jobs
    jobs -p | xargs kill
    wait # Ensure all background jobs are finished before exiting
    echo "All builds have been terminated."
    exit 1
}

# Trap Ctrl+C (SIGINT) and call the handler
trap handle_interrupt SIGINT

# Loop through all arguments for conditional builds
for arg in "${args[@]}"; do
    build_in_background "$arg"
done

# Wait for all background jobs to complete
wait
