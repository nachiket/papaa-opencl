create_solution -name mnist -dir . -force

# Target a Xilinx FPGA board
add_device -vbnv xilinx:adm-pcie-7v3:1ddr:3.0

set host_args "-x bin_mnist.xclbin -m sample -i ../../../../../../../imgs/mnist_test_img_0.pgm"

# Host Compiler Flags
set_property -name host_cflags -value "-g -O0 -std=c++0x -I$::env(PWD)" -objects [current_solution]

add_files "host_mnist_fpga.c"
add_files "lenet5_model.c"
add_files "load_kernel.c"
add_files "pgm.h"
add_files "lenet5_model.h"
set_property file_type "c header files" [get_files "pgm.h"]
set_property file_type "c header files" [get_files "lenet5_model.h"]

# Kernel Definition
create_kernel filter3D -type clc
create_kernel maxpool3D -type clc
create_kernel iplayer -type clc
create_kernel relu_layer -type clc
create_kernel softmax -type clc
add_files -kernel [get_kernels filter3D] "src/kernels.cl"
add_files -kernel [get_kernels maxpool3D] "src/kernels.cl"
add_files -kernel [get_kernels iplayer] "src/kernels.cl"
add_files -kernel [get_kernels relu_layer] "src/kernels.cl"
add_files -kernel [get_kernels softmax] "src/kernels.cl"

# Define Binary Containers
create_opencl_binary bin_mnist
set_property region "OCL_REGION_0" [get_opencl_binary bin_mnist]
create_compute_unit -opencl_binary [get_opencl_binary bin_mnist] -kernel [get_kernels filter3D] -name filter3D
create_compute_unit -opencl_binary [get_opencl_binary bin_mnist] -kernel [get_kernels maxpool3D] -name maxpool3D
create_compute_unit -opencl_binary [get_opencl_binary bin_mnist] -kernel [get_kernels iplayer] -name iplayer
create_compute_unit -opencl_binary [get_opencl_binary bin_mnist] -kernel [get_kernels relu_layer] -name relu_layer
create_compute_unit -opencl_binary [get_opencl_binary bin_mnist] -kernel [get_kernels softmax] -name softmax

# Compile the design for CPU based emulation
compile_emulation -flow cpu
puts "Compiled for CPU emulation..."
run_emulation -flow cpu -args $host_args

# Create estimated resource usage and latency report
report_estimate

# Compile the design for Hardware Emulation
#compile_emulation -flow hardware
#run_emulation -flow hardware -args $args

# Compile the design for execution on the FPGA board
#build_system

# Create the board deployment package for the application
#package_system

