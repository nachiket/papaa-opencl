create_solution -name simple_conv -dir . -force

# Target a Xilinx FPGA board
add_device -vbnv xilinx:adm-pcie-7v3:1ddr:3.0


# Host Compiler Flags
set_property -name host_cflags -value "-g -O0 -std=c++0x -I$::env(PWD)" -objects [current_solution]

# Host source files
add_files "host_app.c"

# Kernel Definition
create_kernel conv_2d -type clc
add_files -kernel [get_kernels conv_2d] "simple.cl"


# Define Binary Containers
create_opencl_binary bin_conv2d
set_property region "OCL_REGION_0" [get_opencl_binary bin_conv2d]
create_compute_unit -opencl_binary [get_opencl_binary bin_conv2d] -kernel [get_kernels conv_2d] -name conv0

# Compile the design for CPU based emulation
compile_emulation -flow cpu
run_emulation -flow cpu -args "bin_conv2d.xclbin"

# Create estimated resource usage and latency report
report_estimate

# Compile the design for Hardware Emulation
#compile_emulation -flow hardware
#run_emulation -flow hardware -args $args

# Compile the design for execution on the FPGA board
#build_system

# Create the board deployment package for the application
#package_system

