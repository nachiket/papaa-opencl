# Available Labs
#=======================================================================================
#	Lab #		|	Description
#---------------------------------------------------------------------------------------
#	1			| Simple 2D convolution with no optimization.
#---------------------------------------------------------------------------------------
#	2			| 2D convolution with loop unrolling.
#---------------------------------------------------------------------------------------
#	3			| 2D convolution with loop pipelining.
#---------------------------------------------------------------------------------------
#	4			| 2D convolution with line buffering.
#---------------------------------------------------------------------------------------

# Set the lab number here 
set lab_no 4

# project name
create_solution -name conv_lab -dir . -force

# Target a Xilinx FPGA board
add_device -vbnv xilinx:adm-pcie-7v3:1ddr:3.0

# arguments to the host application
set host_args "bin_conv2d.xclbin"

# Host Compiler Flags
set_property -name host_cflags -value "-g -O0 -std=c++0x -I$::env(PWD)" -objects [current_solution]

# choose the OpenCL kernel based on the lab number.
switch $lab_no {
	1 {
		set ker_name conv_2d
	}
	2 {
		set ker_name conv_2d_unroll
	}
	3 {
		set ker_name conv_2d_loop_pipeline
	}
	4 {
		set ker_name conv_2d_linebuff
	}
	default {
		puts "!!!!!Invalid Lab Number!!!!!"
		exit
	}
}

# add one more host app argument
lappend host_args $ker_name

# Host source files
add_files "fpga_opencl_lab.c"
add_files "pgm.h"
set_property file_type "c header files" [get_files "pgm.h"]

# Kernel Definition
create_kernel $ker_name -type clc
add_files -kernel [get_kernels $ker_name] "conv_opt.cl"


# Define Binary Containers
create_opencl_binary bin_conv2d
set_property region "OCL_REGION_0" [get_opencl_binary bin_conv2d]
create_compute_unit -opencl_binary [get_opencl_binary bin_conv2d] -kernel [get_kernels $ker_name] -name conv0

#set_property max_memory_ports true [get_kernels $ker_name]
# Compile the design for CPU based emulation
compile_emulation -flow cpu
puts "Comipled for CPU emulation..."
run_emulation -flow cpu -args $host_args

# Create estimated resource usage and latency report
#report_estimate

# All the below stuffs very slow !!!!!!!!!!
# Compile the design for Hardware Emulation
#compile_emulation -flow hardware
#run_emulation -flow hardware -args $args

# Compile the design for execution on the FPGA board
#build_system

# Create the board deployment package for the application
#package_system

