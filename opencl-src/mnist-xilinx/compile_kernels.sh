
echo $1

export PATH=$PATH:/opt/Xilinx/SDAccel/2016.1/bin/
export LC_ALL="en_US.UTF-8"

mkdir -p bin

if [ "$1" = "hw" ]; then
	echo "Compiling kernels to generate hardware"
	xocc -t=hw device/cnn_kernels.cl -o bin/cnn_kernels.xclbin --xdevice xilinx:adm-pcie-7v3:1ddr:3.0 
else
	echo "Compiling kernels for emulation"
	xocc -t sw_emu device/cnn_kernels.cl -o bin/cnn_kernels.xclbin --xdevice xilinx:adm-pcie-7v3:1ddr:3.0 
fi
