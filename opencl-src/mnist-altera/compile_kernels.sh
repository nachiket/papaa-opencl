echo $1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/altera/16.0/quartus/dspba/backend/linux64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu

if [ "$1" = "hw" ]; then
	echo "Compiling kernels to generate hardware"
	aoc  device/cnn_kernels.cl -o bin/cnn_kernels.aocx -v --board de5net_a7
else
	echo "Compiling kernels for emulation"
	aoc -march=emulator device/cnn_kernels.cl -o bin/cnn_kernels.aocx --board de5net_a7
fi
