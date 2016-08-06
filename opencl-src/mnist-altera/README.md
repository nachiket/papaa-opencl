# Description
This project is the FPGA implementation of MNIST digit classification. The project uses Altera OpenCL for high-level synthesis and hecen requires the AOCL SDK  and Quartus tools for compilation and execution.

# To run
- Setup Altera specific environment variables.
- The kernels need to compiled before compiling host application. The kernel compilation can be done for emulation mode or hardware mode(takes hours to compile).
  * To compile for emulation mode, run _./compile_kernels.sh_
  * To compile full hardware btistream, run _./compile_kernels.sh hw_
- The above compilation should produce _*.aocx_ file inside _./bin_ directory.
- To compile host application, do _make all_
- To run the application in the emulation mode, do _make emu_
- To run the application on the FPGA, do _make run_

You need to setup the Altera FPGA board before compiling. Refer to [Altera getting started guide](https://www.altera.com/content/dam/altera-www/global/en_US/pdfs/literature/hb/opencl-sdk/aocl_getting_started.pdf) or [AOCL programming guide](https://www.altera.com/en_US/pdfs/literature/hb/opencl-sdk/aocl_programming_guide.pdf) to setup the board and the environments required for compilation.
