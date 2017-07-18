include sources.mk
CC=gcc
CFLAGS= -g -O3
LIBS= OpenCL
INCS=$(wildcard *.h*)
LIB_DIRS= /opt/intel/opencl
INC_DIRS= /opt/intel/opencl/include

exec=lab2_app

all:$(exec)

$(exec):$(C_SRCS)
	@echo Building....
	$(CC) $(CFLAGS) $(C_SRCS) -o $(exec) $(INCS) \
	$(foreach D,$(INC_DIRS),-I$D) $(foreach D,$(LIB_DIRS),-L$D) \
	$(foreach L,$(LIBS),-l$L)

run:$(exec)
	./$(exec) ../../../imgs/mnist_test_img_0.pgm
	
pyrun:$(exec)
	python host_app.py ../../../imgs/mnist_test_img_0.pgm

print:
	@echo "C_SRCS = $(C_SRCS)"
	@echo "C_FLAGS = $(CFLAGS)"

.phony:clean
clean:
	rm -rf $(exec)
	rm -rf *.pgm