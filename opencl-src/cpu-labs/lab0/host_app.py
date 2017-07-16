import pyopencl as cl
import numpy as np
from scipy.signal import convolve2d
import sys
from PIL import Image
from timeit import default_timer as tm



if len(sys.argv) is not 2:
	print("Usage: %s <image_name.pgm>" % sys.argv[0])
	sys.exit()


# Setup Problem
FILTER_SIZE = np.int32(3)
DTYPE = np.float32

lap_filter = np.ones(FILTER_SIZE**2, dtype=DTYPE)*-1.
lap_filter[4] = 8.
bias = DTYPE(0.01)

input_pgm = Image.open(sys.argv[1])
input_pgm_array = np.array(list(input_pgm.getdata()), dtype=DTYPE) # Row-Major flattened array
input_pgm_array /= 255.0
padded_input = np.zeros(((input_pgm.width+FILTER_SIZE-1)*(input_pgm.height+FILTER_SIZE-1)))
padded_input[:input_pgm_array.size] = input_pgm_array # this looks like the padding done in c?

print("Host: Input image resolution: %dx%d" % (input_pgm.width, input_pgm.height))

# Setup OpenCL
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
prg = cl.Program(ctx, open('conv_kernel.cl').read()).build()

mf = cl.mem_flags

padded_img_size = DTYPE(1).itemsize*(input_pgm.width+FILTER_SIZE-1)*(input_pgm.height+FILTER_SIZE-1)
d_image = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, padded_img_size, padded_input)
d_filter = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, lap_filter.nbytes, lap_filter)
d_output = cl.Buffer(ctx, mf.WRITE_ONLY, input_pgm_array.nbytes)

print("Launching the Kernel...")
start = tm()
prg.conv_2d(queue, (input_pgm.width, input_pgm.height), (1, 1),
			d_image, d_filter, d_output, FILTER_SIZE, bias)
end = tm()
elapsed_time = end - start

h_output = np.empty_like(input_pgm_array)
cl.enqueue_copy(queue, h_output, d_output)

# Validation
input_pgm_array = input_pgm_array.reshape((input_pgm.height, input_pgm.width))
h_output = h_output.reshape(input_pgm_array.shape)
lap_filter = lap_filter.reshape((FILTER_SIZE, FILTER_SIZE))

ref_output = convolve2d(input_pgm_array, lap_filter, 'same')

if np.allclose(h_output, ref_output):
	print("INFO: ****TEST PASSED****")
else:
	print("INFO: TEST FAILED !!!!")

print("Kernel runtime = %0.3f us" % (elapsed_time*1e6))

# still need to normalize/save the output
# failing right now