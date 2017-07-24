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
NUM_WORK_GROUPS = np.int32(4)
FILTER_SIZE = np.int32(3)
DTYPE = np.float32

lap_filter = np.ones(FILTER_SIZE**2, dtype=DTYPE)*-1.
lap_filter[4] = DTYPE(8.)
bias = DTYPE(0.05)

input_pgm = Image.open(sys.argv[1])
PADDED_WIDTH = input_pgm.width+FILTER_SIZE-1
PADDED_HEIGHT = input_pgm.height+FILTER_SIZE-1

h_image = np.array(list(input_pgm.getdata()), dtype=DTYPE) # Row-Major flattened array
h_image /= DTYPE(255.0)
h_image_padded = np.zeros(((PADDED_WIDTH)*(PADDED_HEIGHT)), dtype=DTYPE)
for i in range(input_pgm.height): # for each row
    h_image_padded[i*(PADDED_WIDTH):i*(PADDED_WIDTH)+input_pgm.width] = h_image[i*input_pgm.width:(i+1)*input_pgm.width] 
print("Host: Input image resolution: %dx%d" % (input_pgm.width, input_pgm.height))

ref_output = np.empty_like(h_image, dtype=DTYPE)

# Validation (gross c way, but don't feel like debugging right now)
for row in range(input_pgm.height):
    for col in range(input_pgm.width):
        ref_output[row*(input_pgm.width) + col] = 0

for row in range(input_pgm.height):
    for col in range(input_pgm.width):
        tmp = DTYPE(0)
        for kr in range(FILTER_SIZE):
            for kc in range(FILTER_SIZE):
                tmp += DTYPE(lap_filter[kr*FILTER_SIZE + kc] * h_image_padded[(col+kr)*(input_pgm.width+FILTER_SIZE-1) + row + kc])
        ref_output[col*(input_pgm.width) + row] = tmp + bias

# Setup OpenCL
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
prg = cl.Program(ctx, open('conv_kernel.cl').read()).build()

mf = cl.mem_flags

padded_img_size = DTYPE(1).itemsize*(PADDED_WIDTH)*(PADDED_HEIGHT)
d_image = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, padded_img_size, h_image_padded)
d_filter = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, lap_filter.nbytes, lap_filter)
d_output = cl.Buffer(ctx, mf.WRITE_ONLY, h_image.nbytes)

globalsize = (input_pgm.width, input_pgm.height)
localsize = (input_pgm.width, input_pgm.height/NUM_WORK_GROUPS)
local_buf_size = DTYPE(1).itemsize*(localsize[0]+FILTER_SIZE-1)*(localsize[1]+FILTER_SIZE-1)

print("Launching the Kernel...")
k_event = prg.conv_local(queue, globalsize, localsize, d_image, d_filter, d_output,
                         FILTER_SIZE, FILTER_SIZE, bias, cl.LocalMemory(local_buf_size))
k_event.wait()
elapsed_time = k_event.profile.end - k_event.profile.start

h_output = np.zeros_like(h_image, dtype=DTYPE)
cl.enqueue_copy(queue, h_output, d_output)

ref_output = np.zeros_like(h_output, dtype=DTYPE)

# Validation (gross c way, but don't feel like debugging right now)
for row in range(input_pgm.height):
    for col in range(input_pgm.width):
        tmp = DTYPE(0)
        for kr in range(FILTER_SIZE):
            for kc in range(FILTER_SIZE):
                tmp += DTYPE(lap_filter[kr*FILTER_SIZE + kc] * h_image_padded[(row+kr)*(PADDED_WIDTH) + col + kc])
        ref_output[row*input_pgm.width + col] = tmp + bias

if np.allclose(h_output, ref_output):
    print("INFO: ****TEST PASSED****")
else:
    print("INFO: TEST FAILED !!!!")
    print((h_output-ref_output).reshape((28,28)))

print("Kernel runtime = %0.3f us" % (elapsed_time*1e-3))

h_output = (255.0*(h_output - h_output.min())) / (h_output.max() - h_output.min())
h_output = h_output.reshape((input_pgm.width,input_pgm.height)).astype(np.uint8)
im_out = Image.fromarray(h_output)
im_out.save('ocl_output_python.pgm', 'ppm')
