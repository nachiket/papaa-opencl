#define IMAGE_HEIGHT   (28)
#define IMAGE_WIDTH    (28)
#define NO_COMPUTE_UNITS (2)
#define FILTER_SIZE    (3)

__kernel  __attribute__ ((reqd_work_group_size(IMAGE_WIDTH, (IMAGE_HEIGHT+NO_COMPUTE_UNITS-1)/NO_COMPUTE_UNITS, 1)))
void conv_2d(
    __global float *in,               // W*H input images
    __constant float *filt,           // K*K filter kernel
    __global float *out,              // W*H output images
    const float pBias)                // constant offset/bias
{
    __local float image_buff[IMAGE_WIDTH * ((IMAGE_HEIGHT+NO_COMPUTE_UNITS-1)/NO_COMPUTE_UNITS + FILTER_SIZE-1)] 
		__attribute__((xcl_array_partition(cyclic, 3, 1)));
	// store filter coefficients in the register file for faster access.
    __local float local_filt[FILTER_SIZE * FILTER_SIZE] __attribute__ ((xcl_array_partition(complete, 1)));

    __attribute__((xcl_pipeline_workitems)) {
        int x = get_local_id(0);
        int y = get_local_id(1);
        int row = get_global_id(1);
        if(x < FILTER_SIZE*FILTER_SIZE) {
            local_filt[x] = filt[x];
        }
        image_buff[y * IMAGE_WIDTH + x] = in[row * IMAGE_WIDTH + x];
		// need to transfer FILTER_SIZE-1 extra rows. The work items corresponding to last FILTER_SIZE-1
		// rows take responsibility to transfer this extra rows.
        if(y > (get_local_size(1) - FILTER_SIZE)) {
			image_buff[(y+FILTER_SIZE-1)*IMAGE_WIDTH + x] = in[(row+FILTER_SIZE-1)*IMAGE_WIDTH + x];
		}
    }

    // wait for all work items to copy their share as each work item
    // requires 3x3 neighbor instead of single pixel
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
        
    // loop over rows
    __attribute__((xcl_pipeline_workitems)) {
        int i = get_local_id(0);
        int j = get_local_id(1);
		int row = get_global_id(1);
        for (int r = 0; r < FILTER_SIZE; r++) 
        {
            for(int c = 0; c < FILTER_SIZE; c++)
            {
                sum += local_filt[r * FILTER_SIZE + c]*image_buff[(j + r) * IMAGE_WIDTH + i + c];
            }
        }
        out[row * IMAGE_WIDTH + i] = sum + pBias;
    }
}

