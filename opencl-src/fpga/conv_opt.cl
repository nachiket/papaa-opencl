__kernel  __attribute__ ((reqd_work_group_size(4, 4, 1)))
void conv_2d(
        __global float *in,               // W*H input images
        __constant float *filt,   // K*K filter kernel
        __global float *out,              // W*H output images
        //const int K,                          // filter resolution
        const float pBias)                        // constant offset/bias
{
        // get pixel position
        int W = get_global_size(0);
        int H = get_global_size(1);
        int K = 3;      
        // get image resolution
        int x = get_global_id(0); 
        int y = get_global_id(1);
	__local float local_ker[9];
	int i = get_local_id(0);
 	int j = get_local_id(1);
	if(i == 0 && j == 0) {
		//async_work_group_copy(&local_ker[0], filt, 9, 0);
                for(int i = 0; i < 9; i++)
			local_ker[i] = filt[i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

        float sum = 0;
        int c = 0;

        // loop over rows
        // prevent from read across the image boundary
        if((x < W-K+1)&&  (y < H-K+1)) {
                for (int r = 0; r < K; r++) 
                {
                        for(c = 0; c < K; c++)
                        {
                                sum += local_ker[r*K+c]*in[y*W+r*W+x+c];
                        }
                }
                out[y*W+x] = sum + pBias;
        }
}

__kernel  __attribute__ ((reqd_work_group_size(1, 1, 1)))
void conv_2d_unroll(
        __global float *in,               // W*H input images
        __constant float *filt,   // K*K filter kernel
        __global float *out,              // W*H output images
        //const int K,                          // filter resolution
        const float pBias)                        // constant offset/bias
{
        // get pixel position
        int W = get_global_size(0);
        int H = get_global_size(1);
        int K = 3;      
        // get image resolution
        int x = get_global_id(0); 
        int y = get_global_id(1);

        float sum = 0;
        int c = 0;

        // loop over rows
        // prevent from read across the image boundary
        if((x < W-K+1)&&  (y < H-K+1)) {
		__attribute__((opencl_unroll_hint))
                for (int r = 0; r < K; r++) 
                {
			__attribute__((opencl_unroll_hint))
                        for(c = 0; c < K; c++)
                        {
                                sum += filt[r*K+c]*in[y*W+r*W+x+c];
                        }
                }
                out[y*W+x] = sum + pBias;
        }
}

__kernel  __attribute__ ((reqd_work_group_size(1, 1, 1)))
void conv_2d_loop_pipeline(
        __global float *in,           	// W*H input images
        __constant float *filt,   	// K*K filter kernel
        __global float *out,            // W*H output images
        const float pBias)              // constant offset/bias
{
        // get pixel position
        int W = get_global_size(0);
        int H = get_global_size(1);
        int K = 3;      
        // get image resolution
        int x = get_global_id(0); 
        int y = get_global_id(1);

        float sum = 0;
        int c = 0;
	float pix, coeff;
        // loop over rows
        // prevent from read across the image boundary
        if((x < W-K+1)&&  (y < H-K+1)) {
                for (int r = 0; r < K; r++) 
                {
			__attribute__((xcl_pipeline_loop))
                        for(c = 0; c < K; c++)
                        {	pix = in[y*W+r*W+x+c];	// Load the pixel into private memory.
				coeff = filt[r*K+c];	// Load the filter coefficient into private memory.
                                sum += coeff*pix;	// MAC is supported in DSP. So 1 cycle for this?
                        }
                }
                out[y*W+x] = sum + pBias;
        }
}

