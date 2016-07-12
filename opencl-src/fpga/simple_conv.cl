__kernel  __attribute__ ((reqd_work_group_size(1, 1, 1)))
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
        float sum = 0;
        int c = 0;

        // loop over rows
        // prevent from read across the image boundary
        if((x < W-K+1)&&  (y < H-K+1)) {
                for (int r = 0; r < K; r++) 
                {
                        for(c = 0; c < K; c++)
                        {
                                sum += filt[r*K+c]*in[y*W+r*W+x+c];
                        }
                }
                out[y*W+x] = sum + pBias;
        }
}

