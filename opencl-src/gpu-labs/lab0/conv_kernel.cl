#define FILTER_SIZE 5
__kernel void conv_local(
    __global float *in,               // W*H input images
    __constant float *filt,           // K*K filter kernel
    __global float *out,              // W*H output images
    const int nFilterWidth,
    const int nFilterHeight,
	const float bias,
    __local float * image_buff)                // constant offset/bias
{

    int x = get_local_id(0);
    int y = get_local_id(1);

    int row = get_global_id(1);

    const int ImWidth  = get_global_size(0);
    const int ImHeight = get_global_size(1);

/*    __local float local_filt[ FILTER_SIZE* FILTER_SIZE];
    if(x < nFilterWidth*nFilterHeight)
    {
	local_filt[x] = filt[x];
    }
*/
    image_buff[y * ImWidth + x] = in[row * ImWidth + x];
    if(y > (get_local_size(1) - nFilterHeight))
    {
    	image_buff[(y+nFilterHeight-1)*ImWidth + x] = in[(row+nFilterHeight-1)*ImWidth + x];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
    for (int r = 0; r < nFilterHeight; r++) 
    {
        for(int c = 0; c < nFilterWidth; c++)
        {
            sum += filt[r*nFilterWidth + c]*image_buff[(y + r) * ImWidth + x + c];
        }
    }
    out[row * ImWidth + x] = sum + bias;
}

