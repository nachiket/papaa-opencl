__kernel void convolve(
	const __global float *in, 		// W*H input images
	__constant float *filt, 		// K*K filter kernel
	__global float *out, 			// W*H output images
	const int K,				// filter resolution
        const float pBias) 			// constant offset/bias
{
	// get pixel position
        const int W = get_global_size(0);
        const int H = get_global_size(1);
	
	// get image resolution
	const int x = get_global_id(0); 
	const int y = get_global_id(1);

	float4 sum = 0;

	// loop over rows
	for (int r = 0; r < K; r++) 
	{ 
		// loop over columns
		for(int c = 0,c4 = 0; c < K, c4<ceil(K/4); c+=4,c4++)
		{
			float4 filt4 = vload4(c4,filt[r*K]);
			float4 in4 = vload4(c4,in[(y+r)*W+x]);
			sum += filt4*in4;
		}
		// for the odd last element..
	}
	out[y*W+x] = sum.x + sum.y + sum.z + sum.w + pBias;
}
