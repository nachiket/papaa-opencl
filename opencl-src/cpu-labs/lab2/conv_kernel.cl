__kernel void conv_2d(
	__global float *in, 		// W*H input images
	__constant float *filt, 	// K*K filter kernel
	__global float *out, 		// W*H output images
	const int K,				// filter resolution
	const float bias) 			// constant offset/bias
{
	const int W = get_global_size(0);

	// this work item computes output at (x,y) in the output image
	const int x = get_global_id(0); 
	const int y = get_global_id(1);

	float2 sum2 = 0;
	float2 filter2;
	float2 in2;
	// loop over rows
	for (int r = 0; r < K; r++) 
	{ 
		int c = 0;
		int c2 = 0;
		while (c <= K - 2) { 
			filter2 = vload2(c2, filt + r*K);
			in2 = vload2(c2, in + (r+y)*(W+K-1)+x);
			sum2 += in2 * filter2; 
			c += 2;
			c2++;   
		}
		for(; c < K; c++)
		{
			sum2.x += filt[r * K + c] * in[(y + r) * (W+K-1) + x + c];
		}
	}
	out[y * W + x] = sum2.x + sum2.y + bias;
}

