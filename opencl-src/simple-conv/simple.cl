__kernel void convolve(
	__global float *in, 		// W*H input images
	__constant float *filt, 	// K*K filter kernel
	__global float *out, 		// W*H output images
	const int K,				// filter resolution
	const float pBias) 			// constant offset/bias
{
	// get pixel position
	const int W = get_global_size(0);
	const int H = get_global_size(1);
	
	// get image resolution
	const int x = get_global_id(0); 
	const int y = get_global_id(1);
	//printf("OCL :%d\n", K);
	float sum = 0;
	int c = 0;

	// loop over rows
	// prevent from read across the image boundary
	if((x < W-K+1)&&  (y < H-K+1)) {
		for (int r = 0; r < K; r++) 
		{ 
			// loop over columns
			for(c = 0; c < K; c++)
			{
				//sum += filt[r*K+c]*in[((y+r)*W+x)+c];
				sum += filt[r*K+c]*in[y*W+r*W+x+c];
			}
		}
		out[y*W+x] = sum + pBias;
	}
}
