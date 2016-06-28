__kernel void maxpool3D(
	const __global float * pInput, 
	__global float * pOutput, 
	const int PoolWidth,
	const int PoolHeight,
	const int ImWidth,
	const int ImHeight,
	const int Hstride,
	const int Vstride)
{
	const int x = get_global_id(0); 
	const int y = get_global_id(1);
	const int z = get_global_id(2);
	const int xidx = Hstride*x;
	const int yidx = Vstride*y;
	const int oWidth  = ImWidth/Hstride;
	const int oHeight = ImHeight/Vstride;
	float maxval =0.0;
	for (int r = 0; r <PoolHeight; r++) 
	{ 
		const int idxIntmp = (((z*ImHeight) + yidx + r) * ImWidth) + xidx;
		for(int c = 0; c <PoolWidth; c++)
		{
			const int idxIn = idxIntmp + c;
			maxval = fmax(maxval,pInput[idxIn]); 
		}
	}
	pOutput[(((z*oHeight)+y)*oWidth)+x] = maxval;
}

__kernel void maxpool2D(
	const __global float * pInput, 
	__global float * pOutput, 
	const int PoolWidth,
	const int PoolHeight,
	const int ImWidth,
	const int ImHeight,
	const int Hstride,
	const int Vstride)
{
	const int x = get_global_id(0); 
	const int y = get_global_id(1);
	const int xidx = Hstride*x;
	const int yidx = Vstride*y;
	float maxval =0.0;
	for (int r = 0; r <PoolHeight; r++) 
	{ 
		const int idxIntmp = ((yidx + r) * ImWidth) + xidx;
		for(int c = 0; c <PoolWidth; c++)
		{
			const int idxIn = idxIntmp + c;
			maxval = fmax(maxval,pInput[idxIn]); 
		}
	}
	pOutput[(y*ImWidth)+x] = maxval;
}
