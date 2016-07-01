__kernel void maxpool3D(
	const __global float * pInput, 
	__global float * pOutput,
	const int iWidth,
	const int iHeight, 
	const int nPoolsize,
	const int nStride)
{
	const int x = get_global_id(0); 
	const int y = get_global_id(1);
	const int z = get_global_id(2);
	
	const oWidth  = get_global_size(0);
	const oHeight = get_global_size(1);

	const int xidx = nStride*x;
	const int yidx = nStride*y;
	float maxval =0.0;
	for (int r = 0; r <nPoolsize; r++) 
	{ 
		const int idxIntmp = (((z*iHeight) + yidx + r) * iWidth) + xidx;
		for(int c = 0; c <nPoolsize; c++)
		{
			const int idxIn = idxIntmp + c;
			maxval = fmax(maxval,pInput[idxIn]);
//			if(pInput[idxIn]>maxval)
//				maxval = pInput[idxIn];
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
