__kernel void filter2D(
	const __global unsigned char * pInput, 
	__constant float * pFilter, 
	__global float * pOutput, 
	const int nFilterWidth,
	const int nFilterHeight,
	const int nInMaps,
       __global	const float * pBias) 
{
	const int x = get_global_id(0); 
	const int y = get_global_id(1);
        const int z = get_global_id(2);

        const int ImWidth  = get_global_size(0);
        const int ImHeight = get_global_size(1);
	float sum = 0;
	int c = 0;
	for(int maps = 0; maps<nInMaps; maps++)
	{ 
		for (int r = 0; r <nFilterHeight; r++) 
		{ 
			const int idxFtmp = (maps*nFilterHeight + r) * nFilterWidth; 
			const int idxIntmp = (((maps*ImHeight) + y + r) * ImWidth) + x;
			for(c = 0; c <nFilterWidth; c++)
			{
				const int idxF = idxFtmp + c;
				const int idxIn = idxIntmp + c;
				sum += pFilter[idxF]*pInput[idxIn];
			}
		}
	}
	pOutput[((z*ImHeight + y)*ImWidth)+x] = sum + pBias[z];
}
#if 0			
		int idxF = idxFtmp + c; 
		int idxIn = idxIntmp + c; 
		sum += pFilter[idxF]*pInput[idxIn]; 
		idxF++; 
		idxIn++; 
		sum += pFilter[idxF]*pInput[idxIn]; 
		idxF++; 
		idxIn++; 
		sum += pFilter[idxF]*pInput[idxIn]; 
		idxF++; 
		idxIn++; 
		sum += pFilter[idxF]*pInput[idxIn];
		idxF++;
		idxIn++;
		sum += pFilter[idxF]*pInput[idxIn];
		c += 5; 
#endif
