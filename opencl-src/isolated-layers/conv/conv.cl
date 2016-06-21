__kernel void filter2D(
	const __global float * pInput, 
	__constant float * pFilter, 
	__global float * pOutput, 
	const int nFilterWidth,
	const int nFilterHeight,
	const int nInMaps,
        __global const float * pBias) 
{
	const int x = get_global_id(0); 
	const int y = get_global_id(1);
//        const int z = get_global_id(2);
        const int ImWidth  = get_global_size(0);
        const int ImHeight = get_global_size(1);
//	if((get_local_id(0) == 1) && (get_local_id(1) == 1))
//	printf("%d %d %d \n",ImWidth,ImHeight,get_global_size(2));
//	printf("%d %d %d\n",get_num_groups(0),get_num_groups(1),get_num_groups(2));
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
	pOutput[(y*ImWidth)+x] = sum + *pBias;
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
