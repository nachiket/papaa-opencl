__kernel void filter3D(
	const __global float * pInput, 
	__global float * pFilter, 
	__global float * pOutput, 
	const int nFilterWidth,
	const int nFilterHeight,
	const int nInMaps,
        __global const float * pBias) 
{
	const int x = get_global_id(0); 
	const int y = get_global_id(1);
	const int z = get_global_id(2);

        const int ImWidth  = get_global_size(0);
        const int ImHeight = get_global_size(1);
	
	float sum = 0;
	int c = 0;
	int idxFstart = z*nFilterHeight*nFilterWidth*nInMaps;
/*
	if((get_global_id(0)==0) && (get_global_id(1)==0) && (get_global_id(2)==0))
	 printf("%d %d %d \n", get_num_groups(0),get_num_groups(1),get_num_groups(2));
	if((get_global_id(0)==0) && (get_global_id(1)==0) && (get_global_id(2)==18))
	{
	  for(int i=0;i<28*28;i++)
		printf("%f,",pInput[i]);
	  printf("---------->>>>>>>------\n\n\n");
	}
	if((get_global_id(0)==0) &&( get_global_id(1)==0) && (get_global_id(2)==0))
	{
	 for(int j =0; j < 20; j++)
	 {
	   for(int i=0; i<nInMaps*nFilterHeight*nFilterWidth; i++)
	   {
		printf("%f,",pFilter[j*nFilterHeight*nFilterWidth*nInMaps+i]);
	   }
	   printf("\n");
	 }
	 printf("\n \n \n");
	 for(int j=0;j<20;j++)
	  printf("%f",pBias[j]);
	printf("\n");
	}
	//printf("%d %d %d %p \n",x,y,z, &pOutput[((z*ImHeight*ImWidth)+(y*ImWidth)+x)]);
*/
	for(int maps = 0; maps<nInMaps; maps++)
	{ 
		for (int r = 0; r <nFilterHeight; r++) 
		{ 
			const int idxFtmp = idxFstart + (maps*nFilterHeight + r) * nFilterWidth; 
			const int idxIntmp = (((maps*ImHeight) + y + r) * ImWidth) + x;
			for(c = 0; c <nFilterWidth; c++)
			{
				const int idxF = idxFtmp + c;
				const int idxIn = idxIntmp + c;
				sum += pFilter[idxF]*pInput[idxIn];
			}
		}
	}
	pOutput[((z*ImHeight*ImWidth)+(y*ImWidth)+x)] = sum + pBias[z];
}


__kernel void filter3D_unroll(
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
	const int z = get_global_id(2);

        const int ImWidth  = get_global_size(0);
        const int ImHeight = get_global_size(1);

        float sum = 0;
        int c = 0;

	int idxFtmp = z*nFilterHeight*nFilterWidth*nInMaps;

        for(int maps = 0; maps<nInMaps; maps++)
        {
             for (int r = 0; r <nFilterHeight; r++) 
             {
                int idxF = idxFtmp + ((maps*nFilterHeight + r) * nFilterWidth) + c; 
                int idxIn = ((((maps*ImHeight) + y + r) * ImWidth) + x) + c;
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
              }
        }
        pOutput[(y*ImWidth)+x] = sum + *pBias;
}
