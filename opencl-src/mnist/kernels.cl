__kernel void filter3D(
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
	int idxFstart = z*nFilterHeight*nFilterWidth*nInMaps;
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
                        //maxval = fmax(maxval,pInput[idxIn]);
                        if(pInput[idxIn]>maxval)
                                maxval = pInput[idxIn];
                }
        }
        pOutput[(((z*oHeight)+y)*oWidth)+x] = maxval;
}

__kernel void iplayer(
        const __global float * pInput,
        __global float * pWeights,
        __global float * pOutput,
        const int nInputs,
        __global const float * pBias)
{
        const int x = get_global_id(0);
        float sum = 0;
        for (int i = 0; i <nInputs; i++) 
        {
           sum += pWeights[(x*nInputs)+i]*pInput[i];
        }
        pOutput[x] = sum + pBias[x];
}

__kernel void relu_layer (__global float * pData)
{
        const int x = get_global_id(0);
        float zero = 0.0;
        pData[x] = fmax(zero,pData[x]);
}




__kernel void softmax(
        __global float * pdata)
{
        __local float sum, temp[10];
        const int x = get_local_id(0);
        temp[x] = exp(pdata[x]);

        barrier(CLK_LOCAL_MEM_FENCE);
        if(get_local_id(0)==0)
        {
          for(int i=0; i< get_local_size(0); i++)
                sum += temp[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        pdata[x] = temp[x]/sum; 
}


