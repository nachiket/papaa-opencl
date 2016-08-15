__kernel void filter3D(
        const __global float * pInput,
        const __global float * pFilter,
        __global float * pOutput,
        const int nFilterWidth,
        const int nFilterHeight,
        const int nInMaps,
        const __global float * pBias)
{
        const int x = get_global_id(0); 
        const int y = get_global_id(1);
        const int z = get_global_id(2);

        const int OWidth  = get_global_size(0);
        const int OHeight = get_global_size(1);
        const int ImWidth = OWidth+nFilterWidth-1;
        const int ImHeight = OHeight+nFilterHeight-1;   
        float4 sum4 = 0;
        int idxFstart = z*nFilterHeight*nFilterWidth*nInMaps;

           for(int maps = 0; maps<nInMaps; maps++)
           {
                for (int r = 0; r <nFilterHeight; r++) 
                {
                        const int idxFtmp = idxFstart + (maps*nFilterHeight + r) * nFilterWidth; 
                        const int idxIntmp = (((maps*ImHeight) + y + r) * ImWidth) + x;
        		int c = 0;
			int c4 = 0;
			while(c <= nFilterWidth-4)
			{
				float4 filter4 = vload4(c4,pFilter+idxFtmp);
				float4 in4 = vload4(c4, pInput +idxIntmp);
				sum4 += in4 * filter4;
				c +=4;
				c4++;
			}
                        for(int c1 = c; c1 <nFilterWidth; c1++)
                        {
                                const int idxF = idxFtmp + c1;
                                const int idxIn = idxIntmp + c1;
                                sum4.x += pFilter[idxF]*pInput[idxIn];
                        }
                }
           }
           pOutput[((z*OHeight*OWidth)+(y*OWidth)+x)] = sum4.x + sum4.y + sum4.z + sum4.w + pBias[z];
}

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

        const int oWidth  = get_global_size(0);
        const int oHeight = get_global_size(1);

	float maxval = -3.402823e+37;
	int hstart = y*nStride;
	int wstart = x*nStride;
	int hend = min(hstart+nPoolsize, iHeight);
	int wend = min(wstart+nPoolsize, iWidth);
	for(unsigned int r = hstart; r < hend; r++) {
		for(unsigned int c = wstart; c < wend; c++) {
			unsigned int idx = z*iHeight*iWidth + r * iWidth + c;
			maxval = fmax(maxval, pInput[idx]);
		}
	}
/*
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
                }
        }
*/
        pOutput[(((z*oHeight)+y)*oWidth)+x] = maxval;
}


__kernel void iplayer(
        const __global float * pInput,
        const __global float * pWeights,
        __global float * pOutput,
        const int nInputs,
        const __global float * pBias)
{
        const int x = get_global_id(0);
	const int idxstart = x*nInputs;
        float sum = 0;
        for (int i = 0; i <nInputs; i++) 
        {
           sum += pWeights[idxstart+i]*pInput[i];
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


