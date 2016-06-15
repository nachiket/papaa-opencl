__kernel void filter2D(
	const __global short* pInput, 
	__constant short* pFilter, 
	__global short* pOutput, 
	const int nInWidth,
	const int nFilterWidth) 
{
	const int xOut = get_global_id(0); 
	const int yOut = get_global_id(1);
	long sum = 0;
	int c = 0; 
	for (int r = 0; r <5; r++) 
	{ 
		const int idxFtmp = r * 5;
		const int idxIntmp = ((yOut+r) * nInWidth) + xOut;
			
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
	} 
	pOutput[(yOut*nInWidth)+xOut] = sum/25; 
}

__kernel void maxpool(__global  short* const src,  __global short* dst)
{
    const char stride = 2;
    const char k_w = 2, k_h =2;
    char i,j;
    short _max=0;

    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    const unsigned int width = get_global_size(0);
    unsigned int index;

    for(i=0;i<k_h;i++)
    {
      index = (x*k_w + (i+ (y*k_h))*width);
      for(j=0;j<k_w;j++)
        {
          if(_max < src[(index+j)])
          {
            _max = src[(index+j)];
          }
        }
    }

    dst[(x*(width/stride))+y] = _max;
}

__kernel void relu(__global  short* const buffer)
{
    const int x = get_global_id(0);
    if(buffer[x]<0)
        buffer[x]=0;
}
