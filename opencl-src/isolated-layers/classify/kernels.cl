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
/*	if(get_global_id(0)==0)
	{
	  for(int i=0; i < nInputs ; i++ )
		printf("%d %f,",i, pWeights[i]);
	}*/
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

