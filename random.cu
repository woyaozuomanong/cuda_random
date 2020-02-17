#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include "curand_kernel.h" 

#include <stdio.h>
#include <sys/time.h>


double cpuSecond()
{
#if _WIN32
	_LARGE_INTEGER time_start;    /*开始时间*/
	double dqFreq;                /*计时器频率*/
	LARGE_INTEGER f;            /*计时器频率*/
	QueryPerformanceFrequency(&f);
	dqFreq = (double)f.QuadPart;
	QueryPerformanceCounter(&time_start);
	return time_start.QuadPart / dqFreq ;//单位为秒，精度为1000 000/（cpu主频）微秒
#endif

#if __linux__
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
#endif
}



__global__ void kernel_random(float *dev_random_array,int height,int width,long clock_for_rand)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x<0 || x>width || y<0 || y>height)
    {
        return;
    }

    int pos = y*width + x;

    curandState state;
    curand_init(pos,pos,0,&state);
    dev_random_array[pos] = abs(curand_uniform(&state));
}


int main()
{
    double iStart,iElapse;
    iStart=cpuSecond();
    const int array_size_width = 10;
    const int array_size_height = 10;
    float random_array[array_size_width*array_size_height];
    for(int i=0;i<array_size_width*array_size_height;i++)
    {
        random_array[i] = 0;
    }

    //error status
    cudaError_t cuda_status;

    //only chose one GPU
    cuda_status = cudaSetDevice(0);
    if(cuda_status != cudaSuccess)
    {
        fprintf(stderr,"cudaSetDevice failed! Do you have a CUDA-Capable GPU installed?");
        return 0;
    }

    float *dev_random_array;

     //allocate memory on the GPU
    cuda_status = cudaMalloc((void**)&dev_random_array,sizeof(float)*array_size_width*array_size_height);
    if(cuda_status != cudaSuccess)
    {
        fprintf(stderr,"dev_reference_image cudaMalloc Failed");
        exit( EXIT_FAILURE );
    }


    dim3 threads(16,16);
    dim3 grid(max(array_size_width/threads.x,1),max(array_size_height/threads.y,1));

    long clock_for_rand = clock();
    printf("clock=%d\n",clock_for_rand);
    kernel_random<<<grid,threads>>>(dev_random_array,array_size_width,array_size_height,clock_for_rand);

    //copy out the result
    cuda_status = cudaMemcpy(random_array,dev_random_array,sizeof(float)*array_size_width*array_size_height,cudaMemcpyDeviceToHost);//dev_depthMap
    if(cuda_status != cudaSuccess)
    {
        fprintf(stderr,"cudaMemcpy Failed");
        exit( EXIT_FAILURE );
    }

     for(int i=0;i<array_size_width*array_size_height;i++)
     {
         printf("%f\n",random_array[i]);
     }

    iElapse=cpuSecond()-iStart;
    printf("Total time: %f\n",iElapse);
    //free
    cudaFree(dev_random_array);
    return 0;
}
