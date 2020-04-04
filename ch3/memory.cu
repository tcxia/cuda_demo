#include<stdio.h>

__global__ void use_local_memory_GPU(float in){
    // variable "f" is in local memory and private to each thread 
    float f;

    // parameter "in" is in local memory and private to each thread 
    f = in;

    // ... real code would presumably do other stuff here ... 
}

__global__ void use_global_memory_GPU(float *array){
    // "array" is a pointer into global memory on the device
    array[threadIdx.x] = 2.0f * (float) threadIdx.x;
}


__global__ void use_shared_memory_GPU(float *array){

    // local variable, private to each thread 
    int i, index = threadIdx.x;
    float average, sum = 0.0f;

    // __shared__ variable are visiable to all threads in the thread block
    // and have the same lifetime as the thread block 
    __shared__ float sh_arr[128];

    // copy data from "array" in global memory to sh_arr in shared memory
    // here, each thread is responsible for copying a single element.
    sh_arr[index] = array[index];

    // ensure all the writes to shared memory have completed
    __syncthreads();

    // now, sh_arr is fully populated. let's find the average of all previous elements.
    for(i = 0;i<index;i++){
        sum += sh_arr[i];
    }

    average = sum / (index + 1.0f);

    // if array[index] is greater than the average of array[0...index-1],replace with average.
    if (array[index] > average){
        array[index] = average;
    }

    sh_arr[index] = 3.14;

}


int main(){
    
    use_local_memory_GPU<<<1,128>>>(2.0f);

    float h_arr[128];
    float *d_arr;

    use_local_memory_GPU<<<1,128>>>(d_arr);

    cudaMemcpy((void*)h_arr,(void*)d_arr,sizeof(float)*128,cudaMemcpyHostToDevice);

    return 0;
}
