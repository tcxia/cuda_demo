#include <stdio.h>

__global__ void global_scan(float* d_out, float* d_in){
    int idx = threadIdx.x;
    float out = 0.00f;
    d_out[idx] = d_in[idx];
    __syncthreads();

    for(int interpre=1; interpre < sizeof(d_in);interpre*=2){
        if (idx - interpre >= 0){
            out = d_out[idx] + d_out[idx - interpre];
        }
        __syncthreads();

        if (idx - interpre >= 0){
            d_out[idx] = out;
            out = 0.00f;
        } 
    }
}


__global__ void shmem_scan(float* d_out, float* d_in){
    int idx = threadIdx.x;
    float out = 0.00f;

    extern __shared__ float sdata[];

    sdata[idx] = d_in[idx];

    for (int interpre=1; interpre < sizeof(d_in); interpre*=2){
        if (idx - interpre >= 0){
            out = sdata[idx] + sdata[idx - interpre];
        }

        __syncthreads();

        if (idx - interpre >= 0){
            sdata[idx] = out;
            out = 0.00f;
        }
    }
    d_out[idx] = sdata[idx];
}

int main(int argc, char** argv){
    const int ARRAY_SIZE = 8;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    float h_in[ARRAY_SIZE];
    for(int i=0;i<ARRAY_SIZE;i++){
        h_in[i] = float(i);
    }

    float h_out[ARRAY_SIZE];

    float* d_out;
    float* d_in;

    cudaMalloc((void**) &d_out,ARRAY_BYTES);
    cudaMalloc((void**) &d_in,ARRAY_BYTES);

    cudaMemcpy(d_in,h_in,ARRAY_BYTES,cudaMemcpyHostToDevice);

    // global memory to launch kernel 
    //global_scan<<<1,ARRAY_SIZE>>>(d_out,d_in);

    // share memory to launch kernel
    shmem_scan<<<1,ARRAY_SIZE,ARRAY_SIZE*sizeof(float)>>>(d_out,d_in);

    cudaMemcpy(h_out,d_out,ARRAY_SIZE,cudaMemcpyDeviceToHost);

    for(int i = 0;i<ARRAY_SIZE;i++){
        printf("%f",h_out[i]);
        printf(((i%4)!=3) ? "\t":"\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
