/*************************************************************************
	> File Name: gputimer.h
	> Author: 
	> Mail: 
	> Created Time: Fri Apr  3 02:28:15 2020
 ************************************************************************/

#ifndef _GPUTIMER_H
#define _GPUTIMER_H

struct GpuTimer 
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer(){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer(){
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start(){
        cudaEventRecord(start,0);
    }

    void Stop(){
        cudaEventRecord(stop,0);
    }

    float Elapsed(){
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed,start,stop);
        return elapsed;
    }
};

#endif
