/* Merge Sort with CUDA Dynamic Parallelism */
/* compile with `nvcc -arch=compute_35 -rdc=true main.cu` */

#include <cuda.h>
#include <stdio.h>
#include "gputimer.h"

#define B 256

void fill_array(float x[], int n, bool random=false) {
    for (int i = 0; i < n; i += 1) {
        if (random) x[i] = 1.0 * rand() / RAND_MAX;
        else x[i] = (float) (n - i);
    }
}

int comp(const void *a, const void *b) {
    float *x = (float *) a;
    float *y = (float *) b;
    if (*x < *y) return -1;
    if (*x > *y) return +1;
    return 0;
}

void sort_cpu(float in[], float out[], int n) {
    for (int i = 0; i < n; i += 1)
        out[i] = in[i];
    qsort(out, n, sizeof(float), comp);
}

bool compare_arrays(float a[], float b[], int n) {
    const float EPS = 1E-6;
    for (int i = 0; i < n; i += 1)
        if (abs(a[i] - b[i]) > EPS)
            return true;
    return false;
}

void read_array(float x[], int n) {
    for (int i = 0; i < n; i += 1)
        scanf("%f", &x[i]);
}

void write_array(float x[], int n) {
    for (int i = 0; i < n; i += 1)
        printf("%g ", x[i]);
    printf("\n");
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline __host__ __device__ void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) 
        printf("ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
}

__device__ int bin_search(float *x, int n, float val, bool min) {
    int le = 0, ri = n, mid;
    while (ri > le) {
        mid = (le + ri) / 2;
        if ((x[mid] < val) || (x[mid] == val && min == false)) {
            le = mid + 1; /* go to right */
        } else {
            ri = mid; /* go to left */
        }
    }
    return ri;
}

/* `le`/`ri` is the size of the left/right part  */
__global__ void merge(float *in, float *out, int le, int ri) {
    int idx1, idx2;
    idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx1 >= le + ri) return; /* out of bounds */
    float val = in[idx1];
    if (idx1 < le) {
        idx2 = bin_search(in + le, ri, val, true);
    } else {
        idx1 -= le;
        idx2 = bin_search(in, le, val, false);
    }
    out[idx1 + idx2] = val;
}

__global__ void merge_sort_dyn(float *in, float *out, float *tmp, int n) {
    if (n == 1) { out[0] = in[0]; return; }

    cudaStream_t s[2];
    cudaStreamCreateWithFlags(&s[0], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s[1], cudaStreamNonBlocking);

    merge_sort_dyn<<<1,1,0,s[0]>>>(in, tmp, out, n/2);
    merge_sort_dyn<<<1,1,0,s[1]>>>(in + n/2, tmp + n/2, out + n/2, n - n/2);

    cudaStreamDestroy(s[0]);
    cudaStreamDestroy(s[1]);

    gpuErrchk(cudaDeviceSynchronize());

    if (n <= B) {
        merge<<<1,n>>>(tmp, out, n/2, n - n/2);
    } else {
        merge<<<n/B+(n%B ? 1 : 0),B>>>(tmp, out, n/2, n - n/2);
    }
}

/* when `IO == 1` program will receive input from stdin and write result to stdout */
#define IO 0

int main() {
    int n;

    if (IO) {
        scanf("%d", &n);
    } else {
        n = 32 * 1024;
    }
    
    int num_bytes = n * sizeof(float);

    float *in = (float *) malloc(num_bytes);
    float *out = (float *) malloc(num_bytes);
    float *gold = (float *) malloc(num_bytes);

    if (IO) {
        read_array(in, n);
    } else {
        fill_array(in, n, false);   
    }
     
    sort_cpu(in, gold, n);

    float *d_in, *d_out, *d_tmp;
    cudaMalloc(&d_in, num_bytes);
    cudaMalloc(&d_out, num_bytes);
    cudaMalloc(&d_tmp, num_bytes);

    /* set `cudaLimitDevRuntimeSyncDepth` to max */
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 24);

    GpuTimer timer;

    cudaMemcpy(d_in, in, num_bytes, cudaMemcpyHostToDevice);
    timer.Start();
    merge_sort_dyn<<<1,1>>>(d_in, d_out, d_tmp, n);
    timer.Stop();
    cudaMemcpy(out, d_out, num_bytes, cudaMemcpyDeviceToHost);

    fprintf(stderr, "[merge_sort_dyn] Time: %g ms\n", timer.Elapsed());

    bool ok = !compare_arrays(out, gold, n);
    if (ok) fprintf(stderr, "Outputs match. Success!\n");
    else fprintf(stderr, "Outputs don't match. Failed!\n");

    
        write_array(out, n);
    

    free(in);
    free(out);
    free(gold);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}