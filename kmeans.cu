/* Parallel K-Means Clustering Algorithm implemented in CUDA */
/* compile with `nvcc -arch=compute_35 -rdc=true -lcurand main.cu` */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <curand.h>
#include "gputimer.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline __host__ __device__ void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) 
        printf("ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
}

#define EPS 1E-2

__device__ inline float add(float a, float b) { return a + b; }
__device__ inline float sub(float a, float b) { return a - b; }

/* out <- op(lhs, rhs) */
__global__ void op2(float *lhs, float *rhs, int n, float (*op)(float, float), float *out) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) out[idx] = op(lhs[idx], rhs[idx]);
}

/* out <- op(in) */
__global__ void op1(float *in, int n, float (*op)(float), float *out) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) out[idx] = op(in[idx]);
}

/* each block should calculate reduce[`op`] on last dimension of `in` and write results to `out` */
__global__ void reduce(float *in, float *out, float (*op)(float, float)) {
    extern __shared__ float s[];

    int idx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + idx;

    s[idx] = in[i];
    __syncthreads();

    for (int n = blockDim.x; n > 1; n = ((n + 1) >> 1)) {
        int w = ((n + 1) >> 1);
        if (idx < (n >> 1))
            s[idx] = op(s[idx], s[idx + w]);
        __syncthreads();
    }

    if (idx == 0)
        out[blockIdx.x] = s[0];
}

/* distributes data points into bins based on their cluster index (m x d -> k x d x m) */
__global__ void bin(float *in, float *out, float *out_mask, int *idx) {
	int m = gridDim.x;
	int d = blockDim.x;
	int i = blockIdx.x;
	int c = idx[i];

	int out_idx = c * d * m + threadIdx.x * m + i;
	int in_idx = i * d + threadIdx.x;

	out[out_idx] = in[in_idx];

	if (threadIdx.x == 0)
		out_mask[c * m + i] = 1.0;
}

/* each block divides last dimension of `lhs` by the value in `rhs` */
__global__ void div(float *lhs, float *rhs, float *out) {
	float q = rhs[blockIdx.x];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (q != 0) out[idx] = lhs[idx] / q;
}

/**
 * A device function that updates cluster centroids to equal the mean of each cluster.
 * @param x   [input]  The training set (m x d)
 * @param m   [input]  The number of training examples
 * @param d   [input]  The number of features
 * @param k   [input]  The number of clusters
 * @param mu  [output] The cluster centroids (k x d)
 * @param c   [input] The cluster assigned to each training example
 * @param ti  Memory for storing intermediate results (m x k x d)
 * @return    1 if update was smaller than `EPS` in all dimensions, 0 otherwise
 */
__device__ int update_centroids(float *x, int m, int d, int k, float *mu, int *c,
    float *t1, float *t2, float *t3, float *t4) {
    cudaStream_t s[3];
    cudaStreamCreateWithFlags(&s[0], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s[1], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s[2], cudaStreamNonBlocking);

    cudaMemsetAsync(t1, 0, m * k * d * sizeof(float), s[0]);
    cudaMemsetAsync(t2, 0, m * k * sizeof(float), s[1]);
    cudaMemcpyAsync(t3, mu, k * d * sizeof(float), cudaMemcpyDeviceToDevice, s[2]);
    gpuErrchk(cudaDeviceSynchronize());

    bin<<<m,d>>>(x, t1, t2, c);
    gpuErrchk(cudaDeviceSynchronize());

	reduce<<<k*d,m,m*sizeof(float),s[0]>>>(t1, mu, add);
    reduce<<<k,m,m*sizeof(float),s[1]>>>(t2, t4, add);
    gpuErrchk(cudaDeviceSynchronize());

    cudaStreamDestroy(s[0]);
    cudaStreamDestroy(s[1]);
    cudaStreamDestroy(s[2]);

    div<<<k,d>>>(mu, t4, mu);

    /* calculate centroid displacements */
	op2<<<k,d>>>(t3, mu, k*d, sub, t3);
	op1<<<k,d>>>(t3, k*d, abs, t3);
	reduce<<<k,d,d*sizeof(float)>>>(t3, t2, max);
	reduce<<<1,k,k*sizeof(float)>>>(t2, t1, max);
    gpuErrchk(cudaDeviceSynchronize());

	return (t1[0] < EPS ? 1 : 0);
}

/* t[i, j, k] = (x[i, k] - mu[j, k]) ^ 2 */
__global__ void sqr_dist(float *x, float *mu, float *t) {
    float a = x[blockIdx.x * blockDim.x + threadIdx.x];
    float b = mu[blockIdx.y * blockDim.x + threadIdx.x];
    t[blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x] = (a - b) * (a - b);
}

/* each block searches for its value (from `v`) in last dimension of `x` */
__global__ void parallel_search(float *x, float *v, int *out) {
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int v_idx = blockIdx.x;

    if (x[x_idx] == v[v_idx])
        out[v_idx] = threadIdx.x;
}

/**
 * A device function that assigns each training example to nearest cluster centroid.
 * @param x   [input]  The training set (m x d)
 * @param m   [input]  The number of training examples
 * @param d   [input]  The number of features
 * @param k   [input]  The number of clusters
 * @param mu  [input]  The cluster centroids (k x d)
 * @param c   [output] The cluster assigned to each training example
 * @param ti  Memory for storing intermediate results (m x k x d)
 */
__device__ void cluster(float *x, int m, int d, int k, float *mu, int *c,
    float *t1, float *t2, float *t3, float *t4) {
    dim3 blocks(m, k);
    sqr_dist<<<blocks,d>>>(x, mu, t3);
    reduce<<<m*k,d,d*sizeof(float)>>>(t3, t2, add);
    reduce<<<m,k,k*sizeof(float)>>>(t2, t1, min);
    parallel_search<<<m,k>>>(t2, t1, c);
    // printf("Hello!");
}

/**
 * A GPU kernel for k-means clustering.
 * This kernel should be launched with only a single thread (<<<1,1>>>).
 * It will launch other child kernels on demand.
 * @param x   [input]  The training set (m x d)
 * @param m   [input]  The number of training examples
 * @param d   [input]  The number of features
 * @param k   [input]  The number of clusters
 * @param mu  [output] The cluster centroids (k x d)
 * @param c   [output] The cluster assigned to each training example
 */
__global__ void k_means(float *x, int m, int d, int k, float *mu, int *c) {
    /* allocate memory for storing intermediate results */
    float *t1 = (float*) malloc(m * k * d * sizeof(float));
    float *t2 = (float*) malloc(m * k * d * sizeof(float));
    float *t3 = (float*) malloc(m * k * d * sizeof(float));
    float *t4 = (float*) malloc(m * k * d * sizeof(float));

    /* repeat until convergence */
    int done = 0;
    while (!done) {
        /* optimize c */
        cluster(x, m, d, k, mu, c,
            t1, t2, t3, t4);
        /* optimize mu */
        done = update_centroids(x, m, d, k, mu, c,
            t1, t2, t3, t4);
    }

    /* free allocated memory */
    free(t1);
    free(t2);
    free(t3);
    free(t4);
}

int main() {
    /* read parameters */
    int m, d, k;
    scanf("%d %d %d", &m, &d, &k);

    assert(k <= m);

    /* allocate host memory */    
    int *c = (int*) malloc(m * sizeof(int));
    float *x = (float*) malloc(m * d * sizeof(float));
    float *mu = (float*) malloc(k * d * sizeof(float));

    /* read training examples */
    for (int i = 0; i < m; i += 1)
        for (int j = 0; j < d; j += 1)
            scanf("%f", &x[i * d + j]);

    /* allocate device memory */
    int *d_c;
    cudaMalloc(&d_c, m * sizeof(int));

    float *d_x, *d_mu;
    cudaMalloc(&d_x, m * d * sizeof(float));
    cudaMalloc(&d_mu, k * d * sizeof(float));

    /* set `cudaLimitDevRuntimeSyncDepth` to max */
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 24);

    /* copy dataset to device memory */
    cudaMemcpy(d_x, x, m * d * sizeof(float), cudaMemcpyHostToDevice);

    /* create and initialize CUDA PRNG */
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(0));

    /* initialize cluster centroids randomly */
    curandGenerateUniform(gen, d_mu, k * d);

    /* launch and time kernel */
    GpuTimer timer;
    timer.Start();
    k_means<<<1,1>>>(d_x, m, d, k, d_mu, d_c);
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    timer.Stop();

    /* copy results to host memory */
    cudaMemcpy(mu, d_mu, k * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, d_c, m * sizeof(int), cudaMemcpyDeviceToHost);

    fprintf(stderr, "[k_means] Time: %g ms\n", timer.Elapsed());

    /* print assigned clusters */
    for (int i = 0; i < m; i += 1)
        printf("%d\n", c[i]);

    printf("\n");

    /* print cluster centroids */
    for (int i = 0; i < k; i += 1) {
        for (int j = 0; j < d; j += 1)
            printf("%9g ", mu[i * d + j]);
        printf("\n");
    }

    /* destroy PRNG */
    curandDestroyGenerator(gen);

    /* free allocated host memory */
    free(c);
    free(x);
    free(mu);

    /* free allocated device memory */
    cudaFree(d_c);
    cudaFree(d_x);
    cudaFree(d_mu);

    return 0;
}