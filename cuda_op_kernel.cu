#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define THREADS_PER_BLOCK 512
#define BLOCK_SIZE 8

// SDC kernel for applying transformation
__global__ void SDCKernel(
    const float**** d_FS, 
    const float**** d_FT,
    float****** d_outD,
    int H,
    int W,
    int L,
    int C,
    int S,
    int R,
    int d) 
{ 
    // blocks are 8x8x8 threads, 
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= H) return;
    if (j >= W) return;
    if (k >= L) return;

    for (int l= -R; l <= R; l++) {
        for (int m= -R; m <= R; m++) {
            for (int n = -R; n <= R; n++) {
                int tot = 0;
                for (int c = 0; c < C; c++) {
                    if (i*S+l*d < 0 || i*S+l*d >= H) continue;
                    if (j*S+m*d < 0 || j*S+m*d >= W) continue;
                    if (k*S+n*d < 0 || k*S+n*d >= L) continue;
                    tot += d_FS[i*S][j*S][k*S][c] * d_FT[i*S+l*d][j*S+m*d][k*S+n*d][c];
                }
                d_outD[i][j][k][l][m][n] = tot;
            }
        }
    }
}

void SDCKernelLauncher(
    const float**** FS, 
    const float**** FT,
    float****** outD,
    int H,
    int W,
    int L,
    int C,
    int S,
    int R,
    int d)
{
    // cuda memory shit
    float**** d_FS;
    size_t size = H * W * L * C * sizeof(float);
    cudaMalloc(&d_FS, size);
    cudaMemcpy(d_FS,FS,size,cudaMemcpyHostToDevice);
    float**** d_FT;
    cudaMalloc(&d_FT, size);
    cudaMemcpy(d_FT,FT,size,cudaMemcpyHostToDevice);
    float****** d_outD;
    size_t sizeoutD = ((int)(H/_SS))*((int)(W/_SS))((int)(L/_SS))(2*_RR+1)(2*_RR+1)(2*_RR+1) * sizeof(float);
    cudaMalloc(&d_outD, sizeoutD);

    // set up kernel and dictate tasks
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(ceilf((float) W / (float)dimBlock.x), 
                ceilf((float) H / (float)dimBlock.y),
                ceilf((float) L / (float)dimBlock.z));
    SDCKernel<<<dimGrid, dimBlock>>>(d_FS, d_FT, d_outD, H, W, L, C, S, R, d);
    
    // copy result to host and free and report errors
    cudaMemcpy(outD, d_outD, sizeoutD,
        cudaMemcpyDeviceToHost);
    cudaFree(d_FS);
    cudaFree(d_FT);
    cudaFree(d_outD);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
    {
        printf("SDC kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }
}

// FS gradient
__global__ void FSGradKernel(
    const float**** d_FT,
    const float****** d_dLdO,
    float**** d_grad_FS,
    int H,
    int W,
    int L,
    int C,
    int S,
    int R,
    int d)
{
    // blocks are 8x8x8 threads, 
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    // check in bounds
    if (y >= H) return;
    if (x >= W) return;
    if (z >= L) return;
    // grad is 0 if x y or z are not multiples of s
    if (y % S != 0 || x % S != 0 || z % S != 0) {
        for (int c = 0; c < C; c++) {
            d_grad_FS[y][x][z][c] = 0;
        }
        return;
    }

    
    for (int c = 0; c < C; c++) {
        int tot = 0;
        for (int l= -R; l <= R; l++) {
            for (int m= -R; m <= R; m++) {
                for (int n = -R; n <= R; n++) {
                    if (y+l*d < 0 || y+l*d >= H) continue;
                    if (x+m*d < 0 || x+m*d >= W) continue;
                    if (z+n*d < 0 || z+n*d >= L) continue;
                    tot += d_dLdO[y/s][x/s][z/s][l][m][n] * d_FT[y+l*d][x+m*d][z+n*d][c];
                }
            }
        }
        d_grad_FS[y][x][z][c] = tot;
    }
    return;
}

void FSGradKernelLauncher(
    const float**** FT,
    const float****** dLdO,
    float**** grad_FS,
    int H,
    int W,
    int L,
    int C,
    int S,
    int R,
    int d)
{
    // cuda memory shit
    float**** d_FT;
    size_t size = H * W * L * C * sizeof(float);
    cudaMalloc(&d_FT, channelSize);
    cudaMemcpy(d_FT,FT,channelSize,cudaMemcpyHostToDevice);
    
    float****** d_dLdO;
    size_t sizedLdO = ((int)(H/_SS))*((int)(W/_SS))((int)(L/_SS))(2*_RR+1)(2*_RR+1)(2*_RR+1) * sizeof(float);
    cudaMalloc(&d_dLdO, sizedLdO);
    cudaMemcpy(d_dLdO,dLdO,sizedLdO,cudaMemcpyHostToDevice);

    float**** d_grad_FS;
    cudaMalloc(&d_grad_FS, channelSize);

    // set up kernel and dictate tasks
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(ceilf((float) W / (float)dimBlock.x), 
                ceilf((float) H / (float)dimBlock.y),
                ceilf((float) L / (float)dimBlock.z));

    FSGradKernel<<<dimGrid, dimBlock>>>(d_FT, d_dLdO, d_grad_FS, H, W, L, C, S, R, d);

    // copy result to host and free and report errors
    cudaMemcpy(grad_FS, d_grad_FS, channelSize,
        cudaMemcpyDeviceToHost);
    cudaFree(d_FT);
    cudaFree(d_dLdO);
    cudaFree(d_grad_FS);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
    {
        printf("FS Grad kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }
}


// FT gradient
__global__ void FTGradKernel(
    const float**** d_FS,
    const float****** d_dLdO,
    float**** d_grad_FT,
    int H,
    int W,
    int L,
    int C,
    int S,
    int R,
    int d)
{
    // blocks are 8x8x8 threads, 
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    // check in bounds
    if (y >= H) return;
    if (x >= W) return;
    if (z >= L) return;
    // grad is 0 if x y or z are not multiples of s
    if (y-l*d % S != 0 || x-m*d % S != 0 || z-n*d % S != 0) {
        for (int c = 0; c < C; c++) {
            d_grad_FT[y][x][z][c] = 0;
        }
        return;
    }

    
    for (int c = 0; c < C; c++) {
        int tot = 0;
        for (int l= -R; l <= R; l++) {
            for (int m= -R; m <= R; m++) {
                for (int n = -R; n <= R; n++) {
                    if (y+l*d < 0 || y+l*d >= H) continue;
                    if (x+m*d < 0 || x+m*d >= W) continue;
                    if (z+n*d < 0 || z+n*d >= L) continue;
                    tot += d_dLdO[(y-l*d)/s][(x-m*d)/s][(z-n*d)/s][l][m][n] * d_FS[y-l*d][x-m*d][z-n*d][c];
                }
            }
        }
        d_grad_FS[y][x][z][c] = tot;
    }
    return;
}


void FTGradKernelLauncher(
    const float**** FS,
    const float****** dLdO,
    float**** grad_FT,
    int H,
    int W,
    int L,
    int C,
    int S,
    int R,
    int d)
{
    // cuda memory shit
    float**** d_FS;
    size_t size = H * W * L * C * sizeof(float);
    cudaMalloc(&d_FS, channelSize);
    cudaMemcpy(d_FS,FS,channelSize,cudaMemcpyHostToDevice);
    
    float****** d_dLdO;
    size_t sizedLdO = ((int)(H/_SS))*((int)(W/_SS))((int)(L/_SS))(2*_RR+1)(2*_RR+1)(2*_RR+1) * sizeof(float);
    cudaMalloc(&d_dLdO, sizedLdO);
    cudaMemcpy(d_dLdO,dLdO,sizedLdO,cudaMemcpyHostToDevice);

    float**** d_grad_FT;
    cudaMalloc(&d_grad_FT, channelSize);

    // set up kernel and dictate tasks
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(ceilf((float) W / (float)dimBlock.x), 
                ceilf((float) H / (float)dimBlock.y),
                ceilf((float) L / (float)dimBlock.z));

    FSGradKernel<<<dimGrid, dimBlock>>>(d_FS, d_dLdO, d_grad_FT, H, W, L, C, S, R, d);

    // copy result to host and free and report errors
    cudaMemcpy(grad_FT, d_grad_FT, channelSize,
        cudaMemcpyDeviceToHost);
    cudaFree(d_FS);
    cudaFree(d_dLdO);
    cudaFree(d_grad_FT);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
    {
        printf("FT Grad kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }
}
