#include "gpu.h"

uint32_t * gpuAlloc(void) {
	uint32_t* gpu_mem;

	cudaError_t err = cudaMalloc(&gpu_mem, SCREEN_SIZE * 4);
	if ( err != cudaSuccess ) return NULL;

	return gpu_mem;
};

void gpuFree(void* gpu_mem) {
	cudaFree(gpu_mem);
}

int gpuBlit(void* src, void* dst){
	cudaError_t err = cudaMemcpy(dst, src, SCREEN_SIZE * 4, cudaMemcpyDeviceToHost);
	if ( err != cudaSuccess ) return 1;
	return 0;
}

// ----- 

__host__
__device__
uint32_t getPixColor(int x, int y) {
	return 0xFFFF0000;
}

__global__ void my_kernel(uint32_t* buf) {
	const int xPix = blockDim.x * blockIdx.x + threadIdx.x;
	const int yPix = blockDim.y * blockIdx.y + threadIdx.y;

	unsigned int pos = SCREEN_WIDTH * yPix + xPix;
	
	buf[pos] = getPixColor(xPix, yPix);
}

void gpuRender(uint32_t* buf) {
	const dim3 blocksPerGrid(H_TILES, V_TILES);
	const dim3 threadsPerBlock(TILE_WIDTH, TILE_HEIGHT);
	my_kernel<<<blocksPerGrid, threadsPerBlock>>>(buf);
}
