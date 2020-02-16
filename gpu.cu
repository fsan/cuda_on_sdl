#include "gpu.h"

uint32_t * gpualloc(void) {
	uint32_t* gpu_mem;

	cudaError_t err = cudaMalloc(&gpu_mem, SCREEN_SIZE);
	if ( err != cudaSuccess ) return NULL;

	return gpu_mem;
};

void gpuFree(uint32_t* gpu_mem) {
	cudaFree(gpu_mem);
}

int gpuBlit(const uint32_t* src, const uint32_t* dst){
	cudaError_t err = cudaMemcpy(dst, src, SCREEN_SIZE, cudaMemcpyDeviceToHost);
	if ( err != cudaSuccess ) return 1;
	return 0;
}

// ----- 

__device__ uint8_t getPixColor(int x, int y) {
	return 0xff;
}

__global__ void my_kernel(uint32_t* buf) {
	const int xPix = blockDim.x * blockId.x + threadIdx.x;
	const int yPix = blockDim.y * blockId.y + threadIdx.y;
	buf[SCREEN_WIDTH*yPix + xPix] = getPixColor(xPix, yPix);
}

void gpuRender(const uint8_t* buf) {
	const dim3 blocksPerGrid(H_TILES, V_TILES);
	const dim3 threadsPerBlock(TILE_WIDTH, TILE_HEIGHT);
	my_kernel<<<blocksPerGrid, threadsPerBlock>>>(buf);
}
