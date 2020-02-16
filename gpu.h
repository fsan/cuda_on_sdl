
#include <stdint.h>

#include "const.h"

uint32_t * gpualloc(void) ;
void gpuFree(uint32_t* gpu_mem);
int gpuBlit(const uint32_t* src, const uint32_t* dst);

// ----- 

uint8_t getPixColor(int x, int y);
void my_kernel(uint32_t* buf);
void gpuRender(const uint8_t* buf);
