#pragma once

#include <stdint.h>

#include "const.h"

uint32_t * gpuAlloc(void) ;
void gpuFree(void* gpu_mem);
int gpuBlit(void* src, void* dst);

void gpuRender(uint32_t* buf);
