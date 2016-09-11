#ifndef __HEADER_H__
#define __HEADER_H__
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#endif
