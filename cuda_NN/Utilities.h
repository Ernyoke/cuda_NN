#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <functional>

namespace utils
{
	void PrintMatrix(const double *mat, unsigned n, unsigned m);

	void CheckError(cudaError_t error, unsigned line);
	void CheckError(cudaError_t error, unsigned line, std::function<void(void)> cleanup);
}
