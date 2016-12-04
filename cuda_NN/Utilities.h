#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <functional>

namespace utils
{
	void PrintMatrix(const double *mat, unsigned n, unsigned m);

	void CheckError(cudaError_t error, const char* file, unsigned line);
	void CheckError(cudaError_t error, unsigned line, std::function<void(void)> cleanup);

	struct mnist
	{
		double **images;
		double **labels;
		const unsigned samples = 60000;
		const unsigned width = 28;
		const unsigned height = 28;
		const unsigned classes = 10;

		~mnist()
		{
			for (auto i = 0; i < samples; ++i)
			{
				delete[] images[i];
				delete[] labels[i];
			}

			delete[] images;
			delete[] labels;
		}
	};

	void ReadMnist(struct mnist& mn);
}
