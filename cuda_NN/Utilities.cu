#include "Utilities.h"

#include <iostream>


void utils::PrintMatrix(const double *mat, unsigned n, unsigned m)
{
	std::cout << "n = " << n << " m = " << m << std::endl;
	for (auto i = 0; i < static_cast<decltype(i)>(n); ++i)
	{
		for (auto j = 0; j < static_cast<decltype(j)>(m); ++j)
		{
			std::cout << mat[i * m + j] << " ";
		}
		std::cout << std::endl;
	}
}

void utils::CheckError(cudaError_t error, unsigned line)
{
	if (error != cudaSuccess)
	{
		std::cout << "Error in line " << line << ": " << cudaGetErrorString(error) << std::endl;
		exit(1);
	}
}

void utils::CheckError(cudaError_t error, unsigned line, std::function<void(void)> cleanup)
{
	if (error != cudaSuccess)
	{
		std::cout << "Error in line " << line << ": " << cudaGetErrorString(error) << std::endl;
		exit(1);
	}
}