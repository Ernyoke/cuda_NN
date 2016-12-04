#include "Utilities.h"

#include <iostream>
#include <fstream>


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

void utils::CheckError(cudaError_t error, const char* file, unsigned line)
{
	if (error != cudaSuccess)
	{
		std::cout << "Error in " << file << " line " << line << ": " << cudaGetErrorString(error) << std::endl;
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

void utils::ReadMnist(struct mnist& mn)
{
	// Training image file name
	const std::string training_image_fn = "mnist/train-images.idx3-ubyte";

	// Training label file name
	const std:: string training_label_fn = "mnist/train-labels.idx1-ubyte";

	std::ifstream image;
	std::ifstream label;

	image.open(training_image_fn.c_str(), std::ios::in | std::ios::binary); // Binary image file
	label.open(training_label_fn.c_str(), std::ios::in | std::ios::binary); // Binary label file

	if(!image.is_open())
	{
		std::cerr << training_image_fn.c_str() << " could no be opened!" << std::endl;
		return;
	}

	if (!label.is_open())
	{
		std::cerr << training_label_fn.c_str() << " could no be opened!" << std::endl;
		return;
	}

	// Reading file headers
	char number;
	for (auto i = 1; i <= 16; ++i) {
		image.read(&number, sizeof(char));
	}
	for (auto i = 1; i <= 8; ++i) {
		label.read(&number, sizeof(char));
	}

	//allocating memory
	mn.images = new double*[mn.samples];
	mn.labels = new double*[mn.samples];
	for (auto i = 0; i < mn.samples; ++i)
	{
		mn.images[i] = new double[mn.width * mn.height];
		mn.labels[i] = new double[mn.classes];
	}

	//reading images and labels
	for (auto i = 0; i < mn.samples; ++i)
	{
		for (auto j = 0; j < mn.width * mn.height; ++j)
		{
			image.read(&number, sizeof(char));
			if (number == 0) {
				mn.images[i][j] = 0.0;
			}
			else {
				mn.images[i][j] = 1.0;
			}
		}

		label.read(&number, sizeof(char));
		for (int j = 0; j < mn.classes; ++j) {
			mn.labels[i][j] = 0.0;
		}
		mn.labels[i][number] = 1.0;
	}

	image.close();
	label.close();
}