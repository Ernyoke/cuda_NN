
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <memory>
#include <iostream>
#include <chrono>

#include "Layer.cuh"
#include "Layer_GPU.cuh"
#include "Net.cuh"
#include "Net_GPU.cuh"
#include "Utilities.h"

int main()
{
	/*
	double a[30] = { 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000, 1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000, 2.7000, 2.8000, 2.9000, 3.0000 };
	double res[30] = { 0.0998, 0.1987, 0.2955, 0.3894, 0.4794, 0.5646, 0.6442, 0.7174, 0.7833, 0.8415, 0.8912, 0.9320, 0.9636, 0.9854, 0.9975, 0.9996, 0.9917, 0.9738, 0.9463, 0.9093, 0.8632, 0.8085, 0.7457, 0.6755, 0.5985, 0.5155, 0.4274, 0.3350, 0.2392, 0.1411 };
	*/

	utils::mnist mn;
	std::cout << "Reading mnist data..." << std::endl;
	utils::ReadMnist(mn);
	std::cout << "Creating net..." << std::endl;

	std::shared_ptr<Layer_GPU> layer1(new Layer_GPU(784, 128));
	layer1->InitWeights();
	std::shared_ptr<Layer_GPU> layer2(new Layer_GPU(128, 10));
	layer2->InitWeights();

	Net_GPU net;
	net.AddLayer(layer1);
	net.AddLayer(layer2);

	net.SetMomentum(0.5);
	net.SetTrainRate(0.001);

	net.InitInputs(mn.images, mn.labels, mn.samples, mn.height * mn.width, mn.classes);

	std::cout << "Starting training..." << std::endl;
	for (auto x = 0; x < 100; ++x) {
		std::cout << "Epoch: " << x << std::endl;
		for (auto i = 0; i < 10; ++i)
		{
			auto globalError = 0.0;
			std::cout << "Sample " << i << std::endl;
			for (auto j = 0; j < 10; ++j)
			{
				auto start = std::chrono::steady_clock::now();
				net.Train_GPU(i);
				auto duration = std::chrono::steady_clock::now() - start;
				//std::cout << "Clock: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << std::endl;

				globalError = net.GetError();
				std::cout << globalError << std::endl;
				//std::cout << error << std::endl;
			}
			//globalError /= 10;
			std::cout << globalError << std::endl;
		}
	}


	return 0;
}
