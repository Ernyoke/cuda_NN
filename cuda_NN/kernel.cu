
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

//create nets for CPU testing
Net TestNet1_CPU()
{
	Net net;
	std::shared_ptr<Layer> layer1(new Layer(784, 128));
	layer1->InitWeights();
	std::shared_ptr<Layer> layer2(new Layer(128, 128));
	layer2->InitWeights();
	std::shared_ptr<Layer> outputlayer(new Layer(128, 10));
	outputlayer->InitWeights();

	net.AddLayer(layer1);
	net.AddLayer(layer2);
	net.AddLayer(outputlayer);
	return net;
}

Net TestNet2_CPU()
{
	Net net;
	std::shared_ptr<Layer> layer1(new Layer(784, 128));
	layer1->InitWeights();
	std::shared_ptr<Layer> layer2(new Layer(128, 128));
	layer2->InitWeights();
	std::shared_ptr<Layer> layer3(new Layer(128, 128));
	layer3->InitWeights();
	std::shared_ptr<Layer> outputlayer(new Layer(128, 10));
	outputlayer->InitWeights();

	net.AddLayer(layer1);
	net.AddLayer(layer2);
	net.AddLayer(layer3);
	net.AddLayer(outputlayer);
	return net;
}

Net TestNet3_CPU()
{
	Net net;
	std::shared_ptr<Layer> layer1(new Layer(784, 1024));
	layer1->InitWeights();
	std::shared_ptr<Layer> layer2(new Layer(1024, 1024));
	layer2->InitWeights();
	std::shared_ptr<Layer> outputlayer(new Layer(1024, 10));
	outputlayer->InitWeights();

	net.AddLayer(layer1);
	net.AddLayer(layer2);
	net.AddLayer(outputlayer);
	return net;
}

Net TestNet4_CPU()
{
	Net net;
	std::shared_ptr<Layer> layer1(new Layer(784, 1024));
	layer1->InitWeights();
	std::shared_ptr<Layer> layer2(new Layer(1024, 1024));
	layer2->InitWeights();
	std::shared_ptr<Layer> layer3(new Layer(1024, 1024));
	layer3->InitWeights();
	std::shared_ptr<Layer> outputlayer(new Layer(1024, 10));
	outputlayer->InitWeights();

	net.AddLayer(layer1);
	net.AddLayer(layer2);
	net.AddLayer(layer3);
	net.AddLayer(outputlayer);
	return net;
}

//create nets for GPU testing
Net_GPU TestNet1_GPU()
{
	Net_GPU net;
	std::shared_ptr<Layer_GPU> layer1(new Layer_GPU(784, 128));
	layer1->InitWeights();
	std::shared_ptr<Layer_GPU> layer2(new Layer_GPU(128, 128));
	layer2->InitWeights();
	std::shared_ptr<Layer_GPU> outputlayer(new Layer_GPU(128, 10));
	outputlayer->InitWeights();

	net.AddLayer(layer1);
	net.AddLayer(layer2);
	net.AddLayer(outputlayer);
	return net;
}

Net_GPU TestNet2_GPU()
{
	Net_GPU net;
	std::shared_ptr<Layer_GPU> layer1(new Layer_GPU(784, 128));
	layer1->InitWeights();
	std::shared_ptr<Layer_GPU> layer2(new Layer_GPU(128, 128));
	layer2->InitWeights();
	std::shared_ptr<Layer_GPU> layer3(new Layer_GPU(128, 128));
	layer3->InitWeights();
	std::shared_ptr<Layer_GPU> outputlayer(new Layer_GPU(128, 10));
	outputlayer->InitWeights();

	net.AddLayer(layer1);
	net.AddLayer(layer2);
	net.AddLayer(layer3);
	net.AddLayer(outputlayer);
	return net;
}

Net_GPU TestNet3_GPU()
{
	Net_GPU net;
	std::shared_ptr<Layer_GPU> layer1(new Layer_GPU(784, 1024));
	layer1->InitWeights();
	std::shared_ptr<Layer_GPU> layer2(new Layer_GPU(1024, 1024));
	layer2->InitWeights();
	std::shared_ptr<Layer_GPU> outputlayer(new Layer_GPU(1024, 10));
	outputlayer->InitWeights();

	net.AddLayer(layer1);
	net.AddLayer(layer2);
	net.AddLayer(outputlayer);
	return net;
}

Net_GPU TestNet4_GPU()
{
	Net_GPU net;
	std::shared_ptr<Layer_GPU> layer1(new Layer_GPU(784, 1024));
	layer1->InitWeights();
	std::shared_ptr<Layer_GPU> layer2(new Layer_GPU(1024, 1024));
	layer2->InitWeights();
	std::shared_ptr<Layer_GPU> layer3(new Layer_GPU(1024, 1024));
	layer3->InitWeights();
	std::shared_ptr<Layer_GPU> outputlayer(new Layer_GPU(1024, 10));
	outputlayer->InitWeights();

	net.AddLayer(layer1);
	net.AddLayer(layer2);
	net.AddLayer(layer3);
	net.AddLayer(outputlayer);
	return net;
}


void TrainCPU(Net && net, utils::mnist& mn)
{
	std::cout << "Starting training..." << std::endl;
	long long totalDuration = 0;
	for (auto x = 0; x < 100; ++x) {
		std::cout << "Epoch: " << x << std::endl;
		for (auto i = 0; i < 100; ++i)
		{
			auto globalError = 0.0;
			std::cout << "Sample " << i << std::endl;
			auto start = std::chrono::steady_clock::now();
			for (auto j = 0; j < 10; ++j)
			{
				net.Train(mn.images[i], mn.labels[i]);

				globalError = net.GetError();
			}
			auto duration = std::chrono::steady_clock::now() - start;
			totalDuration = totalDuration + duration.count();
		}
		std::cout << "Average iteration duration per epoch: " << totalDuration / 100 << std::endl;
	}
}

void TrainGPU(Net_GPU && net,  utils::mnist& mn)
{
	net.InitInputs(mn.images, mn.labels, mn.samples, mn.height * mn.width, mn.classes);

	std::cout << "Starting training..." << std::endl;
	long long totalDuration = 0;
	for (auto x = 0; x < 100; ++x) {
		std::cout << "Epoch: " << x << std::endl;
		for (auto i = 0; i < 100; ++i)
		{
			auto globalError = 0.0;
			std::cout << "Sample " << i << std::endl;
			auto start = std::chrono::steady_clock::now();
			for (auto j = 0; j < 10; ++j)
			{
				net.Train_GPU(i);

				globalError = net.GetError();
			}
			auto duration = std::chrono::steady_clock::now() - start;
			totalDuration = totalDuration + duration.count();
		}
		std::cout << "Average iteration duration per epoch: " << totalDuration / 100 << std::endl;
	}
}

int main()
{
	utils::mnist mn;
	std::cout << "Reading mnist data..." << std::endl;
	utils::ReadMnist(mn);

	std::cout << "Runnign test 1. for CPU" << std::endl;
	TrainCPU(TestNet1_CPU(), mn);
	std::cout << "Runnign test 1. for GPU" << std::endl;
	TrainGPU(TestNet1_GPU(), mn);
	std::cout << "Runnign test 2. for CPU" << std::endl;
	TrainCPU(TestNet2_CPU(), mn);
	std::cout << "Runnign test 2. for GPU" << std::endl;
	TrainGPU(TestNet2_GPU(), mn);
	std::cout << "Runnign test 3. for CPU" << std::endl;
	TrainCPU(TestNet3_CPU(), mn);
	std::cout << "Runnign test 3. for GPU" << std::endl;
	TrainGPU(TestNet3_GPU(), mn);
	std::cout << "Runnign test 4. for CPU" << std::endl;
	TrainCPU(TestNet4_CPU(), mn);
	std::cout << "Runnign test 4. for GPU" << std::endl;
	TrainGPU(TestNet4_GPU(), mn);

	return 0;
}
