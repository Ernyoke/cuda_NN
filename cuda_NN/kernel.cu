
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <memory>
#include <iostream>

#include "Layer.cuh"
#include "Net.h"

int main()
{

	Layer layer(3, 2);
	layer.InitWeights();
	double a[30] = { 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000, 1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.5000, 1.6000, 1.7000, 1.8000, 1.9000, 2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.5000, 2.6000, 2.7000, 2.8000, 2.9000, 3.0000 };
	double res[30] = { 0.0998, 0.1987, 0.2955, 0.3894, 0.4794, 0.5646, 0.6442, 0.7174, 0.7833, 0.8415, 0.8912, 0.9320, 0.9636, 0.9854, 0.9975, 0.9996, 0.9917, 0.9738, 0.9463, 0.9093, 0.8632, 0.8085, 0.7457, 0.6755, 0.5985, 0.5155, 0.4274, 0.3350, 0.2392, 0.1411 };

	std::shared_ptr<Layer> inputLayer(new Layer(1, 1));
	inputLayer->InitWeights();
	std::shared_ptr<Layer> hiddenLayer(new Layer(1, 3));
	hiddenLayer->InitWeights();
	std::shared_ptr<Layer> hiddenLayer2(new Layer(3, 3));
	hiddenLayer2->InitWeights();
	std::shared_ptr<Layer> outputLayer(new Layer(3, 1));
	outputLayer->InitWeights();

	Net net;
	net.AddLayer(inputLayer);
	net.AddLayer(hiddenLayer);
	net.AddLayer(hiddenLayer2);
	net.AddLayer(outputLayer);

	net.SetMomentum(0.2);
	net.SetTrainRate(0.01);

	double *inputs = new double[1];
	double *targets = new double[1];

	for (auto i = 0; i < 1000; ++i)
	{
		std::cout << std::endl << "Epoch " << i << ":" << std::endl;
		for (auto j = 0; j < 2; ++j)
		{
			inputs[0] = a[j];
			targets[0] = res[j];

			net.Train(inputs, targets);

			auto error = net.GetError();
			std::cout << error << std::endl;
		}
	}


    return 0;
}
