#include "Net.h"
#include "Utilities.h"

#include <cmath>

Net::Net() : trainRate(0.1), momentum(0.1), error(0)
{
	
}

void Net::AddLayer(std::shared_ptr<Layer> layer)
{
	layers.push_back(layer);
}

void Net::SetTrainRate(double trainRate)
{
	this->trainRate = trainRate;
	for(auto& layer : layers)
	{
		layer->SetTrainRate(trainRate);
	}
}

void Net::SetMomentum(double momentum)
{
	this->momentum = momentum;
	for (auto& layer : layers)
	{
		layer->SetMomentum(momentum);
	}
}



const double* Net::feedForward(const double* inputs)
{
	//get the first layer from the list and feed the input into it
	auto i = 0;
	auto firstLayer = layers.at(i++);
	firstLayer->FeedForward(inputs);
	const double* output = firstLayer->Output();

	//let the data flow through the net
	for (; i < layers.size(); ++i)
	{
		auto& layer = layers.at(i);
		layer->FeedForward(output);
		output = layer->Output();
	}

	return output;
}

void Net::backPropagate(const double* targets)
{
	//get the last layer from the list
	int i = layers.size() - 1;
	auto lastLayer = layers.at(i--);
	lastLayer->BackPropagation(targets);

	//let the data flow through the net
	for (; i >= 0; --i)
	{
		auto& layer = layers.at(i);
		layer->BackPropagation(layers.at(i + 1));
	}
}

void Net::Train(const double* inputs, const double* targets)
{
	//let the input flow throught the net
	const double *output = feedForward(inputs);
	error = errorFunc(output, targets, layers.at(layers.size() - 1)->OutputSize());

	//backpropagate the calculated result
	backPropagate(targets);

	//update weigts
	for (auto& layer : layers)
	{
		layer->UpdateWeights();
	}
}

const double* Net::Activate(const double *inputs)
{
	return  feedForward(inputs);
}

double Net::GetError() const
{
	return error;
}

double Net::errorFunc(const double* outputs, const double* targets, unsigned size) const
{
	double delta = 0.0;
	for (auto i = 0; i < size; ++i)
	{
		delta += std::pow((targets[i] - outputs[i]), 2);
	}
	return std::sqrt((1.0 / static_cast<double>(size)) * delta);
}


