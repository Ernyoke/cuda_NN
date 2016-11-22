#pragma once

#include <vector>
#include <memory>

#include "Layer.cuh"

class Net
{
public:

	Net();
	~Net() {};

	void AddLayer(std::shared_ptr<Layer> layer);
	void Train(const double* inputs, const double *targets);
	const double* Activate(const double *inputs);

	void SetTrainRate(double trainRate);
	void SetMomentum(double momentum);

	double GetError() const;

	void Train_GPU(const double* inputs, const double *targets);

private:
	std::vector< std::shared_ptr<Layer> > layers;
	double trainRate;
	double momentum;

	double error;

	const double* feedForward(const double* inputs);
	void backPropagate(const double *targets);

	const double* feedForward_GPU(const double* inputs);
	void backPropagate_GPU(const double *targets);

	double errorFunc(const double *inputs, const double *targets, unsigned size) const;

};
