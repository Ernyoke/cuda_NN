#pragma once

#include <vector>
#include <memory>

#include "Layer_GPU.cuh"

class Net_GPU
{
public:

	Net_GPU();
	~Net_GPU();

	void AddLayer(std::shared_ptr<Layer_GPU> layer);
	void Train_GPU(unsigned inputPos);
	const double* Activate_GPU(const double *inputs);

	void SetTrainRate(double trainRate);
	void SetMomentum(double momentum);

	double GetError() const;

	void InitInputs(double **inputs, double **targets, unsigned n, unsigned inputSize, unsigned targetSize);

private:

	double **inputs;
	double **targets;
	unsigned nrInputs;
	unsigned inputSize;
	unsigned targetSize;

	std::vector< std::shared_ptr<Layer_GPU> > layers;
	double trainRate;
	double momentum;

	double error;

	const double* feedForward_GPU(const double* inputs);
	void backPropagate_GPU(const double *targets);

	double errorFunc(const double *inputs, const double *targets, unsigned size);

};