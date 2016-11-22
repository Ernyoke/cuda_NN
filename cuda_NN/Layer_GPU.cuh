#pragma once

#include <memory>

class Layer_GPU
{
public:
	Layer_GPU(const unsigned int nrInputs, const unsigned int nrNeurons);
	~Layer_GPU();

	void SetBias(const double bias);
	void InitWeights();
	void SetTrainRate(const double trainRate);
	void SetMomentum(const double momentum);

	const double* Output_GPU() const;
	unsigned OutputSize() const;

	double *SumDW_GPU() const;
	void UpdateWeights_GPU();

	void FeedForward_GPU(const double * inputs);
	void BackPropagation_GPU(const double *inputs);
	void BackPropagation_GPU(const std::shared_ptr<Layer_GPU> &prevLayer);


private:
	double *d_weights;
	double *d_deltaWeights;
	unsigned nrInputs;
	unsigned nrNeurons;
	unsigned sizeW;

	double trainRate;
	double momentum;

	double *d_activationResult;
	double *d_gradients;

	double bias;

	void calcGradients_GPU(const double* targetVals);
	void calcGradients_GPU(const std::shared_ptr<Layer_GPU> &prevLayer);

	void normalizeWeights();
	void initDeltaWeights();

};
