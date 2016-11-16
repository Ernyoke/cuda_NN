#pragma once

#include <memory>

class Layer
{
public:
	Layer(const unsigned int nrInputs, const unsigned int nrNeurons);
	~Layer();

	void SetBias(const double bias);
	void SetWeights(const double *weights);
	void InitWeights();
	void SetTrainRate(const double trainRate);
	void SetMomentum(const double momentum);

	void FeedForward(const double * inputs);
	void BackPropagation(const double *inputs);
	void BackPropagation(const std::shared_ptr<Layer> &prevLayer);

	const double* Output() const;
	unsigned OutputSize() const;

	double SumDW(unsigned int layerIdx) const;

	void UpdateWeights();

	void FeedForward_GPU(const double * inputs);
	void BackPropagation_GPU(const double *inputs);
	void BackPropagation_GPU(const std::shared_ptr<Layer> &prevLayer);


private:
	double *weights;
	double *deltaWeights;
	unsigned nrInputs;
	unsigned nrNeurons;
	unsigned sizeW;

	double trainRate;
	double momentum;

	double *activationResult;
	double *gradients;

	double bias;

	double activationFunc(double value) const;
	double activeationFuncD(double value) const;

	void calcGradients(const double* targetVals);
	void calcGradients(const std::shared_ptr<Layer> &prevLayer);

	double *matmul(const double *inputs) const;
	void initDeltaWeights();

};
