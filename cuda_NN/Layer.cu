#include "Layer.cuh"
#include "Utilities.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <random>
#include <iostream>

#define TILE_DIM 32

Layer::Layer(const unsigned nrInputs, const unsigned nrNeurons) : nrInputs(nrInputs), nrNeurons(nrNeurons), sizeW((nrInputs) * (nrNeurons)), trainRate(0.1), momentum(0.1), bias(1.0)
{
	weights = new double[sizeW];
	deltaWeights = new double[sizeW];
	activationResult = nullptr;
	gradients = nullptr;

	initDeltaWeights();

	utils::CheckError(cudaMalloc((void**)&d_a, nrInputs * sizeof(double)), __LINE__);
	utils::CheckError(cudaMalloc((void**)&d_b, sizeW * sizeof(double)), __LINE__);
	utils::CheckError(cudaMalloc((void**)&d_res, nrNeurons * sizeof(double)), __LINE__);
}

Layer::~Layer()
{
	delete[] weights;
	delete[] deltaWeights;
	delete[] activationResult;
	delete[] gradients;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_res);
}

void Layer::SetBias(const double bias)
{
	this->bias = bias;
}

void Layer::SetWeights(const double* weights)
{
	for (auto i = 0; i < static_cast<decltype(i)>(sizeW); ++i)
	{
		this->weights[i] = weights[i];
	}
}

void Layer::InitWeights()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.1, 1.0);
	for (auto i = 0; i < static_cast<decltype(i)>(sizeW); ++i)
	{
		this->weights[i] = dis(gen);
	}

	//utils::PrintMatrix(weights, nrInputs, nrNeurons + 1);
}

void Layer::initDeltaWeights()
{
	for (auto i = 0; i < static_cast<decltype(i)>(sizeW); ++i)
	{
		this->deltaWeights[i] = 0;
	}
}



void Layer::SetTrainRate(const double trainRate)
{
	this->trainRate = trainRate;
}

void Layer::SetMomentum(const double momentum)
{
	this->momentum = momentum;
}

void Layer::FeedForward(const double* inputs)
{
	//delete the old array with the activations
	delete[] activationResult;

	activationResult = matmul(inputs);

	for (auto i = 0; i < static_cast<decltype(i)>(nrNeurons); ++i)
	{
		activationResult[i] = activationFunc(activationResult[i]);
	}
}

const double* Layer::Output() const
{
	return activationResult;
}

unsigned Layer::OutputSize() const
{
	return nrNeurons;
}

//calculate the backpropagation if the layer is the output layer
void Layer::BackPropagation(const double* inputs)
{
	calcGradients(inputs);
	//std::cout << "Before:" << std::endl;
	//utils::PrintMatrix(weights, nrInputs, nrNeurons);
	//std::cout << "After:" << std::endl;
	//UpdateWeights();
	//utils::PrintMatrix(weights, nrInputs, nrNeurons);
}

//calculate the backpropagation if the layer is hidden/input layer
void Layer::BackPropagation(const std::shared_ptr<Layer> &prevLayer)
{
	calcGradients(prevLayer);
	//UpdateWeights();
}

/*
double Layer::activationFunc(double value) const
{
	auto res = 1.0 / (1.0 + std::exp(-value));
	return res;
}

double Layer::activeationFuncD(double value) const
{
	auto s = activationFunc(value);
	return s * (1.0 - s);
}*/

double Layer::activationFunc(double value) const
{
	auto res = (exp(value) - exp(-value)) / (exp(value) + exp(-value));
	if (res != res)
	{
		std::cout << "val: " << value << std::endl;
	}
	return res;
}

double Layer::activeationFuncD(double value) const
{
	auto s = activationFunc(value);
	return (1.0 - s * s);
}

double* Layer::matmul(const double* inputs) const
{
	auto *result = new double[nrNeurons];

	for (auto i = 0; i < static_cast<decltype(i)>(nrNeurons); ++i)
	{
		result[i] = 0;
		for (auto j = 0; j < static_cast<decltype(i)>(nrInputs); ++j)
		{
			result[i] += inputs[j] * weights[nrNeurons * j + i];
		}
	}

	return result;
}

//calcualte the gradients for the output layer
void Layer::calcGradients(const double* targetVals)
{
	delete[] gradients;
	gradients = new double[nrNeurons];
	for (auto i = 0; i < static_cast<decltype(i)>(nrNeurons); ++i)
	{
		auto delta = targetVals[i] - activationResult[i];
		gradients[i] = delta * activeationFuncD(activationResult[i]);
	}
}

//calculate the gradients for the hidden layer
void Layer::calcGradients(const std::shared_ptr<Layer> &prevLayer)
{
	delete[] gradients;
	gradients = new double[nrNeurons];
	for (auto i = 0; i < static_cast<decltype(i)>(nrNeurons); ++i)
	{
		gradients[i] = prevLayer->SumDW(i) * activeationFuncD(activationResult[i]);
	}
}

double Layer::SumDW(unsigned int layerIdx) const
{
	double sum = 0.0;
	//calculate the sum without including the bias neuron
	for (auto i = 0; i < static_cast<decltype(i)>(nrNeurons); ++i)
	{
		sum += weights[i * nrInputs + layerIdx] * gradients[i];
		if (sizeW < i * nrInputs + layerIdx)
		{
			std::cout << "Out of bound!" << std::endl;
		}
	}
	return sum;
}

void Layer::UpdateWeights()
{
	for (auto i = 0; i < static_cast<decltype(i)>(sizeW); ++i)
	{
		auto oldDeltaWeight = deltaWeights[i];
		auto idx = i % (nrNeurons);
		auto newDeltaWeight = trainRate * activationResult[idx] * gradients[idx] + momentum * oldDeltaWeight;
		deltaWeights[i] = newDeltaWeight;
		weights[i] += deltaWeights[i];
	}
	normalizeWeights();
}

std::tuple<double, double> Layer::min_max() const
{
	auto min = weights[0];
	auto max = weights[0];
	for (auto i = 1; i < static_cast<decltype(i)>(sizeW); ++i)
	{
		if (weights[i] > max)
		{
			max = weights[i];
		}

		if (weights[i] < min)
		{
			min = weights[i];
		}
	}
	return std::make_tuple(min, max);
}


void Layer::normalizeWeights()
{
	auto mm = min_max();
	auto min = std::get<0>(mm);
	auto max = std::get<1>(mm);
	if (max != min) {
		for (auto i = 0; i < static_cast<decltype(i)>(sizeW); ++i)
		{
			weights[i] = (weights[i] - min) / (max - min);
		}
	}
}


//-----------------------------------------------------------------------------------------------------------
//CUDA reatled methods
//-----------------------------------------------------------------------------------------------------------

__global__ void matmul_cu(const double* A, const double* B, double* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)
{
	double CValue = 0;

	int Row = blockIdx.y * TILE_DIM + threadIdx.y;
	int Col = blockIdx.x * TILE_DIM + threadIdx.x;

	__shared__ float As[TILE_DIM][TILE_DIM];
	__shared__ float Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + ACols - 1) / TILE_DIM; k++) {

		if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
		{
			As[threadIdx.y][threadIdx.x] = A[Row * ACols + k*TILE_DIM + threadIdx.x];
		}
		else {
			As[threadIdx.y][threadIdx.x] = 0.0;
		}

		if (k * TILE_DIM + threadIdx.y < BRows && Col < BCols) {
			Bs[threadIdx.y][threadIdx.x] = B[(k * TILE_DIM + threadIdx.y) * BCols + Col];
		}
		else {
			Bs[threadIdx.y][threadIdx.x] = 0.0;
		}

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) {
			CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
		}

		__syncthreads();
	}

	if (Row < CRows && Col < CCols)
	{
		C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue;
	}
}

void Layer::FeedForward_GPU(const double* inputs)
{

	//delete the old array with the activations
	delete[] activationResult;

	utils::CheckError(cudaMemcpy(d_a, inputs, nrInputs * sizeof(double), cudaMemcpyHostToDevice), __LINE__);
	utils::CheckError(cudaMemcpy(d_b, weights, sizeW * sizeof(double), cudaMemcpyHostToDevice), __LINE__);

	dim3 grids(32, 32, 1);
	dim3 blocks(32, 32, 1);

	matmul_cu << < grids, blocks >> > (d_a, d_b, d_res, 1, nrInputs, nrInputs, nrNeurons, 1, nrNeurons);
	utils::CheckError(cudaGetLastError(), __LINE__);

	activationResult = new double[nrNeurons];
	utils::CheckError(cudaMemcpy(activationResult, d_res, nrNeurons * sizeof(double), cudaMemcpyDeviceToHost), __LINE__);

	for (auto i = 0; i < static_cast<decltype(i)>(nrNeurons); ++i)
	{
		activationResult[i] = activationFunc(activationResult[i]);
	}
}

void Layer::BackPropagation_GPU(const double *inputs)
{
	BackPropagation(inputs);
}
void Layer::BackPropagation_GPU(const std::shared_ptr<Layer> &prevLayer)
{
	BackPropagation_GPU(prevLayer);
}



