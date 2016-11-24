#include "Layer_GPU.cuh"
#include "Utilities.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <random>
#include <iostream>
#include <limits>

#define TILE_DIM 32
#define BLOCK_SIZE 32

Layer_GPU::Layer_GPU(const unsigned nrInputs, const unsigned nrNeurons) : nrInputs(nrInputs), nrNeurons(nrNeurons), sizeW((nrInputs) * (nrNeurons)), trainRate(0.1), momentum(0.1), bias(1.0)
{
	utils::CheckError(cudaMalloc((void**)&d_weights, sizeW * sizeof(double)), __LINE__);
	utils::CheckError(cudaMalloc((void**)&d_deltaWeights, sizeW * sizeof(double)), __LINE__);
	utils::CheckError(cudaMalloc((void**)&d_activationResult, nrNeurons * sizeof(double)), __LINE__);
	utils::CheckError(cudaMalloc((void**)&d_gradients, nrNeurons * sizeof(double)), __LINE__);

	initDeltaWeights();
}

Layer_GPU::~Layer_GPU()
{
	cudaFree(d_weights);
	cudaFree(d_deltaWeights);
	cudaFree(d_activationResult);
	cudaFree(d_gradients);
}

void Layer_GPU::SetBias(const double bias)
{
	this->bias = bias;
}

void Layer_GPU::InitWeights()
{
	auto weights = new double[sizeW];
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.1, 1.0);
	for (auto i = 0; i < static_cast<decltype(i)>(sizeW); ++i)
	{
		weights[i] = dis(gen);
	}
	utils::CheckError(cudaMemcpy(d_weights, weights, sizeW * sizeof(double), cudaMemcpyHostToDevice), __LINE__);
	delete[] weights;
}

void Layer_GPU::initDeltaWeights()
{
	auto deltaWeights = new double[sizeW];
	for (auto i = 0; i < static_cast<decltype(i)>(sizeW); ++i)
	{
		deltaWeights[i] = 0;
	}
	utils::CheckError(cudaMemcpy(d_deltaWeights, deltaWeights, sizeW * sizeof(double), cudaMemcpyHostToDevice), __LINE__);
	delete[] deltaWeights;
}



void Layer_GPU::SetTrainRate(const double trainRate)
{
	this->trainRate = trainRate;
}

void Layer_GPU::SetMomentum(const double momentum)
{
	this->momentum = momentum;
}

const double* Layer_GPU::Output_GPU() const
{
	return d_activationResult;
}

unsigned Layer_GPU::OutputSize() const
{
	return nrNeurons;
}

//------------------------------------------------------------------------
// cuda kernels
//------------------------------------------------------------------------
__device__ double cuda_activationFunc(double value)
{
	auto res = (exp(value) - exp(-value)) / (exp(value) + exp(-value));
	return res;
}

__device__ double cuda_activeationFuncD(double value)
{
	auto s = cuda_activationFunc(value);
	return (1.0 - s * s);
}

__global__ void cuda_matmul(const double* A, const double* B, double* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)
{
	double CValue = 0;

	int Row = blockIdx.y * TILE_DIM + threadIdx.y;
	int Col = blockIdx.x * TILE_DIM + threadIdx.x;

	__shared__ float sd_A[TILE_DIM][TILE_DIM];
	__shared__ float sd_B[TILE_DIM][TILE_DIM];

	for (int k = 0; k < (TILE_DIM + ACols - 1) / TILE_DIM; k++) {

		if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
		{
			sd_A[threadIdx.y][threadIdx.x] = A[Row * ACols + k*TILE_DIM + threadIdx.x];
		}
		else {
			sd_A[threadIdx.y][threadIdx.x] = 0.0;
		}

		if (k * TILE_DIM + threadIdx.y < BRows && Col < BCols) {
			sd_B[threadIdx.y][threadIdx.x] = B[(k * TILE_DIM + threadIdx.y) * BCols + Col];
		}
		else {
			sd_B[threadIdx.y][threadIdx.x] = 0.0;
		}

		__syncthreads();

		for (int n = 0; n < TILE_DIM; ++n) {
			CValue += sd_A[threadIdx.y][n] * sd_B[n][threadIdx.x];
		}

		__syncthreads();
	}

	if (Row < CRows && Col < CCols)
	{
		C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) + (blockIdx.x*blockDim.x) + threadIdx.x] = cuda_activationFunc(CValue);
	}
}

__global__  void cuda_total(double * d_input, double * d_output, int len)
{
	// Load a segment of the input vector into shared memory
	__shared__ float partialSum[2 * BLOCK_SIZE];
	int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x*blockDim.x;

	if ((start + t) < len)
	{
		partialSum[t] = d_input[start + t];
	}
	else
	{
		partialSum[t] = 0.0;
	}
	if ((start + blockDim.x + t) < len)
	{
		partialSum[blockDim.x + t] = d_input[start + blockDim.x + t];
	}
	else
	{
		partialSum[blockDim.x + t] = 0.0;
	}

	// Traverse reduction tree
	for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
	{
		__syncthreads();
		if (t < stride)
			partialSum[t] += partialSum[t + stride];
	}
	__syncthreads();

	// Write the computed sum of the block to the output vector at correct index
	if (t == 0 && (globalThreadId * 2) < len)
	{
		d_output[blockIdx.x] = partialSum[t];
	}
}

__global__ void cuda_sumdw(double *d_weights, double *d_gradients, double *d_output, unsigned sizeW, unsigned sizeGrad, unsigned sizeOutput)
{
	int threadId = blockIdx.x*blockDim.x + threadIdx.x;
	if (threadId < sizeOutput)
	{
		for (auto i = 0; i < static_cast<decltype(i)>(sizeGrad); ++i)
		{
			d_output[threadId] += d_weights[i * sizeOutput + threadId] * d_gradients[i];
		}
	}
}

__global__ void cuda_gradientsLastLayer(const double *d_targetVals, const double *d_activationResults, double *d_gradients, unsigned nrNeurons)
{
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nrNeurons) {
		auto delta = d_targetVals[i] - d_activationResults[i];
		d_gradients[i] = delta * cuda_activeationFuncD(d_activationResults[i]);
	}
}

__global__ void cuda_gradients(const double *d_deltas, const double *d_activationResults, double *d_gradients, unsigned nrNeurons)
{
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nrNeurons) {
		d_gradients[i] = d_deltas[i] * cuda_activeationFuncD(d_activationResults[i]);
	}
}

__global__ void cuda_updateWeights(double * d_weights, double * d_deltaWeights, double * d_activationResults, double * d_gradients,
	unsigned sizeW, unsigned nrNeurons, double trainRate, double momentum)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < sizeW)
	{
		auto oldDeltaWeight = d_deltaWeights[i];
		auto idx = i % (nrNeurons);
		auto newDeltaWeight = trainRate * d_activationResults[idx] * d_gradients[idx] + momentum * oldDeltaWeight;
		d_deltaWeights[i] = newDeltaWeight;
		d_weights[i] += d_deltaWeights[i];
	}
}

__global__  void cuda_min_max(double * d_input, double * d_outputMin, double *d_outputMax, int len, double lowest, double highest)
{
	// Load a segment of the input vector into shared memory
	__shared__ double partialMin[2 * BLOCK_SIZE];
	__shared__ double partialMax[2 * BLOCK_SIZE];
	int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x*blockDim.x;

	if ((start + t) < len)
	{
		partialMin[t] = d_input[start + t];
		partialMax[t] = d_input[start + t];
	}
	else
	{
		partialMin[t] = highest;
		partialMax[t] = lowest;
	}
	if ((start + blockDim.x + t) < len)
	{
		partialMin[blockDim.x + t] = d_input[start + blockDim.x + t];
		partialMax[blockDim.x + t] = d_input[start + blockDim.x + t];
	}
	else
	{
		partialMin[blockDim.x + t] = highest;
		partialMax[blockDim.x + t] = lowest;
	}

	// Traverse reduction tree
	for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
	{
		__syncthreads();
		if (t < stride)
		{
			if (partialMin[t] > partialMin[t + stride])
			{
				partialMin[t] = partialMin[t + stride];
			}
			if (partialMax[t] < partialMax[t + stride])
			{
				partialMax[t] = partialMax[t + stride];
			}
		}
	}
	__syncthreads();

	// Write the computed sum of the block to the output vector at correct index
	if (t == 0 && (globalThreadId * 2) < len)
	{
		d_outputMin[blockIdx.x] = partialMin[t];
		d_outputMax[blockIdx.x] = partialMax[t];
	}
}

__global__ void cuda_normalizeWeights(double * d_weights, unsigned sizeW, double min, double max)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < sizeW)
	{
		d_weights[i] = (d_weights[i] - min) / (max - min);
	}
}

//------------------------------------------------------------------------
// end of cuda kernels
//------------------------------------------------------------------------

double *Layer_GPU::SumDW_GPU() const
{
	double *d_sumdw = nullptr;
	utils::CheckError(cudaMalloc((void**)&d_sumdw, nrInputs * sizeof(double)), __LINE__);
	dim3 DimGrid(32, 1, 1);
	dim3 DimBlock(BLOCK_SIZE, 1, 1);

	cuda_sumdw << <DimGrid, DimBlock >> > (d_weights, d_gradients, d_sumdw, sizeW, nrNeurons, nrInputs);

	return d_sumdw;
}

void Layer_GPU::FeedForward_GPU(const double* d_inputs)
{
	dim3 grids(32, 32, 1);
	dim3 blocks(32, 32, 1);

	cuda_matmul << < grids, blocks >> > (d_inputs, d_weights, d_activationResult, 1, nrInputs, nrInputs, nrNeurons, 1, nrNeurons);
	utils::CheckError(cudaGetLastError(), __LINE__);
}

void Layer_GPU::BackPropagation_GPU(const double *inputs)
{
	calcGradients_GPU(inputs);
}

void Layer_GPU::BackPropagation_GPU(const std::shared_ptr<Layer_GPU> &prevLayer)
{
	calcGradients_GPU(prevLayer);
}


//calcualte the gradients for the output layer
void Layer_GPU::calcGradients_GPU(const double* targetVals)
{
	dim3 grids(32, 32, 1);
	dim3 blocks(32, 32, 1);
	double *d_targetVals = nullptr;
	utils::CheckError(cudaMalloc((void**)&d_targetVals, nrInputs * sizeof(double)), __LINE__);
	utils::CheckError(cudaMemcpy(d_targetVals, targetVals, sizeW * sizeof(double), cudaMemcpyHostToDevice), __LINE__);
	cuda_gradientsLastLayer << < grids, blocks >> > (d_targetVals, d_activationResult, d_gradients, nrNeurons);
	utils::CheckError(cudaGetLastError(), __LINE__);
}

//calculate the gradients for the hidden layer
void Layer_GPU::calcGradients_GPU(const std::shared_ptr<Layer_GPU> &prevLayer)
{
	dim3 grids(32, 32, 1);
	dim3 blocks(32, 32, 1);
	cuda_gradients << < grids, blocks >> > (prevLayer->SumDW_GPU(), d_activationResult, d_gradients, nrNeurons);
	utils::CheckError(cudaGetLastError(), __LINE__);
}

void Layer_GPU::UpdateWeights_GPU()
{
	dim3 grids(std::ceil(sizeW / 256.0), 1, 1);
	dim3 blocks(512, 1, 1);
	cuda_updateWeights << <grids, blocks >> > (d_weights, d_deltaWeights, d_activationResult, d_gradients, sizeW, nrNeurons, trainRate, momentum);
	utils::CheckError(cudaGetLastError(), __LINE__);

	if (sizeW > 1) {
		unsigned numOutputElements = sizeW / (BLOCK_SIZE << 1);
		if (sizeW % (BLOCK_SIZE << 1))
		{
			numOutputElements++;
		}
		double *d_outputMin = nullptr;
		double *d_outputMax = nullptr;
		utils::CheckError(cudaMalloc((void**)&d_outputMin, numOutputElements * sizeof(double)), __LINE__);
		utils::CheckError(cudaMalloc((void**)&d_outputMax, numOutputElements * sizeof(double)), __LINE__);

		dim3 DimGrid(numOutputElements, 1, 1);
		dim3 DimBlock(BLOCK_SIZE, 1, 1);

		cuda_min_max << <DimGrid, DimBlock >> > (d_weights, d_outputMin, d_outputMax, sizeW, std::numeric_limits<double>::min(), std::numeric_limits<double>::max());

		double * outputMin = new double[numOutputElements];
		double * outputMax = new double[numOutputElements];
		utils::CheckError(cudaMemcpy(outputMin, d_outputMin, numOutputElements * sizeof(double), cudaMemcpyDeviceToHost), __LINE__);
		utils::CheckError(cudaMemcpy(outputMax, d_outputMax, numOutputElements * sizeof(double), cudaMemcpyDeviceToHost), __LINE__);

		double min = outputMin[0];
		double max = outputMax[0];
		for (unsigned i = 1; i < numOutputElements; i++)
		{
			if (min > outputMin[i])
			{
				min = outputMin[i];
			}
			if (max < outputMax[i])
			{
				max = outputMax[i];
			}
		}

		cuda_normalizeWeights << <grids, blocks >> > (d_weights, sizeW, min, max);
		utils::CheckError(cudaGetLastError(), __LINE__);

		cudaFree(d_outputMin);
		cudaFree(d_outputMax);
		delete[] outputMin;
		delete[] outputMax;
	}
}




