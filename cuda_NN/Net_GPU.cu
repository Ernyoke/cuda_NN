#include "Net_GPU.cuh"
#include "Utilities.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 512

//---------------------------------------------------------------------
// cuda kernels
//---------------------------------------------------------------------
__global__ void cuda_errorFunc(const double *d_outputs, const double *d_targets, double *d_res, unsigned len)
{
	__shared__ double sd_outputs[2 * BLOCK_SIZE];
	__shared__ double sd_targets[2 * BLOCK_SIZE];
	int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x*blockDim.x;

	if ((start + t) < len)
	{
		sd_outputs[t] = d_outputs[start + t];
		sd_targets[t] = d_targets[start + t];
	}
	else
	{
		sd_outputs[t] = 0.0;
		sd_targets[t] = 0.0;
	}
	if ((start + blockDim.x + t) < len)
	{
		sd_outputs[blockDim.x + t] = d_outputs[start + blockDim.x + t];
		sd_targets[blockDim.x + t] = d_targets[start + blockDim.x + t];
	}
	else
	{
		sd_outputs[blockDim.x + t] = 0.0;
		sd_targets[blockDim.x + t] = 0.0;
	}

	for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
	{
		__syncthreads();
		if (t < stride) {
			auto i = t + stride;
			sd_outputs[t] += (sd_targets[i] - sd_outputs[i]) * (sd_targets[i] - sd_outputs[i]);
		}
	}
	__syncthreads();

	if (t == 0 && (globalThreadId * 2) < len)
	{
		d_res[blockIdx.x] = sd_outputs[t];
	}
}
//---------------------------------------------------------------------
// end of cuda kernels
//---------------------------------------------------------------------

Net_GPU::Net_GPU() : trainRate(0.1), momentum(0.1), error(0)
{

}

void Net_GPU::AddLayer(std::shared_ptr<Layer_GPU> layer)
{
	layers.push_back(layer);
}

void Net_GPU::SetTrainRate(double trainRate)
{
	this->trainRate = trainRate;
	for (auto& layer : layers)
	{
		layer->SetTrainRate(trainRate);
	}
}

void Net_GPU::SetMomentum(double momentum)
{
	this->momentum = momentum;
	for (auto& layer : layers)
	{
		layer->SetMomentum(momentum);
	}
}

const double* Net_GPU::feedForward_GPU(const double* inputs)
{
	//get the first layer from the list and feed the input into it
	auto i = 0;
	auto firstLayer = layers.at(i++);
	double *d_inputs = nullptr;
	utils::CheckError(cudaMalloc((void**)&d_inputs, 1 * sizeof(double)), __LINE__);
	utils::CheckError(cudaMemcpy(d_inputs, inputs, 1 * sizeof(double), cudaMemcpyHostToDevice), __LINE__);
	firstLayer->FeedForward_GPU(d_inputs);
	const auto* output = firstLayer->Output_GPU();
	utils::CheckError(cudaMalloc((void**)&d_inputs, 1 * sizeof(double)), __LINE__);
	//let the data flow through the net
	for (; i < layers.size(); ++i)
	{
		auto& layer = layers.at(i);
		layer->FeedForward_GPU(output);
		output = layer->Output_GPU();
	}

	return output;
}

void Net_GPU::backPropagate_GPU(const double* targets)
{
	//get the last layer from the list
	int i = layers.size() - 1;
	auto lastLayer = layers.at(i--);
	lastLayer->BackPropagation_GPU(targets);

	//let the data flow through the net
	for (; i >= 0; --i)
	{
		auto& layer = layers.at(i);
		layer->BackPropagation_GPU(layers.at(i + 1));
	}
}

void Net_GPU::Train_GPU(const double* inputs, const double* targets)
{
	//let the input flow throught the net
	const double *output = feedForward_GPU(inputs);
	error = errorFunc(output, targets, layers.at(layers.size() - 1)->OutputSize());

	//backpropagate the calculated result
	backPropagate_GPU(targets);

	//update weights
	for (auto& layer : layers)
	{
		layer->UpdateWeights_GPU();
	}

}

const double* Net_GPU::Activate_GPU(const double *inputs)
{
	return  feedForward_GPU(inputs);
}

double Net_GPU::GetError() const
{
	return error;
}

double Net_GPU::errorFunc(const double* d_outputs, const double* targets, unsigned size)
{
	auto numOutputElements = size / (BLOCK_SIZE << 1);
	if (size % (BLOCK_SIZE << 1))
	{
		numOutputElements++;
	}

	auto *res = new double[size];
	double *d_res = nullptr;
	double *d_targets = nullptr;

	utils::CheckError(cudaMalloc((void **)&d_targets, size * sizeof(double)), __LINE__);
	utils::CheckError(cudaMalloc((void **)&d_res, numOutputElements * sizeof(double)), __LINE__);

	cudaMemcpy(d_targets, targets, size * sizeof(float), cudaMemcpyHostToDevice);

	dim3 DimGrid(numOutputElements, 1, 1);
	dim3 DimBlock(BLOCK_SIZE, 1, 1);

	cuda_errorFunc << <DimGrid, DimBlock >> >(d_outputs, d_targets, d_res, size);

	cudaMemcpy(res, d_res, numOutputElements * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 1; i < numOutputElements; i++)
	{
		res[0] += res[i];
	}

	error = res[0];

	cudaFree(d_res);
	cudaFree(d_targets);
	delete[] res;

	return error;
}


