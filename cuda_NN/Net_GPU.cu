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

Net_GPU::~Net_GPU()
{
	for (auto i = 0; i < nrInputs; ++i)
	{
		cudaFree(inputs[i]);
		cudaFree(targets[i]);
	}
	delete[] inputs;
	delete[] targets;
}


void Net_GPU::InitInputs(double **inputs, double **targets, unsigned n, unsigned inputSize, unsigned targetSize)
{
	nrInputs = n;
	this->inputSize = inputSize;
	this->targetSize = targetSize;

	this->inputs = new double*[nrInputs];
	this->targets = new double*[nrInputs];

	for (auto i = 0; i < nrInputs; ++i)
	{
		utils::CheckError(cudaMalloc((void**)&this->inputs[i], inputSize * sizeof(double)), __FILE__, __LINE__);
		utils::CheckError(cudaMalloc((void**)&this->targets[i], targetSize * sizeof(double)), __FILE__, __LINE__);

		utils::CheckError(cudaMemcpy(this->inputs[i], inputs[i], inputSize * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
		utils::CheckError(cudaMemcpy(this->targets[i], targets[i], targetSize * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
	}
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

const double* Net_GPU::feedForward_GPU(const double* d_inputs)
{
	//get the first layer from the list and feed the input into it
	auto i = 0;
	auto firstLayer = layers.at(i++);
	firstLayer->FeedForward_GPU(d_inputs);
	const auto* output = firstLayer->Output_GPU();

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

void Net_GPU::Train_GPU(unsigned inputPos)
{
	//let the input flow throught the net
	const double *output = feedForward_GPU(inputs[inputPos]);
	error = errorFunc(output, targets[inputPos], layers.at(layers.size() - 1)->OutputSize());

	//backpropagate the calculated result
	backPropagate_GPU(targets[inputPos]);

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

double Net_GPU::errorFunc(const double* d_outputs, const double* d_targets, unsigned size)
{
	double *activ = new double[10];
	utils::CheckError(cudaMemcpy(activ, d_outputs, 10 * sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	//utils::PrintMatrix(activ, 1, 10);

	auto numOutputElements = size / (BLOCK_SIZE << 1);
	if (size % (BLOCK_SIZE << 1))
	{
		numOutputElements++;
	}

	auto *res = new double[size];
	double *d_res = nullptr;

	utils::CheckError(cudaMalloc((void **)&d_res, numOutputElements * sizeof(double)), __FILE__, __LINE__);

	dim3 DimGrid(numOutputElements, 1, 1);
	dim3 DimBlock(BLOCK_SIZE, 1, 1);

	cuda_errorFunc << <DimGrid, DimBlock >> >(d_outputs, d_targets, d_res, size);

	utils::CheckError(cudaMemcpy(res, d_res, numOutputElements * sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

	for (int i = 1; i < numOutputElements; i++)
	{
		res[0] += res[i];
	}

	auto error = res[0];

	cudaFree(d_res);
	delete[] res;

	return error;
}


