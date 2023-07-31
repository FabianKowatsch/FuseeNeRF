#pragma once
#include <tiny-cuda-nn/optimizer.h>
#pragma warning(push, 0)
#include "torch/torch.h"
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#pragma warning(pop)
#include "module.h"
namespace tcnnNerf {
	class Optimizer {
	public:
		Optimizer(const nlohmann::json& config);
		void allocate(uint32_t n_params, std::vector<std::pair<uint32_t, uint32_t>> layer_sizes);
		void step(float loss_scale, torch::Tensor weights, torch::Tensor gradients);
		nlohmann::json hyperparams();
	private:
		std::shared_ptr<tcnn::Optimizer<ngp::nerf_precision>> m_optimizer = nullptr;
	};
}