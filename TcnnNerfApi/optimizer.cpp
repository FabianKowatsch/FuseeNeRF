#include "optimizer.h"

namespace tcnnNerf {
	Optimizer::Optimizer(const nlohmann::json& config) {
		m_optimizer.reset(tcnn::create_optimizer<ngp::nerf_precision>(config));
	}
	void Optimizer::allocate(uint32_t n_params, std::vector<std::pair<uint32_t, uint32_t>> layer_sizes) {
		m_optimizer->allocate(n_params, layer_sizes);
	}
	void Optimizer::step(float loss_scale, torch::Tensor weights, torch::Tensor gradients) {

		m_optimizer->step(nullptr, loss_scale, nullptr, (ngp::nerf_precision*)void_data_ptr(weights), (ngp::nerf_precision*)void_data_ptr(gradients));
	}
	nlohmann::json Optimizer::hyperparams() {
		return m_optimizer->hyperparams();
	}
}