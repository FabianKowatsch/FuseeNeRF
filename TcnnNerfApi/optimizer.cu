#include "optimizer.h"
#define CHECK_INPUT(x) CHECK_THROW(x.device().is_cuda()); CHECK_THROW(x.is_contiguous())
namespace tcnnNerf {

	Optimizer::Optimizer(const nlohmann::json& config) {
		m_optimizer.reset(tcnn::create_optimizer<ngp::nerf_precision>(config));
	}
	void Optimizer::allocate(uint32_t n_params, std::vector<std::pair<uint32_t, uint32_t>> layer_sizes) {
		m_optimizer->allocate(n_params, layer_sizes);
	}
	void Optimizer::step(float loss_scale, torch::Tensor weights, torch::Tensor weights_fp, torch::Tensor gradients) {

		at::Device device = weights.device();
		CHECK_INPUT(weights);
		CHECK_INPUT(weights_fp);
		CHECK_INPUT(gradients);

		float* weights_fp_ptr = weights_fp.data_ptr<float>();
		ngp::nerf_precision* weights_ptr = (ngp::nerf_precision*)void_data_ptr(weights);
		const ngp::nerf_precision* gradients_ptr = (ngp::nerf_precision*)void_data_ptr(gradients);
		

		const at::cuda::CUDAGuard device_guard{ device };
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		m_optimizer->step(stream, loss_scale, weights_fp_ptr, weights_ptr, gradients_ptr);

		cudaDeviceSynchronize();
	}
	nlohmann::json Optimizer::hyperparams() {
		return m_optimizer->hyperparams();
	}
}