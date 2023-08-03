#pragma once
#include "cpp_api.h"
#include "nerf_module.h"
#pragma warning(push, 0)
#include "torch/torch.h"
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#pragma warning(pop)


namespace tcnnNerf {
	using namespace nerf;
	class Module {
	public:
		Module(NerfModule* module) {
			m_module = std::unique_ptr<NerfModule>(module);
		}
		tcnn::cpp::Context fwd(torch::Tensor input, torch::Tensor params, torch::Tensor output);
		void bwd(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor output, torch::Tensor dL_doutput, torch::Tensor dL_dparams);
		void density(torch::Tensor input, torch::Tensor params, torch::Tensor output);
		void inference(torch::Tensor input, torch::Tensor params, torch::Tensor output);
		torch::Tensor initial_params(size_t seed);
		uint32_t n_input_dims() const;
		uint32_t n_input_dims_density() const;
		uint32_t n_params() const;
		tcnn::cpp::EPrecision param_precision() const;
		c10::ScalarType c10_param_precision() const;
		uint32_t n_output_dims() const;
		uint32_t n_output_dims_density() const;
		tcnn::cpp::EPrecision output_precision() const;
		c10::ScalarType c10_output_precision() const;
		nlohmann::json hyperparams() const;
		std::string name() const;
		std::vector<std::pair<uint32_t, uint32_t>> layer_sizes();
	private:
		std::unique_ptr<NerfModule> m_module;
	};

	void* void_data_ptr(torch::Tensor& x);

	class ContextWrapper {
	public:
		ContextWrapper(std::unique_ptr<tcnn::Context> context) {
			ctx = { std::move(context) };
		}
		~ContextWrapper() {
			ctx.ctx.reset();
		}
		tcnn::cpp::Context ctx;
	};
}


