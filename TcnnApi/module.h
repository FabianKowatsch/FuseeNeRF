#pragma once
#include "cpp_api.h"
#pragma warning(push, 0)
#include "torch/torch.h"
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#pragma warning(pop)


namespace tcnnModule {
	class Module {
	public:
		Module(tcnn::cpp::Module* module) {
			m_module = std::unique_ptr<tcnn::cpp::Module>(module);
		}
		std::tuple<tcnn::cpp::Context, torch::Tensor> fwd(torch::Tensor input, torch::Tensor params);
		std::tuple<torch::Tensor, torch::Tensor> bwd(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor output, torch::Tensor dL_doutput);
		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bwd_bwd_input(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor dL_ddLdinput, torch::Tensor dL_doutput);
		torch::Tensor initial_params(size_t seed);
		uint32_t n_input_dims() const;
		uint32_t n_params() const;
		tcnn::cpp::EPrecision param_precision() const;
		c10::ScalarType c10_param_precision() const;
		uint32_t n_output_dims() const;
		tcnn::cpp::EPrecision output_precision() const;
		c10::ScalarType c10_output_precision() const;
		nlohmann::json hyperparams() const;
		std::string name() const;
	private:
		std::unique_ptr<tcnn::cpp::Module> m_module;
	};

	Module* createNetworkWithInputEncoding(uint32_t n_input_dims, uint32_t n_output_dims, const nlohmann::json& encoding, const nlohmann::json& network);

	Module* createNetwork(uint32_t n_input_dims, uint32_t n_output_dims, const nlohmann::json& network);

	Module* createEncoding(uint32_t n_input_dims, const nlohmann::json& encoding, tcnn::cpp::EPrecision requested_precision);

	void* void_data_ptr(torch::Tensor& x);

	class ContextWrapper {
	public:
		ContextWrapper(std::unique_ptr<tcnn::Context> context) {
			ctx = { std::move(context) };
		}
		tcnn::cpp::Context ctx;
	};
}

