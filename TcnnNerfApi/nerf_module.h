#pragma once
#include "nerf_network.h"
#include <json.hpp>

using json = nlohmann::json;
namespace nerf {

	class NerfModule {

	public:
		NerfModule(ngp::NerfNetwork* model);
		~NerfModule() {}

		void inference(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params);
		tcnn::cpp::Context forward(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params, bool prepare_input_gradients);
		void backward(cudaStream_t stream, const tcnn::cpp::Context& ctx, uint32_t n_elements, float* dL_dinput, const void* dL_doutput, void* dL_dparams, const float* input, const void* output, const void* params);
		void density(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params);
		uint32_t n_input_dims() const;
		uint32_t n_output_dims() const;
		uint32_t n_input_dims_density() const;
		uint32_t n_output_dims_density() const;
		size_t n_params() const;
		std::vector<std::pair<uint32_t, uint32_t>> layer_sizes();
		void initialize_params(size_t seed, float* params_full_precision, float scale = 1.0f);

		nlohmann::json hyperparams() const;
		std::string name() const;

	private:
		std::shared_ptr<tcnn::Network<float, ngp::nerf_precision>> m_network;
		std::shared_ptr<ngp::NerfNetwork> m_network_nerf;
	};

}
