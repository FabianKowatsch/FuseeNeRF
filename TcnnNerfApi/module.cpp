#include "module.h"
//#include <typeinfo>

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CHECK_THROW(x) \
	do { if (!(x)) throw std::runtime_error(std::string(FILE_LINE " check failed " #x)); } while(0)

#define CHECK_INPUT(x) CHECK_THROW(x.device().is_cuda()); CHECK_THROW(x.is_contiguous())
using namespace torch;

namespace tcnnNerf {

	c10::ScalarType torch_type(tcnn::cpp::EPrecision precision) {
		switch (precision) {
		case tcnn::cpp::EPrecision::Fp32: return torch::kFloat32;
		case tcnn::cpp::EPrecision::Fp16: return torch::kHalf;
		default: throw std::runtime_error{ "Unknown precision tcnn->torch" };
		}
	}

	void* void_data_ptr(torch::Tensor& tensor) {
		switch (tensor.scalar_type()) {
		case torch::kFloat32: return tensor.data_ptr<float>();
		case torch::kHalf: return tensor.data_ptr<torch::Half>();
		default: throw std::runtime_error{ "Unknown precision torch->void" };
		}
	}

	tcnn::cpp::Context Module::fwd(torch::Tensor input, torch::Tensor params, torch::Tensor output) {

		CHECK_INPUT(input);
		CHECK_INPUT(params);
		CHECK_INPUT(output);

		// Types
		CHECK_THROW(input.scalar_type() == torch::kFloat32);
		CHECK_THROW(params.scalar_type() == c10_param_precision());
		CHECK_THROW(output.scalar_type() == c10_output_precision());

		// Sizes
		CHECK_THROW(input.size(1) == n_input_dims());
		CHECK_THROW(params.size(0) == n_params());
		CHECK_THROW(output.size(1) == n_output_dims());
		// Device
		at::Device device = input.device();
		CHECK_THROW(device == params.device());
		CHECK_THROW(device == output.device());

		const at::cuda::CUDAGuard device_guard{ device };
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		uint32_t batch_size = (uint32_t)input.size(0);
		//torch::Tensor output = torch::empty({ batch_size, n_output_dims() }, torch::TensorOptions().dtype(c10_output_precision()).device(device).requires_grad(params.requires_grad()));

		tcnn::cpp::Context ctx;

		if (!input.requires_grad() && !params.requires_grad()) {
			m_module->inference(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params));
		}
		else {
			ctx = m_module->forward(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params), input.requires_grad());
		}
		return ctx;
	}

	void Module::bwd(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor output, torch::Tensor dL_doutput, torch::Tensor dL_dparams) {
		if (!ctx.ctx) {
			throw std::runtime_error{ "Module::bwd: called with invalid context. fwd likely (mistakenly) ran in inference mode." };
		}

		CHECK_INPUT(input);
		CHECK_INPUT(params);
		CHECK_INPUT(output);
		CHECK_INPUT(dL_doutput);
		CHECK_INPUT(dL_dparams);

		// Types
		CHECK_THROW(input.scalar_type() == torch::kFloat32);
		CHECK_THROW(params.scalar_type() == c10_param_precision());
		CHECK_THROW(output.scalar_type() == c10_output_precision());
		CHECK_THROW(dL_doutput.scalar_type() == c10_output_precision());
		CHECK_THROW(dL_dparams.scalar_type() == c10_param_precision());

		// Sizes
		CHECK_THROW(input.size(1) == n_input_dims());
		CHECK_THROW(output.size(1) == n_output_dims());
		CHECK_THROW(params.size(0) == n_params());
		CHECK_THROW(output.size(0) == input.size(0));
		CHECK_THROW(dL_doutput.size(0) == input.size(0));
		CHECK_THROW(dL_dparams.size(0) == n_params());

		// Device
		at::Device device = input.device();
		CHECK_THROW(device == params.device());
		CHECK_THROW(device == output.device());
		CHECK_THROW(device == dL_doutput.device());
		CHECK_THROW(device == dL_dparams.device());

		const at::cuda::CUDAGuard device_guard{ device };
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		uint32_t batch_size = (uint32_t)input.size(0);

		/*
		//torch::Tensor dL_dinput;
		if (input.requires_grad()) {
			dL_dinput = torch::empty({ batch_size, input.size(1) }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
		}
		//torch::Tensor dL_dparams;
		if (params.requires_grad()) {
			dL_dparams = torch::empty({ n_params() }, torch::TensorOptions().dtype(c10_param_precision()).device(device));
		}
		*/
		if (params.requires_grad()) {
			m_module->backward(
				stream,
				ctx,
				batch_size,
				nullptr,
				void_data_ptr(dL_doutput),
				params.requires_grad() ? void_data_ptr(dL_dparams) : nullptr,
				input.data_ptr<float>(),
				void_data_ptr(output),
				void_data_ptr(params)
			);
		}
	}

	 void Module::density(torch::Tensor input, torch::Tensor params, torch::Tensor output) {

		CHECK_INPUT(input);
		CHECK_INPUT(params);
		CHECK_INPUT(output);

		// Types
		CHECK_THROW(input.scalar_type() == torch::kFloat32);
		CHECK_THROW(params.scalar_type() == c10_param_precision());
		CHECK_THROW(output.scalar_type() == c10_output_precision());

		// Sizes
		CHECK_THROW(input.size(1) == n_input_dims_density());
		CHECK_THROW(params.size(0) == n_params());
		CHECK_THROW(output.size(1) == n_output_dims_density());
		// Device
		at::Device device = input.device();
		CHECK_THROW(device == params.device());
		CHECK_THROW(device == output.device());

		const at::cuda::CUDAGuard device_guard{ device };
		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		uint32_t batch_size = (uint32_t)input.size(0);
		//torch::Tensor output = torch::empty({ batch_size, n_output_dims_density() }, torch::TensorOptions().dtype(c10_output_precision()).device(device).requires_grad(params.requires_grad()));

		m_module->density(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params));
	}

	torch::Tensor Module::initial_params(size_t seed) {
		torch::Tensor output = torch::zeros({ n_params() }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
		m_module->initialize_params(seed, output.data_ptr<float>());
		return output;
	}

	uint32_t Module::n_input_dims() const {
		return m_module->n_input_dims();
	}

	uint32_t Module::n_input_dims_density() const {
		return m_module->n_input_dims_density();
	}

	uint32_t Module::n_params() const {
		return (uint32_t)m_module->n_params();
	}

	c10::ScalarType Module::c10_param_precision() const {
		return torch_type(param_precision());
	}

	uint32_t Module::n_output_dims() const {
		return m_module->n_output_dims();
	}

	uint32_t Module::n_output_dims_density() const {
		return m_module->n_output_dims_density();
	}

	c10::ScalarType Module::c10_output_precision() const {
		return torch_type(output_precision());
	}

	nlohmann::json Module::hyperparams() const {
		return m_module->hyperparams();
	}

	std::string Module::name() const {
		return m_module->name();
	}

	std::vector<std::pair<uint32_t, uint32_t>> Module::layer_sizes() {
		return m_module->layer_sizes();
	}

	tcnn::cpp::EPrecision Module::output_precision() const {
		if (std::is_same<ngp::nerf_precision, float>::value) {
			return tcnn::cpp::Fp32;
		}
		else {
			return tcnn::cpp::Fp16;
		}
	}

	tcnn::cpp::EPrecision Module::param_precision() const {
		if (std::is_same<ngp::nerf_precision, float>::value) {
			return tcnn::cpp::Fp32;
		}
		else {
			return tcnn::cpp::Fp16;
		}
	}
}
