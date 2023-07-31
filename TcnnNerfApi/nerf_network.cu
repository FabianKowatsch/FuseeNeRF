#include "nerf_network.h"

namespace ngp {

		using json = nlohmann::json;
		NerfNetwork::NerfNetwork(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims, uint32_t dir_offset, const json& pos_encoding, const json& dir_encoding, const json& density_network, const json& rgb_network) : m_n_pos_dims{ n_pos_dims }, m_n_dir_dims{ n_dir_dims }, m_dir_offset{ dir_offset }, m_n_extra_dims{ n_extra_dims } {
			m_pos_encoding.reset(tcnn::create_encoding<nerf_precision>(n_pos_dims, pos_encoding, density_network.contains("otype") && (tcnn::equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || tcnn::equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u));
			uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);
			m_dir_encoding.reset(tcnn::create_encoding<nerf_precision>(m_n_dir_dims + m_n_extra_dims, dir_encoding, rgb_alignment));

			json local_density_network_config = density_network;
			local_density_network_config["n_input_dims"] = m_pos_encoding->padded_output_width();
			if (!density_network.contains("n_output_dims")) {
				local_density_network_config["n_output_dims"] = 16;
			}
			m_density_network.reset(tcnn::create_network<nerf_precision>(local_density_network_config));

			m_rgb_network_input_width = tcnn::next_multiple(m_dir_encoding->padded_output_width() + m_density_network->padded_output_width(), rgb_alignment);

			json local_rgb_network_config = rgb_network;
			local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
			local_rgb_network_config["n_output_dims"] = 3;
			m_rgb_network.reset(tcnn::create_network<nerf_precision>(local_rgb_network_config));
		}

		NerfNetwork::~NerfNetwork() {}

		void NerfNetwork::inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<nerf_precision>& output, bool use_inference_params = true) {
			uint32_t batch_size = input.n();
			tcnn::GPUMatrixDynamic<nerf_precision> density_network_input{ m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout() };
			tcnn::GPUMatrixDynamic<nerf_precision> rgb_network_input{ m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout() };

			tcnn::GPUMatrixDynamic<nerf_precision> density_network_output = rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
			tcnn::GPUMatrixDynamic<nerf_precision> rgb_network_output{ output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout() };

			m_pos_encoding->inference_mixed_precision(
				stream,
				input.slice_rows(0, m_pos_encoding->input_width()),
				density_network_input,
				use_inference_params
			);

			m_density_network->inference_mixed_precision(stream, density_network_input, density_network_output, use_inference_params);

			auto dir_out = rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			m_dir_encoding->inference_mixed_precision(
				stream,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				dir_out,
				use_inference_params
			);

			m_rgb_network->inference_mixed_precision(stream, rgb_network_input, rgb_network_output, use_inference_params);

			linear_kernel(extract_density<nerf_precision>, 0, stream,
				batch_size,
				density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
				output.layout() == tcnn::AoS ? padded_output_width() : 1,
				density_network_output.data(),
				output.data() + 3 * (output.layout() == tcnn::AoS ? 1 : batch_size)
			);
		}

		uint32_t NerfNetwork::padded_density_output_width() const {
			return m_density_network->padded_output_width();
		}

		std::unique_ptr<tcnn::Context> NerfNetwork::forward_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<nerf_precision>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) {
			// Make sure our temporary buffers have the correct size for the given batch size
			uint32_t batch_size = input.n();

			auto forward = std::make_unique<ForwardContext>();

			forward->density_network_input = tcnn::GPUMatrixDynamic<nerf_precision>{ m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout() };
			forward->rgb_network_input = tcnn::GPUMatrixDynamic<nerf_precision>{ m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout() };

			forward->pos_encoding_ctx = m_pos_encoding->forward(
				stream,
				input.slice_rows(0, m_pos_encoding->input_width()),
				&forward->density_network_input,
				use_inference_params,
				prepare_input_gradients
			);

			forward->density_network_output = forward->rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
			forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, prepare_input_gradients);

			auto dir_out = forward->rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			forward->dir_encoding_ctx = m_dir_encoding->forward(
				stream,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				&dir_out,
				use_inference_params,
				prepare_input_gradients
			);

			if (output) {
				forward->rgb_network_output = tcnn::GPUMatrixDynamic<nerf_precision>{ output->data(), m_rgb_network->padded_output_width(), batch_size, output->layout() };
			}

			forward->rgb_network_ctx = m_rgb_network->forward(stream, forward->rgb_network_input, output ? &forward->rgb_network_output : nullptr, use_inference_params, prepare_input_gradients);

			if (output) {
				linear_kernel(extract_density<nerf_precision>, 0, stream,
					batch_size, m_dir_encoding->preferred_output_layout() == tcnn::AoS ? forward->density_network_output.stride() : 1, padded_output_width(), forward->density_network_output.data(), output->data() + 3
				);
			}

			return forward;
		}
		void NerfNetwork::backward_impl(
			cudaStream_t stream,
			const tcnn::Context& ctx,
			const tcnn::GPUMatrixDynamic<float>& input,
			const tcnn::GPUMatrixDynamic<nerf_precision>& output,
			const tcnn::GPUMatrixDynamic<nerf_precision>& dL_doutput,
			tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
			bool use_inference_params = false,
			tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
		) {
			const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

			// Make sure our teporary buffers have the correct size for the given batch size
			uint32_t batch_size = input.n();

			tcnn::GPUMatrix<nerf_precision> dL_drgb{ m_rgb_network->padded_output_width(), batch_size, stream };
			CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream));
			linear_kernel(extract_rgb<nerf_precision>, 0, stream,
				batch_size * 3, dL_drgb.m(), dL_doutput.m(), dL_doutput.data(), dL_drgb.data()
			);

			const tcnn::GPUMatrixDynamic<nerf_precision> rgb_network_output{ (nerf_precision*)output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout() };
			tcnn::GPUMatrixDynamic<nerf_precision> dL_drgb_network_input{ m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout() };
			m_rgb_network->backward(stream, *forward.rgb_network_ctx, forward.rgb_network_input, rgb_network_output, dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode);

			// Backprop through dir encoding if it is trainable or if we need input gradients
			if (m_dir_encoding->n_params() > 0 || dL_dinput) {
				tcnn::GPUMatrixDynamic<nerf_precision> dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
				tcnn::GPUMatrixDynamic<float> dL_ddir_encoding_input;
				if (dL_dinput) {
					dL_ddir_encoding_input = dL_dinput->slice_rows(m_dir_offset, m_dir_encoding->input_width());
				}

				m_dir_encoding->backward(
					stream,
					*forward.dir_encoding_ctx,
					input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
					forward.rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width()),
					dL_ddir_encoding_output,
					dL_dinput ? &dL_ddir_encoding_input : nullptr,
					use_inference_params,
					param_gradients_mode
				);
			}

			tcnn::GPUMatrixDynamic<nerf_precision> dL_ddensity_network_output = dL_drgb_network_input.slice_rows(0, m_density_network->padded_output_width());
			linear_kernel(add_density_gradient<nerf_precision>, 0, stream,
				batch_size,
				dL_doutput.m(),
				dL_doutput.data(),
				dL_ddensity_network_output.layout() == tcnn::RM ? 1 : dL_ddensity_network_output.stride(),
				dL_ddensity_network_output.data()
			);

			tcnn::GPUMatrixDynamic<nerf_precision> dL_ddensity_network_input;
			if (m_pos_encoding->n_params() > 0 || dL_dinput) {
				dL_ddensity_network_input = tcnn::GPUMatrixDynamic<nerf_precision>{ m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout() };
			}

			m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, forward.density_network_output, dL_ddensity_network_output, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);

			// Backprop through pos encoding if it is trainable or if we need input gradients
			if (dL_ddensity_network_input.data()) {
				tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
				if (dL_dinput) {
					dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
				}

				m_pos_encoding->backward(
					stream,
					*forward.pos_encoding_ctx,
					input.slice_rows(0, m_pos_encoding->input_width()),
					forward.density_network_input,
					dL_ddensity_network_input,
					dL_dinput ? &dL_dpos_encoding_input : nullptr,
					use_inference_params,
					param_gradients_mode
				);
			}
		}
		void NerfNetwork::density(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<nerf_precision>& output, bool use_inference_params = true) {
			if (input.layout() != tcnn::CM) {
				throw std::runtime_error("NerfNetwork::density input must be in column major format.");
			}

			uint32_t batch_size = output.n();
			tcnn::GPUMatrixDynamic<nerf_precision> density_network_input{ m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout() };

			m_pos_encoding->inference_mixed_precision(
				stream,
				input.slice_rows(0, m_pos_encoding->input_width()),
				density_network_input,
				use_inference_params
			);

			m_density_network->inference_mixed_precision(stream, density_network_input, output, use_inference_params);
		}
		std::unique_ptr<tcnn::Context> NerfNetwork::density_forward(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<nerf_precision>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) {
			if (input.layout() != tcnn::CM) {
				throw std::runtime_error("NerfNetwork::density_forward input must be in column major format.");
			}

			// Make sure our temporary buffers have the correct size for the given batch size
			uint32_t batch_size = input.n();

			auto forward = std::make_unique<ForwardContext>();

			forward->density_network_input = tcnn::GPUMatrixDynamic<nerf_precision>{ m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout() };

			forward->pos_encoding_ctx = m_pos_encoding->forward(
				stream,
				input.slice_rows(0, m_pos_encoding->input_width()),
				&forward->density_network_input,
				use_inference_params,
				prepare_input_gradients
			);

			if (output) {
				forward->density_network_output = tcnn::GPUMatrixDynamic<nerf_precision>{ output->data(), m_density_network->padded_output_width(), batch_size, output->layout() };
			}

			forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, output ? &forward->density_network_output : nullptr, use_inference_params, prepare_input_gradients);

			return forward;
		}
		void NerfNetwork::density_backward(
			cudaStream_t stream,
			const tcnn::Context& ctx,
			const tcnn::GPUMatrixDynamic<float>& input,
			const tcnn::GPUMatrixDynamic<nerf_precision>& output,
			const tcnn::GPUMatrixDynamic<nerf_precision>& dL_doutput,
			tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
			bool use_inference_params = false,
			tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
		) {
			if (input.layout() != tcnn::CM || (dL_dinput && dL_dinput->layout() != tcnn::CM)) {
				throw std::runtime_error("NerfNetwork::density_backward input must be in column major format.");
			}

			const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

			// Make sure our temporary buffers have the correct size for the given batch size
			uint32_t batch_size = input.n();

			tcnn::GPUMatrixDynamic<nerf_precision> dL_ddensity_network_input;
			if (m_pos_encoding->n_params() > 0 || dL_dinput) {
				dL_ddensity_network_input = tcnn::GPUMatrixDynamic<nerf_precision>{ m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout() };
			}

			m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, output, dL_doutput, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);

			// Backprop through pos encoding if it is trainable or if we need input gradients
			if (dL_ddensity_network_input.data()) {
				tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
				if (dL_dinput) {
					dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
				}

				m_pos_encoding->backward(
					stream,
					*forward.pos_encoding_ctx,
					input.slice_rows(0, m_pos_encoding->input_width()),
					forward.density_network_input,
					dL_ddensity_network_input,
					dL_dinput ? &dL_dpos_encoding_input : nullptr,
					use_inference_params,
					param_gradients_mode
				);
			}
		}
		void NerfNetwork::set_params_impl(nerf_precision* params, nerf_precision* inference_params, nerf_precision* gradients) {
			size_t offset = 0;
			m_density_network->set_params(params + offset, inference_params + offset, gradients + offset);
			offset += m_density_network->n_params();

			m_rgb_network->set_params(params + offset, inference_params + offset, gradients + offset);
			offset += m_rgb_network->n_params();

			m_pos_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
			offset += m_pos_encoding->n_params();

			m_dir_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
			offset += m_dir_encoding->n_params();
		}
		void NerfNetwork::initialize_params(tcnn::pcg32& rnd, float* params_full_precision, float scale = 1) {
			m_density_network->initialize_params(rnd, params_full_precision, scale);
			params_full_precision += m_density_network->n_params();

			m_rgb_network->initialize_params(rnd, params_full_precision, scale);
			params_full_precision += m_rgb_network->n_params();

			m_pos_encoding->initialize_params(rnd, params_full_precision, scale);
			params_full_precision += m_pos_encoding->n_params();

			m_dir_encoding->initialize_params(rnd, params_full_precision, scale);
			params_full_precision += m_dir_encoding->n_params();
		}

		size_t NerfNetwork::n_params() const {
			return m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params();
		}

		uint32_t NerfNetwork::padded_output_width() const {
			return std::max(m_rgb_network->padded_output_width(), (uint32_t)4);
		}

		uint32_t NerfNetwork::input_width() const {
			return m_dir_offset + m_n_dir_dims + m_n_extra_dims;
		}

		uint32_t NerfNetwork::input_width_density() const {
			return m_pos_encoding->input_width();
		}

		uint32_t NerfNetwork::output_width() const {
			return 4;
		}

		uint32_t NerfNetwork::n_extra_dims() const {
			return m_n_extra_dims;
		}

		uint32_t NerfNetwork::required_input_alignment() const {
			return 1; // No alignment required due to encoding
		}
		std::vector<std::pair<uint32_t, uint32_t>> NerfNetwork::layer_sizes() const {
			auto layers = m_density_network->layer_sizes();
			auto rgb_layers = m_rgb_network->layer_sizes();
			layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
			return layers;
		}
		uint32_t NerfNetwork::width(uint32_t layer) const {
			if (layer == 0) {
				return m_pos_encoding->padded_output_width();
			}
			else if (layer < m_density_network->num_forward_activations() + 1) {
				return m_density_network->width(layer - 1);
			}
			else if (layer == m_density_network->num_forward_activations() + 1) {
				return m_rgb_network_input_width;
			}
			else {
				return m_rgb_network->width(layer - 2 - m_density_network->num_forward_activations());
			}
		}
		uint32_t NerfNetwork::num_forward_activations() const {
			return m_density_network->num_forward_activations() + m_rgb_network->num_forward_activations() + 2;
		}

		std::pair<const nerf_precision*, tcnn::MatrixLayout> NerfNetwork::forward_activations(const tcnn::Context& ctx, uint32_t layer) const {
			const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
			if (layer == 0) {
				return { forward.density_network_input.data(), m_pos_encoding->preferred_output_layout() };
			}
			else if (layer < m_density_network->num_forward_activations() + 1) {
				return m_density_network->forward_activations(*forward.density_network_ctx, layer - 1);
			}
			else if (layer == m_density_network->num_forward_activations() + 1) {
				return { forward.rgb_network_input.data(), m_dir_encoding->preferred_output_layout() };
			}
			else {
				return m_rgb_network->forward_activations(*forward.rgb_network_ctx, layer - 2 - m_density_network->num_forward_activations());
			}
		}
		const std::shared_ptr<tcnn::Encoding<nerf_precision>>& NerfNetwork::pos_encoding() const {
			return m_pos_encoding;
		}
		const std::shared_ptr<tcnn::Encoding<nerf_precision>>& NerfNetwork::dir_encoding() const {
			return m_dir_encoding;
		}

		const std::shared_ptr<tcnn::Network<nerf_precision>>& NerfNetwork::density_network() const {
			return m_density_network;
		}

		const std::shared_ptr<tcnn::Network<nerf_precision>>& NerfNetwork::rgb_network() const {
			return m_rgb_network;
		}

		tcnn::json NerfNetwork::hyperparams() const {
			json density_network_hyperparams = m_density_network->hyperparams();
			density_network_hyperparams["n_output_dims"] = m_density_network->padded_output_width();
			return {
				{"otype", "NerfNetwork"},
				{"pos_encoding", m_pos_encoding->hyperparams()},
				{"dir_encoding", m_dir_encoding->hyperparams()},
				{"density_network", density_network_hyperparams},
				{"rgb_network", m_rgb_network->hyperparams()},
			};
		}


}