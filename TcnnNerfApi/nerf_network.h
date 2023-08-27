/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

 /** @file   nerf_network.h
  *  @author Thomas Müller, NVIDIA
  *  @brief  A network that first processes 3D position to density and
  *          subsequently direction to color.
  */

// custom implementation based on https://github.com/NVlabs/instant-ngp/blob/master/include/neural-graphics-primitives/nerf_network.h
#pragma once

#include <cuda_runtime.h>
#include <common.h>
#include <encoding.h>
#include <gpu_matrix.h>
#include <memory.h>
#include <gpu_memory.h>
#include <multi_stream.h>
#include <network.h>
#include <network_with_input_encoding.h>

namespace ngp {

	using nerf_precision = tcnn::network_precision_t;

	template <typename K, typename T, typename ... Types>
	inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args) {
		if (n_elements <= 0) {
			return;
		}
		kernel << <tcnn::n_blocks_linear(n_elements), tcnn::n_threads_linear, shmem_size, stream >> > (n_elements, args...);
	}

	template <typename T>
	__global__ void extract_density(
		const uint32_t n_elements,
		const uint32_t density_stride,
		const uint32_t rgbd_stride,
		const T* __restrict__ density,
		T* __restrict__ rgbd
	) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		rgbd[i * rgbd_stride] = density[i * density_stride];
	}

	template <typename T>
	__global__ void extract_rgb(
		const uint32_t n_elements,
		const uint32_t rgb_stride,
		const uint32_t output_stride,
		const T* __restrict__ rgbd,
		T* __restrict__ rgb
	) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		const uint32_t elem_idx = i / 3;
		const uint32_t dim_idx = i - elem_idx * 3;

		rgb[elem_idx * rgb_stride + dim_idx] = rgbd[elem_idx * output_stride + dim_idx];
	}

	template <typename T>
	__global__ void add_density_gradient(
		const uint32_t n_elements,
		const uint32_t rgbd_stride,
		const T* __restrict__ rgbd,
		const uint32_t density_stride,
		T* __restrict__ density
	) {
		const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n_elements) return;

		density[i * density_stride] += rgbd[i * rgbd_stride + 3];
	}

	class NerfNetwork : public tcnn::Network<float, nerf_precision> {
	public:
		using json = nlohmann::json;

		NerfNetwork(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims, uint32_t dir_offset, const json& pos_encoding, const json& dir_encoding, const json& density_network, const json& rgb_network);

		virtual ~NerfNetwork();

		void inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<nerf_precision>& output, bool use_inference_params) override;

		uint32_t padded_density_output_width() const;

		std::unique_ptr<tcnn::Context> forward_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<nerf_precision>* output, bool use_inference_params, bool prepare_input_gradients) override;

		void backward_impl(
			cudaStream_t stream,
			const tcnn::Context& ctx,
			const tcnn::GPUMatrixDynamic<float>& input,
			const tcnn::GPUMatrixDynamic<nerf_precision>& output,
			const tcnn::GPUMatrixDynamic<nerf_precision>& dL_doutput,
			tcnn::GPUMatrixDynamic<float>* dL_dinput,
			bool use_inference_params,
			tcnn::EGradientMode param_gradients_mode
		) override;

		void density(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<nerf_precision>& output, bool use_inference_params);

		std::unique_ptr<tcnn::Context> density_forward(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<nerf_precision>* output, bool use_inference_params, bool prepare_input_gradients);

		void density_backward(
			cudaStream_t stream,
			const tcnn::Context& ctx,
			const tcnn::GPUMatrixDynamic<float>& input,
			const tcnn::GPUMatrixDynamic<nerf_precision>& output,
			const tcnn::GPUMatrixDynamic<nerf_precision>& dL_doutput,
			tcnn::GPUMatrixDynamic<float>* dL_dinput,
			bool use_inference_params,
			tcnn::EGradientMode param_gradients_mode
		);

		void set_params_impl(nerf_precision* params, nerf_precision* inference_params, nerf_precision* gradients) override;

		void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, float scale) override;

		size_t n_params() const override;

		uint32_t padded_output_width() const override;

		uint32_t input_width() const override;

		uint32_t output_width() const override;

		uint32_t n_extra_dims() const;

		uint32_t input_width_density() const;

		uint32_t required_input_alignment() const;

		std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const;

		uint32_t width(uint32_t layer) const override;

		uint32_t num_forward_activations() const override;

		std::pair<const nerf_precision*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const override;

		const std::shared_ptr<tcnn::Encoding<nerf_precision>>& pos_encoding() const;

		const std::shared_ptr<tcnn::Encoding<nerf_precision>>& dir_encoding() const;

		const std::shared_ptr<tcnn::Network<nerf_precision>>& density_network() const;

		const std::shared_ptr<tcnn::Network<nerf_precision>>& rgb_network() const;

		tcnn::json hyperparams() const override;

	private:
		std::shared_ptr<tcnn::Network<nerf_precision>> m_density_network;
		std::shared_ptr<tcnn::Network<nerf_precision>> m_rgb_network;
		std::shared_ptr<tcnn::Encoding<nerf_precision>> m_pos_encoding;
		std::shared_ptr<tcnn::Encoding<nerf_precision>> m_dir_encoding;

		uint32_t m_rgb_network_input_width;
		uint32_t m_n_pos_dims;
		uint32_t m_n_dir_dims;
		uint32_t m_n_extra_dims; // extra dimensions are assumed to be part of a compound encoding with dir_dims
		uint32_t m_dir_offset;

		// // Storage of forward pass data
		struct ForwardContext : public tcnn::Context {
			tcnn::GPUMatrixDynamic<nerf_precision> density_network_input;
			tcnn::GPUMatrixDynamic<nerf_precision> density_network_output;
			tcnn::GPUMatrixDynamic<nerf_precision> rgb_network_input;
			tcnn::GPUMatrix<nerf_precision> rgb_network_output;

			std::unique_ptr<Context> pos_encoding_ctx;
			std::unique_ptr<Context> dir_encoding_ctx;

			std::unique_ptr<Context> density_network_ctx;
			std::unique_ptr<Context> rgb_network_ctx;
		};
	};

}
