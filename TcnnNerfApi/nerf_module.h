/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 /** @file   cpp_api.h
  *  @author Thomas Müller, NVIDIA
  *  @brief  API to be consumed by cpp (non-CUDA) programs.
  */

// ustom implementation based on https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/cpp_api.h
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
