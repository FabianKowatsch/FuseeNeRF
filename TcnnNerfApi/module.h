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

 /** @file   torch_bindings.cu
  *  @author Thomas Müller, Jacob Munkberg, Jon Hasselgren, Or Perel, NVIDIA
  */

  // custom implementation based on https://github.com/NVlabs/tiny-cuda-nn/blob/master/bindings/torch/tinycudann/bindings.cpp
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


