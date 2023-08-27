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

 /** @file   optimizer.h
  *  @author Thomas Müller, NVIDIA
  *  @brief  API for optimizers
*/
// custom implementation based on https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/optimizer.h
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