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

/** @file   cpp_api.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  API to be consumed by cpp (non-CUDA) programs.
 */
#include <common.h>
#include <cpp_api.h>
#include <encoding.h>
#include <multi_stream.h>
#include "nerf_module.h"
#include <pcg32.h>


namespace nerf {
		NerfModule::NerfModule(ngp::NerfNetwork* model) : m_network{ model }, m_network_nerf{ model } {}
		
		void NerfModule::inference(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params) {
			m_network->set_params((ngp::nerf_precision*)params, (ngp::nerf_precision*)params, nullptr);

			tcnn::GPUMatrix<float, tcnn::MatrixLayout::ColumnMajor> input_matrix((float*)input, m_network->input_width(), n_elements);
			tcnn::GPUMatrix<ngp::nerf_precision, tcnn::MatrixLayout::ColumnMajor> output_matrix((ngp::nerf_precision*)output, m_network->padded_output_width(), n_elements);

			// Run on our own custom stream to ensure CUDA graph capture is possible.
			// (Significant possible speedup.)
			tcnn::SyncedMultiStream synced_stream{ stream, 2 };
			m_network->inference_mixed_precision(synced_stream.get(1), input_matrix, output_matrix);
		}

		tcnn::cpp::Context NerfModule::forward(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params, bool prepare_input_gradients) {
			m_network->set_params((ngp::nerf_precision*)params, (ngp::nerf_precision*)params, nullptr);

			tcnn::GPUMatrix<float, tcnn::MatrixLayout::ColumnMajor> input_matrix((float*)input, m_network->input_width(), n_elements);
			tcnn::GPUMatrix<ngp::nerf_precision, tcnn::MatrixLayout::ColumnMajor> output_matrix((ngp::nerf_precision*)output, m_network->padded_output_width(), n_elements);

			// Run on our own custom stream to ensure CUDA graph capture is possible.
			// (Significant possible speedup.)
			tcnn::SyncedMultiStream synced_stream{ stream, 2 };
			return { m_network->forward(synced_stream.get(1), input_matrix, &output_matrix, false, prepare_input_gradients) };
		}

		void NerfModule::backward(cudaStream_t stream, const tcnn::cpp::Context& ctx, uint32_t n_elements, float* dL_dinput, const void* dL_doutput, void* dL_dparams, const float* input, const void* output, const void* params) {
			m_network->set_params((ngp::nerf_precision*)params, (ngp::nerf_precision*)params, (ngp::nerf_precision*)dL_dparams);

			tcnn::GPUMatrix<float, tcnn::MatrixLayout::ColumnMajor> input_matrix((float*)input, m_network->input_width(), n_elements);
			tcnn::GPUMatrix<float, tcnn::MatrixLayout::ColumnMajor> dL_dinput_matrix(dL_dinput, m_network->input_width(), n_elements);

			tcnn::GPUMatrix<ngp::nerf_precision, tcnn::MatrixLayout::ColumnMajor> output_matrix((ngp::nerf_precision*)output, m_network->padded_output_width(), n_elements);
			tcnn::GPUMatrix<ngp::nerf_precision, tcnn::MatrixLayout::ColumnMajor> dL_doutput_matrix((ngp::nerf_precision*)dL_doutput, m_network->padded_output_width(), n_elements);

			// Run on our own custom stream to ensure CUDA graph capture is possible.
			// (Significant possible speedup.)
			tcnn::SyncedMultiStream synced_stream{ stream, 2 };
			m_network->backward(synced_stream.get(1), *ctx.ctx, input_matrix, output_matrix, dL_doutput_matrix, dL_dinput ? &dL_dinput_matrix : nullptr, false, dL_dparams ? tcnn::EGradientMode::Overwrite : tcnn::EGradientMode::Ignore);
		}


		void NerfModule::density(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params) {
			m_network->set_params((ngp::nerf_precision*)params, (ngp::nerf_precision*)params, nullptr);

			tcnn::GPUMatrix<float, tcnn::MatrixLayout::ColumnMajor> input_matrix((float*)input, m_network_nerf->input_width_density(), n_elements);
			tcnn::GPUMatrix<ngp::nerf_precision, tcnn::MatrixLayout::ColumnMajor> output_matrix((ngp::nerf_precision*)output, m_network_nerf->padded_density_output_width(), n_elements);

			// Run on our own custom stream to ensure CUDA graph capture is possible.
			// (Significant possible speedup.)
			tcnn::SyncedMultiStream synced_stream{ stream, 2 };
			m_network_nerf->density(synced_stream.get(1), input_matrix, output_matrix, true);
		}

		uint32_t NerfModule::n_input_dims() const {
			return m_network->input_width();
		}
		uint32_t NerfModule::n_input_dims_density() const {
			return m_network_nerf->input_width_density();
		}

		size_t NerfModule::n_params() const {
			return m_network->n_params();
		}
		void NerfModule::initialize_params(size_t seed, float* params_full_precision, float scale) {
			tcnn::pcg32 rng{ seed };
			m_network->initialize_params(rng, params_full_precision, scale);
		}
		std::vector<std::pair<uint32_t, uint32_t>> NerfModule::layer_sizes() {
			return m_network->layer_sizes();
		}
		uint32_t NerfModule::n_output_dims() const {		
			return m_network->padded_output_width();
		}
		uint32_t NerfModule::n_output_dims_density() const {
			return m_network_nerf->padded_density_output_width();
		}
		json NerfModule::hyperparams() const {
			return m_network->hyperparams();
		}
		std::string NerfModule::name() const {
			return m_network->name();
		}

}



