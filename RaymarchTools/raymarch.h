#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#pragma warning(push, 0)
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#pragma warning(pop)

#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


void near_far_from_aabb(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor aabb, const uint32_t N, const float min_near, at::Tensor nears, at::Tensor fars);
void sph_from_ray(const at::Tensor rays_o, const at::Tensor rays_d, const float radius, const uint32_t N, at::Tensor coords);
void morton3D(const at::Tensor coords, const uint32_t N, at::Tensor indices);
void morton3D_invert(const at::Tensor indices, const uint32_t N, at::Tensor coords);
void packbits(const at::Tensor grid, const uint32_t N, const float density_thresh, at::Tensor bitfield);

void march_rays_train(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor grid, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor rays, at::Tensor counter, at::Tensor noises);
void composite_rays_train_forward(const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, const at::Tensor rays, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor weights_sum, at::Tensor depth, at::Tensor image);
void composite_rays_train_backward(const at::Tensor grad_weights_sum, const at::Tensor grad_image, const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, const at::Tensor rays, const at::Tensor weights_sum, const at::Tensor image, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor grad_sigmas, at::Tensor grad_rgbs);

void march_rays(const uint32_t n_alive, const uint32_t n_step, const at::Tensor rays_alive, const at::Tensor rays_t, const at::Tensor rays_o, const at::Tensor rays_d, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t C, const uint32_t H, const at::Tensor grid, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor noises);
void composite_rays(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor weights_sum, at::Tensor depth, at::Tensor image);