/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

 /** @file   nerf_device.cuh
  *  @author Thomas M�ller & Alex Evans, NVIDIA
  */

  // functions based on https://github.com/NVlabs/instant-ngp/blob/master/src/testbed_nerf.cu

#include "raymarch_shared.h"
#include "ray_sampler_header.h"
#include "common.h"

template <typename TYPE>
__global__ void compacted_coord_cuda(
    const uint32_t n_rays,
    BoundingBox aabb,
    const uint32_t max_samples_compacted,
    const TYPE *network_output,
    ENerfActivation density_activation,
    const NerfCoordinate *__restrict__ coords_in,
    NerfCoordinate *__restrict__ coords_out,
    const uint32_t *__restrict__ numsteps_in,
    uint32_t *__restrict__ numsteps_counter,
    uint32_t *__restrict__ numsteps_out,
    uint32_t *compacted_rays_counter)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n_rays)
    {
        return;
    }

    uint32_t numsteps = numsteps_in[i * 2 + 0];
    uint32_t base = numsteps_in[i * 2 + 1];
    coords_in += base;
    network_output += base;

    float T = 1.f;
    float EPSILON = 1e-4f;
    uint32_t compacted_numsteps = 0;
    for (; compacted_numsteps < numsteps; ++compacted_numsteps)
    {
        if (T < EPSILON)
        {
            break;
        }
        const float dt = unwarp_dt(coords_in->dt);
        const float local_density_output = *network_output;
        // density activation
        float density = network_to_density(local_density_output, density_activation);
        const float alpha = 1.f - __expf(-density * dt);
        T *= (1.f - alpha);

        // relocate pointer
        network_output += 1;
        coords_in += 1;
    }

    network_output -= compacted_numsteps; // rewind the pointer
    coords_in -= compacted_numsteps;

    uint32_t compacted_base = atomicAdd(numsteps_counter, compacted_numsteps); // first entry in the array is a counter
    compacted_numsteps = min(max_samples_compacted - min(max_samples_compacted, compacted_base), compacted_numsteps);
    numsteps_out[i * 2 + 0] = compacted_numsteps;
    numsteps_out[i * 2 + 1] = compacted_base;
    if (compacted_numsteps == 0)
    {
        return;
    }
    uint32_t rays_idx = atomicAdd(compacted_rays_counter, 1);
    coords_out += compacted_base;
    for (uint32_t j = 0; j < compacted_numsteps; ++j)
    {
        coords_out[j] = coords_in[j];
    }
}

void compacted_coord_api(const torch::Tensor &network_output,
    const torch::Tensor &coords_in,
    const torch::Tensor &rays_numsteps,
    const int &density_activation_i,
    const float &aabb0,
    const float &aabb1,
    torch::Tensor &coords_out,
    torch::Tensor &rays_numsteps_compacted,
    torch::Tensor &compacted_rays_counter,
    torch::Tensor &compacted_numstep_counter
    ){
    /*
     * @brief compacted_coord_api
     * @in-param   'network_output' (n_elements, 1)
     * @in-param   'coords_in' (n_elements,)
     * @in-param   'rays_numsteps'
     * @in-param   'bg_color_cpu'
     * @in-param   'rgb_activation_i'
     * @in-param   'density_activation_i'
     * @in-param   'aabb0'
     * @in-param   'aabb1'
     * @out-param  'coords_out'
     * @out-param  'rays_numsteps_compacted'
     * @out-param  'compacted_rays_counter'
     * @out-param  'compacted_numstep_counter'
     */

    cudaStream_t stream = 0;
    // input
    float* coords_in_p = (float*)coords_in.data_ptr();
    float* network_output_p = (float*)network_output.data_ptr();
    uint32_t* rays_numsteps_p = (uint32_t*)rays_numsteps.data_ptr();

    // output
    float* coords_out_p = (float*)coords_out.data_ptr();
    uint32_t* rays_numsteps_compacted_p = (uint32_t*)rays_numsteps_compacted.data_ptr();
    uint32_t* compacted_rays_counter_p = (uint32_t*)compacted_rays_counter.data_ptr();
    uint32_t* compacted_numstep_counter_p = (uint32_t*)compacted_numstep_counter.data_ptr();

    const unsigned int compacted_elements = coords_out.sizes()[0];
    const uint32_t n_rays = rays_numsteps.sizes()[0];
    BoundingBox m_aabb = BoundingBox(Eigen::Vector3f::Constant(aabb0), Eigen::Vector3f::Constant(aabb1));
    uint32_t padded_output_width = network_output.sizes()[1];

    ENerfActivation density_activation = ENerfActivation(density_activation_i);

    linear_kernel(compacted_coord_cuda<float>,0,stream,
        n_rays, m_aabb, compacted_elements,
        (float*)network_output_p,density_activation,
        (NerfCoordinate*)coords_in_p,(NerfCoordinate*)coords_out_p,
        (uint32_t*)rays_numsteps_p,(uint32_t*)compacted_numstep_counter_p,
        (uint32_t*)rays_numsteps_compacted_p,(uint32_t*)compacted_rays_counter_p);

    cudaDeviceSynchronize();
}
