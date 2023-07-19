#include "raymarch_shared.h"
#include "ray_sampler_header.h"
#include "common.h"
#include "device_atomic_functions.h"
__global__ void rays_sampler_cuda(
    const uint32_t n_rays,
    BoundingBox aabb,
    const uint32_t max_samples,
    const Vector3f* __restrict__ rays_o,
    const Vector3f* __restrict__ rays_d,
    const uint8_t* __restrict__ density_grid,
    const float cone_angle_constant,
    const float* __restrict__ metadata,
    const uint32_t* __restrict__ imgs_index,
    uint32_t* __restrict__ ray_counter,
    uint32_t* __restrict__ numsteps_counter,
    uint32_t* __restrict__ ray_indices_out,
    uint32_t* __restrict__ numsteps_out,
    PitchedPtr<NerfCoordinate> coords_out,
    const Matrix<float, 3, 4>* training_xforms,
    float near_distance,
    default_rng_t rng
)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    
    // i (0,n_rays)
    // assert(i == n_rays); or 
    // assert(i == imgs_index.size()); // UH?
    if (i >= n_rays)
        return;
    uint32_t img = imgs_index[i];
    rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());

    const Matrix<float, 3, 4> xform = training_xforms[img];

    const Vector2f focal_length(metadata[6], metadata[7]);
    const Vector3f light_dir(metadata[8], metadata[9], metadata[10]);

    const Vector3f light_dir_warped = warp_direction(light_dir);

    Vector3f ray_o = rays_o[i];
    Vector3f ray_d = rays_d[i];

    Vector2f tminmax = aabb.ray_intersect(ray_o, ray_d);

    float cone_angle = cone_angle_constant;

    tminmax.x() = fmaxf(tminmax.x(), near_distance);

    float startt = tminmax.x();

    startt += calc_dt(startt, cone_angle) * random_val(rng);

    Vector3f idir = ray_d.cwiseInverse();

    // first pass to compute an accurate number of steps

    uint32_t j = 0;
    float t = startt;
    Vector3f pos;
 
    while (aabb.contains(pos = ray_o + t * ray_d) && j < NERF_STEPS())
    {

        float dt = calc_dt(t, cone_angle);
        uint32_t mip = mip_from_dt(dt, pos);
        if (density_grid_occupied_at(pos, density_grid, mip))
        {
            ++j;
            t += dt;
        }
        else
        {
            uint32_t res = NERF_GRIDSIZE() >> mip;
            t = advance_to_next_voxel(t, cone_angle, pos, ray_d, idir, res);
        }
    }


    uint32_t numsteps = j;
    // assert(numsteps > 0); UH?
    uint32_t base = atomicAdd(numsteps_counter, numsteps);


    if (base + numsteps > max_samples)
    {
        // UH: log if we are skipping rays 
        numsteps_out[2 * i + 0] = 0;
        numsteps_out[2 * i + 1] = base;
        return;
    }

    coords_out += base;

    uint32_t ray_idx = atomicAdd(ray_counter, 1);
    ray_indices_out[i] = ray_idx;
    // TODO:
    numsteps_out[2 * i + 0] = numsteps;
    numsteps_out[2 * i + 1] = base;


    if (j == 0)
    {
        //printf("early exit \n");
        ray_indices_out[i] = -1;
        return;
    }
    //printf("continued!\n");

 
    Vector3f warped_dir = warp_direction(ray_d);
    t = startt;
    j = 0;

    if (ray_idx % 100 == 0) {
        //printf("warped dir: %f, %f, %f | startt: %f\n", warped_dir.x(), warped_dir.y(), warped_dir.z(), t);
        //printf("warped pos: %f | warped dir: %f | warped pos: %f | warped dt: %f | warped lightdir: %f | coords stride: %i \n", warp_position(pos, aabb).x(), warped_dir.x(), warp_dt(t), light_dir_warped.x(), coords_out.stride_in_bytes);
    }


    while (aabb.contains(pos = ray_o + t * ray_d) && j < numsteps)
    {
        float dt = calc_dt(t, cone_angle);
        uint32_t mip = mip_from_dt(dt, pos);
        if (density_grid_occupied_at(pos, density_grid, mip))
        {
           
            coords_out(j)->set_with_optional_light_dir(warp_position(pos, aabb), warped_dir, warp_dt(dt), light_dir_warped, coords_out.stride_in_bytes);
            ++j;
            t += dt;
        }
        else
        {
            uint32_t res = NERF_GRIDSIZE() >> mip;
            t = advance_to_next_voxel(t, cone_angle, pos, ray_d, idir, res);
        }
    }
    /*
    */
}

void rays_sampler_api(
    const at::Tensor rays_o,
    const at::Tensor rays_d,
    const at::Tensor density_grid_bitfield,
    const at::Tensor metadata,
    const at::Tensor imgs_id,
    const at::Tensor xforms,
    const float aabb0,
    const float aabb1,
    const float near_distance,
    const float cone_angle_constant,
    at::Tensor coords_out,
    at::Tensor rays_index,
    at::Tensor rays_numsteps,
    at::Tensor ray_numstep_counter
) {

    cudaStream_t stream = 0;

    // input
    float* rays_o_pointer = (float*)rays_o.data_ptr();
    float* rays_d_pointer = (float*)rays_d.data_ptr();
    Vector3f* rays_o_p = (Vector3f*)rays_o_pointer;
    Vector3f* rays_d_p = (Vector3f*)rays_d_pointer;
    Eigen::Matrix<float, 3, 4>* transforms_p = (Eigen::Matrix<float, 3, 4>*)xforms.data_ptr();
    float* metadata_p = (float*)metadata.data_ptr();

    uint8_t* density_grid_bitfield_p = (uint8_t*)density_grid_bitfield.data_ptr();
    uint32_t* imgs_id_p = (uint32_t*)imgs_id.data_ptr();

    // output

    uint32_t* rays_index_p = (uint32_t*)rays_index.data_ptr();
    uint32_t* rays_numsteps_p = (uint32_t*)rays_numsteps.data_ptr();
    uint32_t* ray_numstep_counter_p = (uint32_t*)ray_numstep_counter.data_ptr();
    NerfCoordinate* coords_out_p = (NerfCoordinate*)coords_out.data_ptr();


    const unsigned int num_elements = coords_out.sizes()[0];
    const uint32_t n_rays = rays_o.sizes()[0];
    BoundingBox m_aabb = BoundingBox(Eigen::Vector3f::Constant(aabb0), Eigen::Vector3f::Constant(aabb1));

    linear_kernel(rays_sampler_cuda, 0, stream,
        n_rays, m_aabb, num_elements, (Vector3f*)rays_o_p, (Vector3f*)rays_d_p,
        (uint8_t*)density_grid_bitfield_p, cone_angle_constant,
        metadata_p, (uint32_t*)imgs_id_p, (uint32_t*)ray_numstep_counter_p,
        ((uint32_t*)ray_numstep_counter_p) + 1,
        (uint32_t*)rays_index_p,
        (uint32_t*)rays_numsteps_p,
        PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_out_p, 1, 0, 0),
        transforms_p, near_distance, rng);

    rng.advance();
    cudaDeviceSynchronize();

}
