#define EXPORT_API extern "C" __declspec(dllexport)

#include "common.h"

EXPORT_API void generateGridSamplesApi(const torch::Tensor* density_grid,
    int density_grid_ema_step, int n_elements, int max_cascade,
    float thresh, float aabb0, float aabb1,
    torch::Tensor* density_grid_positions_uniform,
    torch::Tensor* density_grid_indices_uniform) {

    generate_grid_samples_nerf_nonuniform_api(*density_grid, density_grid_ema_step, n_elements, max_cascade, thresh, aabb0, aabb1, *density_grid_positions_uniform, *density_grid_indices_uniform);
}

EXPORT_API void markUntrainedGridApi(const torch::Tensor* focal_lengths,
    const torch::Tensor* transforms,
    int n_elements, int n_images,
    int img_resolution0, int img_resolution1,
    torch::Tensor* density_grid) {

    mark_untrained_density_grid_api(*focal_lengths, *transforms, n_elements, n_images, img_resolution0, img_resolution1, *density_grid);
}

EXPORT_API void splatGridSamplesNerfMaxNearestNeighbourApi(const torch::Tensor* mlp_out,
    const torch::Tensor* density_grid_indices, int padded_output_width,
    int n_density_grid_samples, torch::Tensor* density_grid_tmp) {

    splat_grid_samples_nerf_max_nearest_neighbor_api(*mlp_out, *density_grid_indices, padded_output_width, n_density_grid_samples, *density_grid_tmp);
}

EXPORT_API void emaGridSamplesNerfApi(const torch::Tensor* density_grid_tmp, int n_elements,
    float decay, torch::Tensor* density_grid) {

    ema_grid_samples_nerf_api(*density_grid_tmp, n_elements, decay, *density_grid);
}

EXPORT_API void updateBitfieldApi(const torch::Tensor* density_grid,
    torch::Tensor* density_grid_mean, torch::Tensor* density_grid_bitfield) {

    update_bitfield_api(*density_grid, *density_grid_mean, *density_grid_bitfield);
}

EXPORT_API void sampleRaysApi(
    const torch::Tensor* rays_o,
    const torch::Tensor* rays_d,
    const torch::Tensor* density_grid_bitfield,
    const torch::Tensor* metadata,
    const torch::Tensor* imgs_id,
    const torch::Tensor* xforms,
    float aabb0,
    float aabb1,
    float near_distance,
    float cone_angle_constant,
    torch::Tensor* coords_out,
    torch::Tensor* rays_index,
    torch::Tensor* rays_numsteps,
    torch::Tensor* ray_numstep_counter) {

    rays_sampler_api(*rays_o, *rays_d, *density_grid_bitfield, *metadata, *imgs_id, *xforms, aabb0, aabb1, near_distance, cone_angle_constant, *coords_out, *rays_index, *rays_numsteps, *ray_numstep_counter);

}

EXPORT_API void compactedCoordsApi(
    const torch::Tensor* network_output,
    const torch::Tensor* coords_in,
    const torch::Tensor* rays_numsteps,
    const torch::Tensor* bg_color_in,
    int rgb_activation_i,
    int density_activation_i,
    float aabb0,
    float aabb1,
    torch::Tensor* coords_out,
    torch::Tensor* rays_numsteps_compacted,
    torch::Tensor* compacted_rays_counter,
    torch::Tensor* compacted_numstep_counter) {

    compacted_coord_api(*network_output, *coords_in, *rays_numsteps, *bg_color_in, rgb_activation_i, density_activation_i, aabb0, aabb1, *coords_out, *rays_numsteps_compacted, *compacted_rays_counter, *compacted_numstep_counter);
}

EXPORT_API void calculateRGBsForwardApi(
    const torch::Tensor* network_output,
    const torch::Tensor* coords_in,
    const torch::Tensor* rays_numsteps,
    const torch::Tensor* rays_numsteps_compacted,
    const torch::Tensor* training_background_color,
    int rgb_activation_i,
    int density_activation_i,
    float aabb0,
    float aabb1,
    torch::Tensor* rgb_output) {

    calc_rgb_forward_api(*network_output, *coords_in, *rays_numsteps, *rays_numsteps_compacted, *training_background_color, rgb_activation_i, density_activation_i, aabb0, aabb1, *rgb_output);
}

EXPORT_API void calculateRGBsBackwardApi(
    const torch::Tensor* network_output,
    const torch::Tensor* rays_numsteps_compacted,
    const torch::Tensor* coords_in,
    const torch::Tensor* grad_x,
    const torch::Tensor* rgb_output,
    const torch::Tensor* density_grid_mean,

    int rgb_activation_i,
    int density_activation_i,
    float aabb0,
    float aabb1,
    torch::Tensor* dloss_doutput) {

    calc_rgb_backward_api(*network_output, *rays_numsteps_compacted, *coords_in, *grad_x, *rgb_output, *density_grid_mean, rgb_activation_i, density_activation_i, aabb0, aabb1, *dloss_doutput);
}

EXPORT_API void calculateRGBsInferenceApi(
    const torch::Tensor* network_output,
    const torch::Tensor* coords_in,
    const torch::Tensor* rays_numsteps,
    const torch::Tensor* bg_color_cpu,
    int rgb_activation_i,
    int density_activation_i,
    float aabb0,
    float aabb1,
    torch::Tensor* rgb_output,
    torch::Tensor* alpha_output) {

    calc_rgb_inference_api(*network_output, *coords_in, *rays_numsteps, *bg_color_cpu, rgb_activation_i, density_activation_i, aabb0, aabb1, *rgb_output, *alpha_output);
}