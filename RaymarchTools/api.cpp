#define EXPORT_API extern "C" __declspec(dllexport)
#include "raymarch.h"

EXPORT_API void nearFarFromAabbApi(const torch::Tensor* rays_o, const torch::Tensor* rays_d, const torch::Tensor* aabb, const unsigned int N, const float min_near, torch::Tensor* nears, torch::Tensor* fars) {

	near_far_from_aabb(*rays_o, *rays_d, *aabb, N, min_near, *nears, *fars);
}

EXPORT_API void sphereFomRayApi(const torch::Tensor* rays_o, const torch::Tensor* rays_d, const torch::Tensor* aabb, const unsigned int N, const float radius, torch::Tensor* coords) {

	sph_from_ray(*rays_o, *rays_d, radius, N, *coords);
}

EXPORT_API void morton3DApi(const torch::Tensor* coords, const unsigned int N, torch::Tensor* indices) {
	morton3D(*coords, N, *indices);
}

EXPORT_API void morton3DInvertApi(const torch::Tensor* indices, const unsigned int N, torch::Tensor* coords) {
	morton3D_invert(*indices, N, *coords);
}

EXPORT_API void packbitsApi(const torch::Tensor* grid, const unsigned int N, const float densityThreshhold, torch::Tensor* bitfield) {
	packbits(*grid, N, densityThreshhold, *bitfield);
}

EXPORT_API void raymarchTrainApi(const torch::Tensor* rays_o,
	const torch::Tensor* rays_d,
	const torch::Tensor* grid,
	const float bound,
	const float dtGamma,
	const unsigned int maxSteps,
	const unsigned int N,
	const unsigned int C,
	const unsigned int H,
	const unsigned int M,
	const torch::Tensor* nears,
	const torch::Tensor* fars,
	torch::Tensor* xyzs,
	torch::Tensor* dirs,
	torch::Tensor* deltas,
	torch::Tensor* rays,
	torch::Tensor* counter,
	torch::Tensor* noises
) {
	march_rays_train(*rays_o, *rays_d, *grid, bound, dtGamma, maxSteps, N, C, H, M, *nears, *fars, *xyzs, *dirs, *deltas, *rays, *counter, *noises);
}

EXPORT_API void compositeRaysTrainForwardApi(const torch::Tensor* sigmas,
	const torch::Tensor* rgbs,
	const torch::Tensor* deltas,
	const torch::Tensor* rays,
	const unsigned int M,
	const unsigned int N,
	const float tThresh,
	torch::Tensor* weightsSum,
	torch::Tensor* depth,
	torch::Tensor* image) {
	composite_rays_train_forward(*sigmas, *rgbs, *deltas, *rays, M, N, tThresh, *weightsSum, *depth, *image);
}

EXPORT_API void compositeRaysTrainBackwardApi(const torch::Tensor* gradWeightsSum,
	const torch::Tensor* gradImage,
	const torch::Tensor* sigmas,
	const torch::Tensor* rgbs,
	const torch::Tensor* deltas,
	const torch::Tensor* rays,
	const torch::Tensor* weightsSum,
	const torch::Tensor* image,
	const unsigned int M,
	const unsigned int N,
	const float tThresh,
	torch::Tensor* gradSigmas,
	torch::Tensor* gradRgbs) {
	composite_rays_train_backward(*gradWeightsSum, *gradImage, *sigmas, *rgbs, *deltas, *rays, *weightsSum, *image, M, N, tThresh, *gradSigmas, *gradRgbs);
}

EXPORT_API void raymarchApi(const unsigned int nAlive,
	const unsigned int nStep,
	const torch::Tensor* raysAlive,
	const torch::Tensor* raysT,
	const torch::Tensor* rays_o,
	const torch::Tensor* rays_d,
	const float bound,
	const float dtGamma,
	const unsigned int maxSteps,
	const unsigned int C,
	const unsigned int H,
	const torch::Tensor* grid,
	const torch::Tensor* nears,
	const torch::Tensor* fars,
	torch::Tensor* xyzs,
	torch::Tensor* dirs,
	torch::Tensor* deltas,
	torch::Tensor* noises
) {
	march_rays(nAlive, nStep, *raysAlive, *raysT, *rays_o, *rays_d, bound, dtGamma, maxSteps, C, H, *grid, *nears, *fars, *xyzs, *dirs, *deltas, *noises);
}

EXPORT_API void compositeRaysApi(const unsigned int nAlive,
	const unsigned int nStep,
	const float tThresh,
	torch::Tensor* raysAlive,
	torch::Tensor* raysT,
	torch::Tensor* sigmas,
	torch::Tensor* rgbs,
	torch::Tensor* deltas,
	torch::Tensor* weightsSum,
	torch::Tensor* depth,
	torch::Tensor* image
) {
	composite_rays(nAlive, nStep, tThresh, *raysAlive, *raysT, *sigmas, *rgbs, *deltas, *weightsSum, *depth, *image);
}