/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#pragma once
#include "raymarch_shared.h"

using namespace Eigen;


#ifdef __NVCC__
#define NGP_HOST_DEVICE __host__ __device__
#else
#define NGP_HOST_DEVICE
#endif

#define TCNN_HOST_DEVICE __host__ __device__
#define TCNN_MIN_GPU_ARCH 70

static constexpr float UNIFORM_SAMPLING_FRACTION = 0.5f;
// constexpr uint32_t n_threads_linear = 128;

inline __device__ float clamp_(float val, float lower, float upper){return val < lower ? lower : (upper < val ? upper : val);}
inline __device__ float calc_dt(float t, float cone_angle){ return clamp_(t * cone_angle, MIN_CONE_STEPSIZE(), MAX_CONE_STEPSIZE());}

// inline __device__ float calc_dt(float t, float cone_angle){return MIN_CONE_STEPSIZE() * 0.5;}

enum class EColorSpace : int
{
    Linear,
    SRGB,
    VisPosNeg,
};


inline __device__ int mip_from_pos(const Vector3f &pos)
{
    int exponent;
    float maxval = (pos - Vector3f::Constant(0.5f)).cwiseAbs().maxCoeff();
    frexpf(maxval, &exponent);
    return min(NERF_CASCADES() - 1, max(0, exponent + 1));
}

inline __device__ int mip_from_dt(float dt, const Vector3f &pos)
{
    int mip = mip_from_pos(pos);
    dt *= 2 * NERF_GRIDSIZE();
    if (dt < 1.f)
        return mip;
    int exponent;
    frexpf(dt, &exponent);
    return min(NERF_CASCADES() - 1, max(exponent, mip));
}


// other needed structure
struct CameraDistortion
{
    float params[4] = {};
#ifdef __NVCC__
    inline __host__ __device__ bool is_zero() const
    {
        return params[0] == 0.0f && params[1] == 0.0f && params[2] == 0.0f && params[3] == 0.0f;
    }
#endif
};


struct NerfDirection
{
    NGP_HOST_DEVICE NerfDirection(const Eigen::Vector3f &dir, float dt) : d{dir} {}
    Eigen::Vector3f d;
};

struct NerfCoordinate
{
    NGP_HOST_DEVICE NerfCoordinate(const Eigen::Vector3f &pos, const Eigen::Vector3f &dir, float dt) : pos{pos, dt}, dt{dt}, dir{dir, dt} {}
    NGP_HOST_DEVICE void set_with_optional_light_dir(const Eigen::Vector3f &pos, const Eigen::Vector3f &dir, float dt, const Eigen::Vector3f &light_dir, uint32_t stride_in_bytes)
    {
        this->dt = dt;
        this->pos = NerfPosition(pos, dt);
        this->dir = NerfDirection(dir, dt);

        if (stride_in_bytes >= sizeof(Eigen::Vector3f) + sizeof(NerfCoordinate))
        {
            *(Eigen::Vector3f *)(this + 1) = light_dir;
        }
    }
    NGP_HOST_DEVICE void copy_with_optional_light_dir(const NerfCoordinate &inp, uint32_t stride_in_bytes)
    {
        *this = inp;
        if (stride_in_bytes >= sizeof(Eigen::Vector3f) + sizeof(NerfCoordinate))
        {
            *(Eigen::Vector3f *)(this + 1) = *(Eigen::Vector3f *)(&inp + 1);
        }
    }

    NerfPosition pos;
    float dt;
    NerfDirection dir;
};


template <typename T>
struct PitchedPtr
{
    TCNN_HOST_DEVICE PitchedPtr() : ptr{nullptr}, stride_in_bytes{sizeof(T)} {}
    TCNN_HOST_DEVICE PitchedPtr(T *ptr, size_t stride_in_elements, size_t offset = 0, size_t extra_stride_bytes = 0) : ptr{ptr + offset}, stride_in_bytes{(uint32_t)(stride_in_elements * sizeof(T) + extra_stride_bytes)} {}

    template <typename U>
    TCNN_HOST_DEVICE explicit PitchedPtr(PitchedPtr<U> other) : ptr{(T *)other.ptr}, stride_in_bytes{other.stride_in_bytes} {}

    TCNN_HOST_DEVICE T *operator()(uint32_t y) const
    {
        return (T *)((const char *)ptr + y * stride_in_bytes);
    }

    TCNN_HOST_DEVICE void operator+=(uint32_t y)
    {
        ptr = (T *)((const char *)ptr + y * stride_in_bytes);
    }

    TCNN_HOST_DEVICE void operator-=(uint32_t y)
    {
        ptr = (T *)((const char *)ptr - y * stride_in_bytes);
    }

    TCNN_HOST_DEVICE explicit operator bool() const
    {
        return ptr;
    }

    T *ptr;
    uint32_t stride_in_bytes;
};


using default_rng_t = pcg32;

template <typename T, uint32_t N_ELEMS>
struct vector_t
{
    TCNN_HOST_DEVICE T &operator[](uint32_t idx)
    {
        return data[idx];
    }

    TCNN_HOST_DEVICE T operator[](uint32_t idx) const
    {
        return data[idx];
    }

    T data[N_ELEMS];
    static constexpr uint32_t N = N_ELEMS;
};


constexpr uint32_t batch_size_granularity = 128;

static constexpr uint32_t MARCH_ITER = 10000;

static constexpr uint32_t MIN_STEPS_INBETWEEN_COMPACTION = 1;
static constexpr uint32_t MAX_STEPS_INBETWEEN_COMPACTION = 8;

inline __host__ __device__ float calc_cone_angle(float cosine, const Eigen::Vector2f &focal_length, float cone_angle_constant)
{
    // Pixel size. Doesn't always yield a good performance vs. quality
    // trade off. Especially if training pixels have a much different
    // size than rendering pixels.
    // return cosine*cosine / focal_length.mean();

    return cone_angle_constant;
}


inline __device__ float distance_to_next_voxel(const Vector3f &pos, const Vector3f &dir, const Vector3f &idir, uint32_t res)
{ // dda like step
    Vector3f p = res * pos;
    float tx = (floorf(p.x() + 0.5f + 0.5f * sign(dir.x())) - p.x()) * idir.x();
    float ty = (floorf(p.y() + 0.5f + 0.5f * sign(dir.y())) - p.y()) * idir.y();
    float tz = (floorf(p.z() + 0.5f + 0.5f * sign(dir.z())) - p.z()) * idir.z();
    float t = min(min(tx, ty), tz);

    return fmaxf(t / res, 0.0f);
}

inline __device__ float advance_to_next_voxel(float t, float cone_angle, const Vector3f &pos, const Vector3f &dir, const Vector3f &idir, uint32_t res)
{
    // Analytic stepping by a multiple of dt. Make empty space unequal to non-empty space
    // due to the different stepping.
    // float dt = calc_dt(t, cone_angle);
    // return t + ceilf(fmaxf(distance_to_next_voxel(pos, dir, idir, res) / dt, 0.5f)) * dt;

    // Regular stepping (may be slower but matches non-empty space)
    float t_target = t + distance_to_next_voxel(pos, dir, idir, res);
    do
    {
        t += calc_dt(t, cone_angle);
    } while (t < t_target);
    return t;
}

inline __device__ uint32_t cascaded_grid_idx_at(Vector3f pos, uint32_t mip)
{
    float mip_scale = scalbnf(1.0f, -mip);
    pos -= Vector3f::Constant(0.5f);
    pos *= mip_scale;
    pos += Vector3f::Constant(0.5f);

    Vector3i i = (pos * NERF_GRIDSIZE()).cast<int>();

    uint32_t idx = morton3D(
        clamp(i.x(), 0, (int)NERF_GRIDSIZE() - 1),
        clamp(i.y(), 0, (int)NERF_GRIDSIZE() - 1),
        clamp(i.z(), 0, (int)NERF_GRIDSIZE() - 1));

    return idx;
}

inline __device__ bool density_grid_occupied_at(const Vector3f &pos, const uint8_t *density_grid_bitfield, uint32_t mip)
{
    uint32_t idx = cascaded_grid_idx_at(pos, mip);
    return density_grid_bitfield[idx / 8 + grid_mip_offset(mip) / 8] & (1 << (idx % 8));
}

inline __device__ float cascaded_grid_at(Vector3f pos, const float *cascaded_grid, uint32_t mip)
{
    uint32_t idx = cascaded_grid_idx_at(pos, mip);
    return cascaded_grid[idx + grid_mip_offset(mip)];
}

inline __device__ float &cascaded_grid_at(Vector3f pos, float *cascaded_grid, uint32_t mip)
{
    uint32_t idx = cascaded_grid_idx_at(pos, mip);
    return cascaded_grid[idx + grid_mip_offset(mip)];
}



inline __device__ Vector3f unwarp_position(const Vector3f &pos, const BoundingBox &aabb)
{
    // return {logit(pos.x()) + 0.5f, logit(pos.y()) + 0.5f, logit(pos.z()) + 0.5f};
    // return pos;

    return aabb.min + pos.cwiseProduct(aabb.diag());
}

inline __device__ Vector3f unwarp_position_derivative(const Vector3f &pos, const BoundingBox &aabb)
{
    // return {logit(pos.x()) + 0.5f, logit(pos.y()) + 0.5f, logit(pos.z()) + 0.5f};
    // return pos;

    return aabb.diag();
}

inline __device__ Vector3f warp_position_derivative(const Vector3f &pos, const BoundingBox &aabb)
{
    return unwarp_position_derivative(pos, aabb).cwiseInverse();
}

inline __device__ Vector3f warp_direction(const Vector3f &dir)
{
    return (dir + Vector3f::Ones()) * 0.5f;
}

inline __device__ Vector3f unwarp_direction(const Vector3f &dir)
{
    return dir * 2.0f - Vector3f::Ones();
}

inline __device__ Vector3f warp_direction_derivative(const Vector3f &dir)
{
    return Vector3f::Constant(0.5f);
}

inline __device__ Vector3f unwarp_direction_derivative(const Vector3f &dir)
{
    return Vector3f::Constant(2.0f);
}


inline __device__ float unwarp_dt(float dt)
{
    float max_stepsize = MIN_CONE_STEPSIZE() * (1 << (NERF_CASCADES() - 1));
    return dt * (max_stepsize - MIN_CONE_STEPSIZE()) + MIN_CONE_STEPSIZE();
}


__device__ inline float random_val(uint32_t seed, uint32_t idx)
{
    pcg32 rng(((uint64_t)seed << 32) | (uint64_t)idx);
    return rng.next_float();
}



template <typename RNG>
inline __host__ __device__ Eigen::Vector2f random_val_2d(RNG &rng)
{
    return {rng.next_float(), rng.next_float()};
}


inline __device__ float network_to_rgb(float val, ENerfActivation activation)
{
    switch (activation)
    {
    case ENerfActivation::None:
        return val;
    case ENerfActivation::ReLU:
        return val > 0.0f ? val : 0.0f;
    case ENerfActivation::Logistic:
        return logistic(val);
    case ENerfActivation::Exponential:
        return __expf(clamp(val, -10.0f, 10.0f));
    default:
        assert(false);
    }
    return 0.0f;
}
template <typename T>
inline __device__ Array3f network_to_rgb(const vector_t<T, 4> &local_network_output, ENerfActivation activation)
{
    return {
        network_to_rgb(float(local_network_output[0]), activation),
        network_to_rgb(float(local_network_output[1]), activation),
        network_to_rgb(float(local_network_output[2]), activation)};
}



inline __host__ __device__ float linear_to_srgb(float linear)
{
    if (linear < 0.0031308f)
    {
        return 12.92f * linear;
    }
    else
    {
        return 1.055f * std::pow(linear, 0.41666f) - 0.055f;
    }
}

inline __host__ __device__ Eigen::Array3f linear_to_srgb(const Eigen::Array3f &x)
{
    return {linear_to_srgb(x.x()), linear_to_srgb(x.y()), (linear_to_srgb(x.z()))};
}

inline __host__ __device__ float srgb_to_linear(float srgb)
{
    if (srgb <= 0.04045f)
    {
        return srgb / 12.92f;
    }
    else
    {
        return std::pow((srgb + 0.055f) / 1.055f, 2.4f);
    }
}

inline __host__ __device__ Eigen::Array3f srgb_to_linear(const Eigen::Array3f &x)
{
    return {srgb_to_linear(x.x()), srgb_to_linear(x.y()), (srgb_to_linear(x.z()))};
}
struct LossAndGradient
{
    Eigen::Array3f loss;
    Eigen::Array3f gradient;

    __host__ __device__ LossAndGradient operator*(float scalar)
    {
        return {loss * scalar, gradient * scalar};
    }

    __host__ __device__ LossAndGradient operator/(float scalar)
    {
        return {loss / scalar, gradient / scalar};
    }
};

inline __host__ __device__ LossAndGradient huber_loss(const Array3f& target, const Array3f& prediction, float alpha = 1) {
    Array3f difference = prediction - target;
    Array3f abs_diff = difference.abs();
    Array3f square = 0.5f / alpha * difference * difference;
    return {
        {
            abs_diff.x() > alpha ? (abs_diff.x() - 0.5f * alpha) : square.x(),
            abs_diff.y() > alpha ? (abs_diff.y() - 0.5f * alpha) : square.y(),
            abs_diff.z() > alpha ? (abs_diff.z() - 0.5f * alpha) : square.z(),
        },
        {
            abs_diff.x() > alpha ? (difference.x() > 0 ? 1.0f : -1.0f) : (difference.x() / alpha),
            abs_diff.y() > alpha ? (difference.y() > 0 ? 1.0f : -1.0f) : (difference.y() / alpha),
            abs_diff.z() > alpha ? (difference.z() > 0 ? 1.0f : -1.0f) : (difference.z() / alpha),
        },
    };
}

inline __device__ float network_to_rgb_derivative(float val, ENerfActivation activation)
{
    switch (activation)
    {
    case ENerfActivation::None:
        return 1.0f;
    case ENerfActivation::ReLU:
        return val > 0.0f ? 1.0f : 0.0f;
    case ENerfActivation::Logistic:
    {
        float density = logistic(val);
        return density * (1 - density);
    };
    case ENerfActivation::Exponential:
        return __expf(clamp(val, -10.0f, 10.0f));
    default:
        assert(false);
    }
    return 0.0f;
}

inline __device__ float network_to_density_derivative(float val, ENerfActivation activation)
{
    switch (activation)
    {
    case ENerfActivation::None:
        return 1.0f;
    case ENerfActivation::ReLU:
        return val > 0.0f ? 1.0f : 0.0f;
    case ENerfActivation::Logistic:
    {
        float density = logistic(val);
        return density * (1 - density);
    };
    case ENerfActivation::Exponential:
        return __expf(clamp(val, -15.0f, 15.0f));
    default:
        assert(false);
    }
    return 0.0f;
}
