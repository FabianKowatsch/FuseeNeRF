# Fusee-NeRF-Viewer

This Project aims to provide an executable application for rendering Neural Radiance Fields in realtime. This is achieved by reimplementing the NeRF functionalities of Nvidias [Instant-NGP](https://github.com/NVlabs/instant-ngp) and rendering the resulting images in realtime using the 3D-Library [FUSEE](https://fusee3d.org/). Since FUSEE uses the .NET Framework, an interface for working with matrices stored on the GPU and managed C# Code is needed. Therefore, this project makes use of [TorchSharp](https://github.com/dotnet/TorchSharp) and follows other reimplementations of Instant-NGP such as [xrnerf](https://github.com/openxrlab/xrnerf) and [torch-ngp](https://github.com/ashawkey/torch-ngp). [Tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)'s networks are used to further increase the performance.

## Usage

- Open the .sln file in MS Visual Studio
- specify the launch parameters in Config/config.json, add a path to the dataset
- Instant-NGPs dataset and the original NeRF synthetic datasets are supported, although most testing has been done on the synthetic lego dataset

```json
{
    "dataPath": "D:\\path\\to\\data folder\\containing\\transforms.json file",
    "trainDataFilename": "transforms_train",
    "evalDataFilename": "transforms_val",
    "datasetType": "synthetic",
    "aabbMin": 0.0,
    "aabbMax": 1.0,
    "aabbScale": 0.33,
    "offset": [ 0.5, 0.5, 0.5 ],
    "bgColor": [ 1.0, 1.0, 1.0 ],
    "useRandomBgColor":  true,
    "imageDownscale": 4.0,
    "nRays": 4096,
    "stepsToTrain":  3000,
    "gradScale": 1.0,
    "optimizerCfg": {
        "otype": "Ema",
        "decay": 0.95,
        "nested": {
            "otype": "ExponentialDecay",
            "decay_start": 20000,
            "decay_interval": 10000,
            "decay_base": 0.33,
            "nested": {
                "otype": "Adam",
                "learning_rate": 1e-2,
                "beta1": 0.9,
                "beta2": 0.99,
                "epsilon": 1e-15,
                "l2_reg": 1e-6
            }
        }
    },
    "posEncodingCfg": {
        "otype": "HashGrid",
        "n_levels": 8,
        "n_features_per_level": 4,
        "log2_hashmap_size": 19,
        "base_resolution": 16
    },
    "dirEncodingCfg": {
        "n_dims_to_encode": 3,
        "otype": "SphericalHarmonics",
        "degree": 4
    },
    "sigmaNetCfg": {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 1
    },
    "colorNetCfg": {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 2
    }
}
```
- rebuild the Solution or just the FuseeNeRF project for *Release x64* and launch \FuseeNeRF\Build\x64\Release\net7.0-windows\FuseeApp.exe

## Requirements

- Windows x64
- An Nvidia GPU, currently only GPUs with compute capability >= 70 are supported (tested on RTX 3070)
- A valid installation of [CUDA 11.7](https://developer.nvidia.com/cuda-downloads)
    - Add \NVIDIA GPU Computing Toolkit\CUDA\v11.7 to a CUDA_PATH environment variable
- Requires libtorch 2.0.1: download libtorch or pytorch with libtorch 2.0.1 backend [here](https://pytorch.org/get-started/locally/)
    - Add \libtorch directory to a new TORCH_DIR environment variable
- A valid installation of MS Visual Studio (tested with Visual Studio 2022 Community, other versions may work too)
- A valid installation of .NET 7 and the latest C++ Redistributables (can be installed together with MS Visual Studio)

## Installation

- Build Nvidias [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) project by following their [instructions](https://github.com/NVlabs/tiny-cuda-nn#compilation-windows--linux) and copy the resulting tiny-cuda-nn.lib to FuseeNeRF/TcnnApi/dependencies/Libraries
- Open the project in Visual Studio to install all C# dependencies
- Build the project and launch FuseeApp.exe

## About the project

### How it works

This application is designed to be a real-time 3D Viewer for Neural Radiance Fields in C#, extending the capabilities of the game engine [FUSEE](https://fusee3d.org/). The project reimplements some features of [Instant-NGP](https://github.com/NVlabs/instant-ngp) and makes them usable in the .NET framework. Rendering Neural Radiance Fields in real-time requires different optimaztion steps. The Hash grid encoding provided by instant-ngp, in combination with their optimized rendering algorithms and the optimized networks [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) allow for high performant NeRF rendering. The three optimizations are accessed with the help of CUDA APis, which can be called in .NET thanks to the interoperability between C# and C++/CUDA. As an interface for using the data in the context of C# and CUDA, [TorchSharp](https://github.com/dotnet/TorchSharp) is used. The training data is loaded in C#, and passed as rays via an API to a CUDA kernel to generate training samples and update a density grid and bitfield. These samples are passed to an API using tiny-cuda-nn in combination with instant-ngp's [NeRF implementation](https://github.com/NVlabs/instant-ngp/blob/master/include/neural-graphics-primitives/nerf_network.h) to query a neural network with a highly optimized input encoding. The generated results can be used in volume rendering kernels to calculate an image, or comapare the data to ground truth values to calculate a loss and gradients. The gradients are propagated backward and finally passed to the tiny-cuda-nn API to be used by their Adam implementation to update the networks parameters.

### Structure

```
FuseeNeRF
├── FuseeApp
│   ├── FuseeNeRF
│   ├── Program
├── InstantNeRF
│   ├── Autograd & Context
│   ├── Config
│   ├── DataProvider
│   ├── GridSampler
│   ├── MLP & Tcnn Nerf Apis
│   ├── Network
│   ├── Optimizer
│   ├── VolumeRenderer
│   ├── Trainer
│   ├── Utils
├── RaymarchApi
│   ├── CUDA kernels & headers
│   ├── API
├── TcnnNerfApi
│   ├── Module
│   ├── NerfNetwork
│   ├── Optimizer
│   ├── API
...
├── Build
│   ├── x64\Release\net7.0-windows\FuseeApp.exe
├── Config
│   ├── config.json

```

## Work in Progress

### Current State

The viewer is able to render images at 200x200 pixels up to 800x800 pixels at ~7 frames pare second during live training and images witha  resolution of 100x100 at ~60 frames per second during inference after the model stopped training. This seems related to a bug caused by the camera pose, ray generation/marching, the density bitfield or a combination of these issues. At resolutions > 100x100, more and more rays seem to be skipped, by either not hitting the bounding box or by the bitfield being calculated wrongly due to a wrong camera pose transformation. Further testing has to be done. For reference, consider reading the following files:

- matrixToNGP() in Utils.cs & [nerf_loader.cu](https://github.com/NVlabs/instant-ngp/blob/master/src/nerf_loader.cu), [nerf_loader.h](https://github.com/NVlabs/instant-ngp/blob/master/include/neural-graphics-primitives/nerf_loader.h), [matrix_nerf2ngp](https://github.com/openxrlab/xrnerf/blob/main/xrnerf/datasets/utils/hashnerf.py) for pose transformation
- getRaysFromPose() in Utils.cs & [get_rays.py](https://github.com/openxrlab/xrnerf/blob/main/xrnerf/datasets/load_data/get_rays.py#L35) for ray generation

This issue also causes the controls of the viewer to work incorrectly, so a default pose is set.


### Future Goals

- Fixing the issues related to the pose
- User Interface

## License

This project uses code from various codebases and is limited by their restrictions. The licenses can found in the corresponding subdirectories

## References

Credits and thanks to:

- [Instant-NGP](https://github.com/NVlabs/instant-ngp)
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- [xrnerf](https://github.com/openxrlab/xrnerf)
- [torch-ngp](https://github.com/ashawkey/torch-ngp)
- [TorchSharp](https://github.com/dotnet/TorchSharp)
- [FUSEE](https://fusee3d.org/)

