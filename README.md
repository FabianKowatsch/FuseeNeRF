# Fusee-NeRF-Viewer

This Project aims to provide an executable application for rendering Neural Radiance Fields in realtime. This is achieved by reimplementing the NeRF functionalities of Nvidias [Instant-NGP](https://github.com/NVlabs/instant-ngp) and rendering the resulting images in realtime using the 3D-Library [FUSEE](https://fusee3d.org/). Since FUSEE uses the .NET Framework, an interface for working with matrices stored on the GPU and managed C# Code is needed. Therefore, this project makes use of [TorchSharp](https://github.com/dotnet/TorchSharp) and follows other reimplementations of Instant-NGP such as [xrnerf](https://github.com/openxrlab/xrnerf) and [torch-ngp](https://github.com/ashawkey/torch-ngp).

## Usage

- Open the .sln file in MS Visual Studio
- specify the launch parameters in Config/config.json
- Instant-NGPs dataset and the original NeRFs datasets are supported, although most testing has been done on the synthetic lego dataset

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
    "imageDownscale": 2.0,
    "nRays": 2048,
    "gradScale": 128.0,
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
- A valid installation of MS Visual Studio (tested with VS 2022 Community, other versions may work too)
- A valid installation of .NET 7 and the latest C++ Redistributables (can be installed together with MS VS)

## Installation

- Build Nvidias [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) project by following their [instructions](https://github.com/NVlabs/tiny-cuda-nn#compilation-windows--linux) and copying tiny-cuda-nn.lib to FuseeNeRF/TcnnApi/dependencies/Libraries
- Open the project in Visual Studio to install all C# dependencies
- Build the project and launch FuseeApp.exe

## About the project

Currently, two different approaches are implemented, with one being the approach of torch-ngp and handling more work directly in C# (main branch). The other approach is closer to xrnerf & Instant-NGP and handles most of the important calculations in native CUDA-Code (dev-branch).

### How it works


## Work in Progress

### Current State

Currently no results can be seen:
- An error causes the parameters of the network to reach the numerical limits and turn to NaN after the first optimization step.
- This result is achieved by uising the following parameters on the main branch:
    - Any offset in the range of [0, 1]
    - Any background color in the range of [0, 1]
    - Gradient Scaler: on and off
    - Dataset: lego and fox
    - Mixed Precision training (float16 & float32): on and off
- On the dev branch, all density related Tensors become invalid when using a DisposeScope for training to avoid memory leaks
    - ...although they are created outside of the scope and not reassigned, their values are only adjusted by accessing the underlying storage in a CUDA Kernel

### Future Goals

- Configuration(done)
- Fixing NaN Error(done)
- Fixing Rendering
- User Interface

## References

Credits to:
- [Instant-NGP](https://github.com/NVlabs/instant-ngp)
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- [xrnerf](https://github.com/openxrlab/xrnerf)
- [torch-ngp](https://github.com/ashawkey/torch-ngp)