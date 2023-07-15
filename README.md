# Fusee-NeRF-Viewer

This Project aims to provide an executable application for rendering Neural Radiance Fields in realtime. This is achieved by reimplementing the NeRF functionalities of Nvidias [Instant-NGP](https://github.com/NVlabs/instant-ngp) and rendering the resulting images in realtime using the 3D-Library [FUSEE](https://fusee3d.org/). Since FUSEE uses the .NET Framework, an interface for working with matrices stored on the GPU and managed C# Code is needed. Therefore, this project makes use of [TorchSharp](https://github.com/dotnet/TorchSharp) and follows other reimplementations of Instant-NGP such as [xrnerf](https://github.com/openxrlab/xrnerf) and [torch-ngp](https://github.com/ashawkey/torch-ngp).

## Usage

Currently, changes to the parameters like the dataset are hardcoded and need to be made by editing FuseeNeRF.cs. A User Interface and a .json config file will be added in the future

- Open the .sln file in MS Visual Studio
- specify the launch parameters in FuseeNeRF.cs like this
- Instant-NGPs dataset and the original NeRFs datasets are supported, although most testing has been done on the synthetic lego dataset

```csharp
    string pathToData = @"C:\downloads\nerf_synthetic\nerf_synthetic\lego";

    Device device = cuda.is_available() ? CUDA : CPU;

    DataProvider trainData = new DataProvider(device, pathToData, "transforms_train", "train", downScale: 2.0f, radiusScale: 1.0f, offset: new float[] { 0f, 0f, 0f }, bound: 1.0f, numRays: 2048, preload: false, datasetType: "synthetic");
    DataProvider evalData = new DataProvider(device, pathToData, "transforms_val", "val", downScale: 2.0f, radiusScale: 1.0f, offset: new float[] { 0f, 0f, 0f }, bound: 1.0f, numRays: 2048, preload: false, datasetType: "synthetic");

    NerfRenderer renderer = new NerfRenderer("NerfRenderer");
    
    TorchSharp.Modules.Adam optimizer = optim.Adam(renderer.mlp.getParams(), lr: 0.01, beta1: 0.9, beta2: 0.99, eps: 1e-15);

    Loss<Tensor, Tensor, Tensor> criterion = torch.nn.MSELoss(reduction: nn.Reduction.None);

    Trainer trainer = new Trainer("NGP001", renderer, optimizer, criterion, 1, subdirectoryName: "workspace_lego_synthetic");
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
- On the dev branch, raymarching causes the output tensors memory to become inaccessable by CUDA.
    - this is most likely caused by the CUDA-Kernel ending early due to an error caused by wrong ray calculation, as most rays cant even reach the Bounding Box based on their directions and origins
    - after the [while loop](https://github.com/FabianKowatsch/FuseeNeRF/blob/dev/RaymarchApi/ray_sampler.cu#L60), all output tensors have corrupted underlying memory
    - the directions and origins of the rays are calculated by using the camera transforms extracted from the colmap .json files ([reading  json data](https://github.com/FabianKowatsch/FuseeNeRF/blob/dev/InstantNeRF/DataProvider.cs#L160))
    - the camera poses are then transformed similarly to the [transformation in instant-ngp](https://github.com/NVlabs/instant-ngp/blob/090aed613499ac2dbba3c2cede24befa248ece8a/include/neural-graphics-primitives/nerf_loader.h#L101): [c# version](https://github.com/FabianKowatsch/FuseeNeRF/blob/5a1eca87b29153474a8d295a6101a32eecd574c4/InstantNeRF/Utils.cs#L39C34-L39C34) ([transformation in xrnerf](https://github.com/openxrlab/xrnerf/blob/f8020561b91b27848bf893b34c69a3d17de151c5/xrnerf/datasets/utils/hashnerf.py), [transformation in torch-ngp](https://github.com/ashawkey/torch-ngp/blob/b6e080468925f0bb44827b4f8f0ed08291dcf8a9/nerf/provider.py#L19))
    - the direction of the rays are then calculated with these transformed camera poses [here](https://github.com/FabianKowatsch/FuseeNeRF/blob/5a1eca87b29153474a8d295a6101a32eecd574c4/InstantNeRF/Utils.cs#L39C34-L39C34) ([reference: xrnerf](https://github.com/openxrlab/xrnerf/blob/f8020561b91b27848bf893b34c69a3d17de151c5/xrnerf/datasets/load_data/get_rays.py#L72))

### Future Goals

- Fixing NaN Error
- Configuration
- User Interface

## References

Credits to:
- [Instant-NGP](https://github.com/NVlabs/instant-ngp)
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
- [xrnerf](https://github.com/openxrlab/xrnerf)
- [torch-ngp](https://github.com/ashawkey/torch-ngp)
