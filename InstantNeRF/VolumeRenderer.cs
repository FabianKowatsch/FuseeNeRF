﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;
using static InstantNeRF.VolumeRenderingApi;

namespace InstantNeRF
{
    public class VolumeRenderer: nn.Module
    {
        private Tensor bgColor;
        private DifferentiableRenderer renderer;
        public VolumeRenderer(float[] bgColor) : base("VolumeRenderer")
        {
            this.bgColor = torch.tensor(bgColor, torch.float32);
            this.renderer = new DifferentiableRenderer();

        }

        public Dictionary<string, Tensor> forward(GridSampler sampler, Dictionary<string, Tensor> data)
        {
            Tensor networkOutput = data["raw"];
            Tensor coords = data["coords"];
            Tensor numSteps = data["rayNumsteps"];
            
            DataInfo dataInfo = sampler.getData();
            float aabbMin = dataInfo.aabbMin;
            float aabbMax = dataInfo.aabbMax;

            Dictionary<string, Tensor> result = new Dictionary<string, Tensor>() { };
            if (this.training)
            {
                // Use a DifferentiableRenderer object to store results for the backward pass, similar to AutogradFunction
                Tensor numStepsCompacted = data["rayNumstepsCompacted"];
                Tensor bgColor = data["bgColor"].detach();
                this.renderer = new DifferentiableRenderer();
                Tensor rgbs = renderer.Forward(
                    networkOutput, 
                    coords, 
                    numSteps, 
                    numStepsCompacted, 
                    bgColor, 
                    sampler.densityMean, 
                    sampler.rgbActivation, 
                    sampler.densityActivation, 
                    aabbMin, 
                    aabbMax);
                result.Add("rgb", rgbs);
                return result;
            }
            else
            {
                (Tensor rgbs, Tensor alphas) = VolumeRenderingApi.Inference(
                    networkOutput, 
                    coords, 
                    numSteps, 
                    this.bgColor, 
                    sampler.rgbActivation, 
                    sampler.densityActivation, 
                    aabbMin, 
                    aabbMax);
                result.Add("rgb", rgbs);
                result.Add("alpha", alphas);
                return result;
            }
        }
        public Tensor backward(Tensor groundtruthRgbs)
        {
            return this.renderer.Backward(groundtruthRgbs);
        }

    }
}
