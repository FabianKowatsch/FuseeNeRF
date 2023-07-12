﻿
using System.Runtime.InteropServices;
using static TorchSharp.torch;
using TorchSharp;

namespace InstantNeRF
{
    public static class VolumeRenderingApi
    {
        [DllImport("RaymarchApi.dll")]
        private static extern void calculateRGBsForwardApi(
            IntPtr network_output,
            IntPtr coords_in,
            IntPtr rays_numsteps,
            IntPtr rays_numsteps_compacted,
            IntPtr training_background_color,
            int rgb_activation_i,
            int density_activation_i,
            float aabb0,
            float aabb1,
            IntPtr rgb_output);

        [DllImport("RaymarchApi.dll")]
        private static extern void calculateRGBsBackwardApi(
            IntPtr network_output,
            IntPtr rays_numsteps_compacted,
            IntPtr coords_in,
            IntPtr grad_x,
            IntPtr rgb_output,
            IntPtr density_grid_mean,
            int rgb_activation_i,
            int density_activation_i,
            float aabb0,
            float aabb1,
            IntPtr dloss_doutput);

        [DllImport("RaymarchApi.dll")]
        private static extern void calculateRGBsInferenceApi(
            IntPtr network_output,
            IntPtr coords_in,
            IntPtr rays_numsteps,
            IntPtr bg_color_cpu,
            int rgb_activation_i,
            int density_activation_i,
            float aabb0,
            float aabb1,
            IntPtr rgb_output,
            IntPtr alpha_output);

        public static (Tensor colors, Tensor alphas) Inference(
           Tensor output,
           Tensor positions,
           Tensor rayNumsteps,
           Tensor bgColorInference,
           int rgbActivation,
           int densityActivation,
           float aabb0,
           float aabb1)
        {
            Device device = output.device;
            positions.detach_();
            rayNumsteps.detach_();
            long nRaysPerBatch = rayNumsteps.size(0);
            Tensor rgbs = torch.zeros(new long[] { nRaysPerBatch, 3 }, torch.float32, device);
            Tensor alphas = torch.zeros(new long[] { nRaysPerBatch, 1 }, torch.float32, device);
            calculateRGBsInferenceApi(output.Handle, positions.Handle, rayNumsteps.Handle, bgColorInference.Handle, rgbActivation, densityActivation, aabb0, aabb1, rgbs.Handle, alphas.Handle);

            return (rgbs.detach(), alphas.detach());
        }


        public class DifferentiableRenderer
        {
            public DifferentiableRenderer() { }

            private VolumeRenderingContext? ctx;
            public Tensor Forward(Tensor output,
                Tensor positions, 
                Tensor rayNumsteps, 
                Tensor rayNumstepsCompacted, 
                Tensor bgColorTrain, 
                Tensor densityMean, 
                int rgbActivation,
                int densityActivation,
                float aabbMin, 
                float aabbMax)
            {
                Device device = output.device;
                positions.detach_();
                rayNumsteps.detach_();
                rayNumstepsCompacted.detach_();
                densityMean.detach_();
                long nRaysPerBatch = rayNumsteps.size(0);
                Tensor rgbs = torch.zeros(new long[] { nRaysPerBatch, 3 }, torch.float32, device);

                calculateRGBsForwardApi(output.Handle,
                    positions.Handle,
                    rayNumsteps.Handle,
                    rayNumstepsCompacted.Handle,
                    bgColorTrain.Handle,
                    rgbActivation,
                    densityActivation,
                    aabbMin,
                    aabbMax,
                    rgbs.Handle);

                this.ctx = new VolumeRenderingContext(rgbActivation, densityActivation, aabbMin, aabbMax);
                ctx.saveForBackward(new List<Tensor> { output, rayNumsteps, rayNumstepsCompacted, positions, rgbs, densityMean });

                return rgbs;
            }

            public void Backward()
            {

                if (this.ctx != null)
                {
                    Tensor output = ctx.savedTensors[0];
                    Tensor rayNumsteps = ctx.savedTensors[1];
                    Tensor rayNumstepsCompacted = ctx.savedTensors[2];
                    Tensor positions = ctx.savedTensors[3];
                    Tensor rgbs = ctx.savedTensors[4];
                    Tensor densityMean = ctx.savedTensors[5];

                    long nElements = output.size(0);
                    Device device = output.device;

                    Tensor outputGrad = torch.zeros(new long[] { nElements, 4 }, torch.float32, device);
                    Tensor colorsGrad = rgbs.grad() ?? torch.zeros_like(rgbs);

                    calculateRGBsBackwardApi(output.Handle, 
                        rayNumstepsCompacted.Handle,
                        positions.Handle,
                        colorsGrad.Handle,
                        rgbs.Handle,
                        densityMean.Handle,
                        ctx.rgbActivation,
                        ctx.densityActivation,
                        ctx.aabbMin,
                        ctx.aabbMax,
                        outputGrad.Handle);

                    output.backward(new List<Tensor>() { outputGrad });
                }

            }


        }

    }
}
