using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;

namespace InstantNeRF
{
    public static class VolumeRendering
    {
        [DllImport("RaymarchTools.dll")]
        private static extern void compositeRaysTrainForwardApi(IntPtr sigmas,
        IntPtr rgbs,
        IntPtr deltas,
        IntPtr rays,
        uint M,
        uint N,
        float tThresh,
        IntPtr weightsSum,
        IntPtr depth,
        IntPtr image
        );
        [DllImport("RaymarchTools.dll")]
        private static extern void compositeRaysTrainBackwardApi(IntPtr gradWeightsSum,
            IntPtr gradImage,
            IntPtr sigmas,
            IntPtr rgbs,
            IntPtr deltas,
            IntPtr rays,
            IntPtr weightsSum,
            IntPtr image,
            uint M,
            uint N,
            float tThresh,
            IntPtr gradSigmas,
            IntPtr gradRgbs
            );
        [DllImport("RaymarchTools.dll")]
        private static extern void compositeRaysApi(uint nAlive,
          uint nStep,
          float tThresh,
          IntPtr raysAlive,
          IntPtr raysT,
          IntPtr sigmas,
          IntPtr rgbs,
          IntPtr deltas,
          IntPtr weightsSum,
          IntPtr depth,
          IntPtr image
          );

        public static void compositeRays(
            long nAlive,
            long nStep,
            Tensor raysAlive,
            Tensor raysTerminated,
            Tensor sigmas,
            Tensor rgbs,
            Tensor deltas,
            Tensor weightsSum,
            Tensor depth,
            Tensor image,
            double tThreshhold)
        {
            compositeRaysApi(Convert.ToUInt32(nAlive),
                Convert.ToUInt32(nStep),
                Convert.ToSingle(tThreshhold),
                raysAlive.Handle,
                raysTerminated.Handle,
                sigmas.to_type(torch.float32).Handle,
                rgbs.nan_to_num().to_type(torch.float32).Handle,
                deltas.Handle,
                weightsSum.Handle,
                depth.Handle,
                image.Handle
                );
        }

        public class CompositeRaysTrain
        {
            public CompositeRaysTrain() { }

            private RaymarchContext? ctx;
            public (Tensor weightsSum, Tensor depth, Tensor image) Forward(Tensor sigmas, Tensor rgbs, Tensor deltas, Tensor rays, double tThreshhold = 1e-4)
            {
                sigmas = sigmas.contiguous();
                rgbs = rgbs.contiguous();
                long M = sigmas.shape[0];
                long N = rays.shape[0];

                //Console.WriteLine("M:" + M);
                //Console.WriteLine("N:" + N);

                ScalarType type = sigmas.dtype;
                Device device = sigmas.device;

                rgbs = rgbs.to_type(type);

                Tensor weightsSum = torch.empty(N, type, device, requires_grad: true);
                Tensor depth = torch.empty(N, type, device, requires_grad: true);
                Tensor image = torch.empty(new long[] { N, 3 }, type, device, requires_grad: true);

                compositeRaysTrainForwardApi(sigmas.Handle, rgbs.Handle, deltas.Handle, rays.Handle, Convert.ToUInt32(M), Convert.ToUInt32(N), Convert.ToSingle(tThreshhold), weightsSum.Handle, depth.Handle, image.Handle);

                this.ctx = new RaymarchContext(M, N, tThreshhold);
                ctx.saveForBackward(new List<Tensor> { sigmas, rgbs, deltas, rays, weightsSum, depth, image });

                return (weightsSum, depth, image);
            }

            public void Backward()
            {

                if (this.ctx != null)
                {


                    Tensor sigmas = ctx.savedTensors[0];
                    Tensor rgbs = ctx.savedTensors[1];
                    Tensor deltas = ctx.savedTensors[2];
                    Tensor rays = ctx.savedTensors[3];
                    Tensor weightsSum = ctx.savedTensors[4];
                    Tensor depth = ctx.savedTensors[5];
                    Tensor image = ctx.savedTensors[6];

                    Tensor imageGrad = image.grad() ?? torch.empty(0);
                    Tensor weightsSumGrad = weightsSum.grad() ?? torch.empty(0);
                    Tensor depthGrad = depth.grad() ?? torch.zeros_like(depth);

                    if (weightsSumGrad.numel() == 0 || imageGrad.numel() == 0 || depthGrad.numel() == 0)
                    {
                        Console.WriteLine("no grads for backward pass");
                        return;
                    }
                    weightsSumGrad = weightsSumGrad.contiguous();
                    imageGrad = imageGrad.contiguous();
                    Tensor sigmasGrad = torch.zeros_like(sigmas);
                    Tensor rgbsGrad = torch.zeros_like(rgbs);


                    compositeRaysTrainBackwardApi(
                        weightsSumGrad.Handle,
                        imageGrad.Handle,
                        sigmas.Handle,
                        rgbs.Handle,
                        deltas.Handle,
                        rays.Handle,
                        weightsSum.Handle,
                        image.Handle,
                        Convert.ToUInt32(ctx.M),
                        Convert.ToUInt32(ctx.N),
                        Convert.ToSingle(ctx.tThreshhold),
                        sigmasGrad.Handle,
                        rgbsGrad.Handle
                        );

                    //Utils.printMean(depthGrad, "depthGrad");
                    //Utils.printMean(weightsSumGrad, "weightssumGrad");
                    //Utils.printMean(imageGrad, "imageGrad");
                    rgbs.backward(grad_tensors: new List<Tensor>() { rgbsGrad });
                    sigmas.backward(grad_tensors: new List<Tensor>() { sigmasGrad });
                }

            }


        }

    }
}
