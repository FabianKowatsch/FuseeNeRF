using ICSharpCode.SharpZipLib.Zip;
using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static Tensorboard.TensorShapeProto.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;

namespace InstantNeRF
{
    public static class Utils
    {

        public static Tensor linearToSrgb(Tensor x)
        {
            return torch.where(x < 0.0031308, 12.92 * x, 1.055 * torch.pow(x, 0.41666) - 0.055);
        }
        public static Tensor srgbToLinear(Tensor x)
        {
            return torch.where(x < 0.04045, x / 12.92, torch.pow((x + 0.055) / 1.055, 2.4));
        }
        public static NerfMode modeFromString(string str)
        {
            switch (str)
            {
                case "train":
                    return NerfMode.TRAIN;
                case "val":
                    return NerfMode.VAL;
                case "test":
                    return NerfMode.TEST;
                default:
                    throw new Exception("Wrong argument");
            }
        }
        public static float[,] arrayToNGPMatrix(float[,] pose, float scale, float[] offset)
        {
            float[,] output = new float[,]
            {
                {pose[1,0], -pose[1,1], -pose[1,2], pose[1,3] * scale + offset[0]},
                 {pose[2,0], -pose[2,1], -pose[2,2], pose[2,3] * scale + offset[1]},
                  {pose[0,0], -pose[0,1], -pose[0,2], pose[0,3] * scale + offset[2]},
                   {0, 0, 0, 1},

            };

            return output;
        }
        public static float[,] fuseeMatrixToNGPMatrix(float[] pose, float scale, float[] offset)
        {
            float[,] output = new float[,]
            {
                {pose[4], -pose[5], -pose[6], pose[7] * scale + offset[0]},
                 {pose[8], -pose[9], -pose[10], pose[11] * scale + offset[1]},
                  {pose[0], -pose[1], -pose[2], pose[3] * scale + offset[2]},
                   {0, 0, 0, 1},

            };

            return output;
        }

        public static Dictionary<string, Tensor> getRays(Tensor poses, Tensor intrinsics, int height, int width, Tensor errorMap, int n = -1, int patchSize = 1)
        {
            poses = poses.unsqueeze(0);
            errorMap = errorMap.unsqueeze(0);
            Device device = poses.device;
            long batchSize = poses.shape[0];
            Tensor fX = intrinsics[0];
            Tensor fY = intrinsics[1];
            Tensor cX = intrinsics[2];
            Tensor cY = intrinsics[3];

            Tensor[] ij = torch.meshgrid(new List<Tensor> { torch.linspace(0, width - 1, width, device: device), torch.linspace(0, height - 1, height, device: device) });
            Tensor i = ij[0].t().reshape(1, height * width).expand(batchSize, height * width) + 0.5f;
            Tensor j = ij[1].t().reshape(1, height * width).expand(batchSize, height * width) + 0.5f;

            Dictionary<string, Tensor> results = new Dictionary<string, Tensor>();

            Tensor indices;
            if (n > 0)
            {
                n = Math.Min(n, height * width);
                if (patchSize > 1)
                {
                    int numPatch = n / (int)Math.Pow(patchSize, 2);
                    Tensor indicesX = torch.randint(0, height - patchSize, size: numPatch, device: device);
                    Tensor indicesY = torch.randint(0, width - patchSize, size: numPatch, device: device);
                    indices = torch.stack(new List<Tensor> { indicesX, indicesY });
                    Tensor[] patchedIJ = torch.meshgrid(new List<Tensor> { torch.arange(patchSize, device: device), torch.arange(patchSize, device: device) });
                    Tensor offsets = torch.stack(new List<Tensor> { patchedIJ[0].reshape(-1), patchedIJ[1].reshape(-1) });
                    indices = indices.unsqueeze(1) + offsets.unsqueeze(0);
                    indices = indices.view(-1, 2);
                    indices = indices.select(1, 0) * width + indices.select(1, 1);
                    indices = indices.expand(batchSize, n);
                }
                else if (errorMap.numel() == 0)
                {
                    indices = torch.randint(0, height * width, size: n, device: device);
                    indices = indices.expand(batchSize, n);
                }
                else
                {
                    Tensor indicesCoarse = torch.multinomial(errorMap.to(device), n, replacement: false);
                    Tensor indicesX = indicesCoarse.floor_divide(128);
                    Tensor indicesY = indicesCoarse % 128;
                    int sampledX = height / 128;
                    int sampledY = width / 128;
                    indicesX = torch.LongTensor(indicesX * sampledX + torch.rand(batchSize, n, device: device) * sampledX).clamp(max: height - 1);
                    indicesY = torch.LongTensor(indicesY * sampledY + torch.rand(batchSize, n, device: device) * sampledY).clamp(max: width - 1);
                    indices = indicesX * width + indicesY;

                    results["indicesCoarse"] = indicesCoarse;

                }
                i = torch.gather(i, -1, indices);
                j = torch.gather(j, -1, indices);
                
            }
            else
            {
                indices = torch.arange(height * width, device: device).expand(batchSize, height * width);
            }
            results["indices"] = indices;
            Tensor zs = torch.ones_like(i);
            Tensor xs = (i - cX) / fX * zs;
            Tensor ys = (j - cY) / fY * zs;

            Tensor directions = torch.stack(new List<Tensor>() { xs, ys, zs }, dim: -1);
            directions = directions / torch.norm(directions, dimension: -1, keepdim: true);


            Tensor raysDirection = directions.matmul(poses.slice(1, 0, 3, 1).slice(2, 0, 3, 1).transpose(-1, -2));
            Tensor raysOrigin = poses.slice(1, 0, 3, 1).select(2, 3);
            raysOrigin = raysOrigin.expand_as(raysDirection);
            results["raysOrigin"] = raysOrigin;
            results["raysDirection"] = raysDirection;

            return results;
        }

        public static void printFirstNValues(Tensor t, int n, string name = "Tensor")
        {
            long dims = t.Dimensions;
            Console.WriteLine(name + ": type: " + t.dtype + " dims: " + dims + " numel: " + t.numel() + " grad: " + t.requires_grad + " leaf: " + t.is_leaf);
            for (int i = 0; i < n; i++)
            {
                if (t.size(Convert.ToInt32(dims) - 1) > i)
                {
                    switch (dims)
                    {
                        case 1:
                            {
                                t[i].print();
                                break;
                            }

                        case 2:
                            {
                                t[0, i].print();
                                break;
                            }

                        case 3:
                            {
                                t[0, 0, i].print();
                                break;
                            }

                        default: break;
                    }
                }

            }
            Console.WriteLine();
        }
        public static void printMean(Tensor t, string name = "Tensor")
        {
            long dims = t.Dimensions;
            Console.WriteLine(name + ": type: " + t.dtype + " dims: " + dims + " numel: " + t.numel() + " grad: " + t.requires_grad + " leaf: " + t.is_leaf);
            if (t.dtype == torch.float32 || t.dtype == torch.float16)
            {
                Console.Write("mean: ");
                t.mean().print();

            }
            else
            {
                Console.Write("sum: ");
                t.sum().print();
            }
            Console.WriteLine();
        }
    }
    public enum ColorSpace
    {
        Linear,
        SRGB
    }
    public enum NerfMode
    {
        TRAIN = 0,
        VAL = 1,
        TEST = 2
    }

    public class ExponentialMovingAverage
    {

        private float decay;
        private List<Parameter> shadowParams;
        private List<Parameter> currentParams;
        private List<Parameter> collectedParams;
        public ExponentialMovingAverage(float decay, IEnumerable<Parameter> parameters)
        {
            this.decay = decay;
            this.shadowParams = initShadowParams(parameters.ToList());
            this.currentParams = parameters.ToList();
            this.collectedParams = new List<Parameter>();
        }
        private List<Parameter> initShadowParams(List<Parameter> parameters)
        {
            List<Parameter> result = new List<Parameter>();
            parameters.ForEach(param =>
            {
                result.Add(param.clone().detach().AsParameter());
            });
            return result;
        }



        public void update()
        {
            float decay = this.decay;

            float oneMinusDecay = 1.0f - decay;
            var x = shadowParams.Zip(currentParams);
            using (torch.no_grad())
            {
                foreach (var item in x)
                {
                    Tensor tmp = (item.First - item.Second);
                    tmp = tmp.mul_(oneMinusDecay);
                    item.First.sub_(tmp);
                }
            }

        }
        public void copyTo()
        {
            var x = shadowParams.Zip(currentParams);
            foreach (var item in x)
            {
                item.Second.copy_(item.First);
            }
        }
        public void store()
        {
            currentParams.ForEach(param =>
            {
                collectedParams.Add(param.clone().AsParameter());
            });
        }
        public void reStore()
        {
            var x = collectedParams.Zip(currentParams);
            foreach (var item in x)
            {
                item.Second.copy_(item.First);
            }
        }

        public Dictionary<string, dynamic> state_dict()
        {
            return new Dictionary<string, dynamic> { { "decay", this.decay }, { "shadow_params", shadowParams }, { "collected_params", collectedParams } };

        }

        public void load_state_dict(Dictionary<string, dynamic> state_dict)
        {
            this.decay = state_dict["decay"];
            this.shadowParams = state_dict["shadow_params"];
            this.collectedParams = state_dict["collectedParams"];
        }

    }
}
