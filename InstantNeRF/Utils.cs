/*
Copyright (c) 2022 hawkey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

//custom implementation based on https://github.com/ashawkey/torch-ngp/blob/main/nerf/provider.py and https://github.com/ashawkey/torch-ngp/blob/main/nerf/utils.py

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

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
        public static Tensor posesToNGP(Tensor poses, float scale, float[] offset)
        {

            List<Tensor> posesList = new List<Tensor>();

            for (int i = 0; i < poses.shape[0]; i++)
            {
                Tensor pose = matrixToNGP(poses.select(0, i), scale, offset);
                posesList.Add(pose);
            }
            Tensor result = torch.stack(posesList).to(float32);
            return result;
        }
        public static Tensor matrixToNGP(Tensor matrix, float scale, float[] offset)
        {
            // used in torch-ngp
            /*
           TorchSharp.Utils.TensorAccessor<float> pose = matrix.data<float>();
           float[,] output = new float[,]
           {
               {pose[1,0], -pose[1,1], -pose[1,2], pose[1,3] * scale + offset[0]},
                {pose[2,0], -pose[2,1], -pose[2,2], pose[2,3] * scale + offset[1]},
                 {pose[0,0], -pose[0,1], -pose[0,2], pose[0,3] * scale + offset[2]}

           };
           return torch.from_array(output, dtype: torch.float32);
           */

            // following instant-ngps / XRNerfs matrix transformations
            Tensor ngpMatrix = matrix.slice(0, 0, 3, 1).slice(1, 0, 4, 1).t();
            ngpMatrix[0] *= 1;
            ngpMatrix[1] *= -1;
            ngpMatrix[2] *= -1;
            ngpMatrix[3, 0] = ngpMatrix[3, 0] * scale + offset[0];
            ngpMatrix[3, 1] = ngpMatrix[3, 1] * scale + offset[1];
            ngpMatrix[3, 2] = ngpMatrix[3, 2] * scale + offset[2];

            Tensor permutation = torch.from_array(new long[] { 1, 2, 0 }, dtype: torch.int64);
            ngpMatrix = ngpMatrix.index_select(1, permutation);
            return ngpMatrix.t();

        }

        public static Tensor fuseeMatrixToNGP(Tensor matrix, float scale, float[] offset)
        {
            Tensor ngpMatrix = matrix.slice(0, 0, 3, 1).slice(1, 0, 4, 1).t();

            ngpMatrix[0] *= 1;
            ngpMatrix[1] *= -1;
            ngpMatrix[2] *= -1;
            ngpMatrix[3, 0] = ngpMatrix[3, 0] * scale + offset[0];
            ngpMatrix[3, 1] = ngpMatrix[3, 1] * scale + offset[1];
            ngpMatrix[3, 2] = ngpMatrix[3, 2] * scale + offset[2];

            return ngpMatrix.t();
        }

        public static Tensor loadRays(Tensor poses, Tensor images, Tensor K, int width, int height)
        {
            Tensor result;
            List<Tensor> raysList = new List<Tensor>();

            for (int i = 0; i < poses.shape[0]; i++)
            {
                (Tensor origins, Tensor dirs) = getRaysFromPose(width, height, K, poses[i]);

                Tensor raysResult = torch.concatenate(new List<Tensor>() { origins, dirs }, 2);
                raysList.Add(raysResult);
            }
            Tensor rays = torch.stack(raysList, 0);
            result = torch.concatenate(new List<Tensor>() { rays, images }, 3); // [N, H, W, ro[3] + rd[3] + rgba[4], added rgba]

            Tensor imageIndices = torch.arange(images.shape[0]).reshape(-1, 1, 1, 1); // [N,1,1,1]

            imageIndices = torch.broadcast_to(imageIndices, result.shape[0], result.shape[1], result.shape[2], 1); // [N, H, W, 1]

            result = torch.concatenate(new List<Tensor>() { result, imageIndices }, 3); // [N, H, W, 10+1, added indices]

            result = result.reshape(-1, 11).to(float32); // [N*H*W, 10+1]

            return result;
        }

        public static (Tensor rayOrigins, Tensor rayDirections) getRaysFromPose(int width, int height, Tensor K, Tensor camToWorld)
        {
            // approach following the work in XRNerf

            Tensor[] ij = torch.meshgrid(new List<Tensor> { torch.arange(0, width, 1, float32), torch.arange(0, height, 1, float32) });
            Tensor i = ij[0].t() + 0.5f;
            Tensor j = ij[1].t() + 0.5f;
            Tensor directions = torch.stack(new List<Tensor> { (i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], torch.ones_like(i) }, -1);
            Tensor camToWorld3x3 = camToWorld.slice(0, 0, 3, 1).slice(1, 0, 3, 1);
            Tensor rayDirections = torch.matmul(camToWorld3x3, directions.unsqueeze(3)).select(-1, 0);
            rayDirections = rayDirections / torch.norm(rayDirections, -1, true);

            Tensor rayOrigins = torch.broadcast_to(camToWorld.slice(0, 0, 3, 1).select(-1, camToWorld.size(-1) - 1), rayDirections.shape);


            // alternate approach following the strict algorithm of generating rays, similar results
            /*
            Tensor[] ij = torch.meshgrid(new List<Tensor> { torch.arange(0, height, 1, float32), torch.arange(0, width, 1, float32) });
            Tensor iNorm = (ij[0] - K[0, 2]) / K[0, 0];
            Tensor jNorm = (ij[1] - K[1, 2]) / K[1, 1];
            Tensor rayDirsCam = torch.stack(new List<Tensor>() { iNorm,  jNorm, torch.ones_like(jNorm) }, -1);
            Tensor rotation = camToWorld.slice(0, 0, 3, 1).slice(1, 0, 3, 1);
            Tensor translation = camToWorld.slice(0, 0, 3, 1).select(1, 3);
            Tensor rayDirections = torch.matmul(rayDirsCam, rotation.t());
            Tensor rayOrigins = translation.view(1, 1, 3).expand(height, width, -1);

            rayDirections = rayDirections / torch.norm(rayDirections, -1, keepdim: true);
            */

            return (rayOrigins, rayDirections);
        }

        // Helper functions for debugging
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
        }

        public static void printDims(Tensor t, string name = "Tensor")
        {
            Console.WriteLine(name + ": type: " + t.dtype + " device: " + t.device + " n: " + t.numel() + " contigous: " + t.is_contiguous() + " grad: " + t.requires_grad + " leaf: " + t.is_leaf);
            for (int i = 0; i < t.shape.Length; i++)
            {
                Console.WriteLine(name + " : " + i + " | " + t.shape[i]);
            }
            Console.WriteLine(" ");
        }

        public static void print2DTensor(Tensor t, string name = "Tensor")
        {
            Console.WriteLine(name + ": type: " + t.dtype + " device: " + t.device + " n: " + t.numel());
            for (int i = 0; i < t.size(0); i++)
            {

                for (int j = 0; j < t.size(1); j++)
                {
                    Console.WriteLine(name + " : [" + i + "," + j + "]" + " = " + t[i, j].item<float>());
                }
            }
            Console.WriteLine(" ");
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

    // unused since the addition of tiny-cuda-nn's optimizer
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
