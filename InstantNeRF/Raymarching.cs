using InstantNeRF;
using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


namespace InstantNeRF
{
    public static class RaymarchUtils
    {
        [DllImport("RaymarchTools.dll")]
        private static extern void nearFarFromAabbApi(IntPtr rays_o, IntPtr rays_d, IntPtr aabb, uint N, float min_near, IntPtr nears, IntPtr fars);

        [DllImport("RaymarchTools.dll")]
        private static extern void sphereFomRayApi(IntPtr rays_o, IntPtr rays_d, IntPtr aabb, uint N, float radius, IntPtr coords);

        [DllImport("RaymarchTools.dll")]
        private static extern void morton3DApi(IntPtr coords, uint N, IntPtr indices);

        [DllImport("RaymarchTools.dll")]
        private static extern void morton3DInvertApi(IntPtr indices, uint N, IntPtr coords);

        [DllImport("RaymarchTools.dll")]
        private static extern void packbitsApi(IntPtr grid, uint N, float densityThreshhold, IntPtr bitfield);

        [DllImport("RaymarchTools.dll")]
        private static extern void raymarchTrainApi(IntPtr rays_o,
            IntPtr rays_d,
            IntPtr grid,
            float bound,
            float dtGamma,
            uint maxSteps,
            uint N,
            uint C,
            uint H,
            uint M,
            IntPtr nears,
            IntPtr fars,
            IntPtr xyzs,
            IntPtr dirs,
            IntPtr deltas,
            IntPtr rays,
            IntPtr counter,
            IntPtr noises
            );

    

        [DllImport("RaymarchTools.dll")]
        private static extern void raymarchApi(uint nAlive,
            uint nStep,
            IntPtr raysAlive,
            IntPtr raysT,
            IntPtr rays_o,
            IntPtr rays_d,
            float bound,
            float dtGamma,
            uint maxSteps,
            uint C,
            uint H,
            IntPtr grid,
            IntPtr nears,
            IntPtr fars,
            IntPtr xyzs,
            IntPtr dirs,
            IntPtr deltas,
            IntPtr noises
            );

      

        public static Tensor morton3D(Tensor coords)
        {
            if (!coords.is_cuda)
                coords.to(torch.CUDA);
            int N = Convert.ToInt32(coords.shape[0]);
            Tensor indices = torch.empty(N, dtype: torch.int32, device: coords.device);
            morton3DApi(torch.IntTensor(coords).Handle, Convert.ToUInt32(N), indices.Handle);
            return indices;
        }
        public static Tensor morton3DInvert(Tensor indices)
        {
            if (!indices.is_cuda)
                indices.to(torch.CUDA);
            long N = indices.shape[0];
            Tensor coords = torch.empty(new long[] { N, 3 }, dtype: torch.int32, device: indices.device);
            morton3DInvertApi(torch.IntTensor(indices).Handle, Convert.ToUInt32(N), coords.Handle);
            return coords;
        }
        public static Tensor packbits(Tensor grid, float threshHold, Tensor bitfield)
        {
            if (!grid.is_cuda)
                grid.to(torch.CUDA);
            grid = grid.contiguous();
            long C = grid.shape[0];
            long H3 = grid.shape[1];
            double Ndouble = C * H3 / 8d;
            uint N = Convert.ToUInt32(Math.Floor(Ndouble));
            grid = torch.FloatTensor(grid);
            packbitsApi(grid.Handle, N, threshHold, bitfield.Handle);
            return bitfield;
        }
        public static (Tensor nears, Tensor fars) nearFarFromAabb(Tensor raysOrigin, Tensor raysDirection, Tensor aabb, float minNear = 0.2f)
        {
            if (!raysOrigin.is_cuda)
                raysOrigin.to(torch.CUDA);
            if (!raysDirection.is_cuda)
                raysDirection.to(torch.CUDA);

            raysOrigin = FloatTensor(raysOrigin).contiguous().view(-1, 3);
            raysDirection = FloatTensor(raysDirection).contiguous().view(-1, 3);

            long N = raysOrigin.shape[0];
            Tensor nears = torch.empty(N, dtype: raysOrigin.dtype, device: raysOrigin.device);
            Tensor fars = torch.empty(N, dtype: raysOrigin.dtype, device: raysOrigin.device);

            nearFarFromAabbApi(raysOrigin.Handle, raysDirection.Handle, aabb.Handle, Convert.ToUInt32(N), minNear, nears.Handle, fars.Handle);
            return (nears, fars);
        }

        public static Tensor sphereFromRay(Tensor raysOrigin, Tensor raysDirection, Tensor aabb, float radius)
        {
            if (!raysOrigin.is_cuda)
                raysOrigin.to(torch.CUDA);
            if (!raysDirection.is_cuda)
                raysDirection.to(torch.CUDA);

            raysOrigin = FloatTensor(raysOrigin).contiguous().view(-1, 3);
            raysDirection = FloatTensor(raysDirection).contiguous().view(-1, 3);
            long N = raysOrigin.shape[0];
            Tensor coords = torch.empty(N, dtype: raysOrigin.dtype, device: raysOrigin.device);
            sphereFomRayApi(raysOrigin.Handle, raysDirection.Handle, aabb.Handle, Convert.ToUInt32(N), radius, coords.Handle);
            return coords;
        }

        public static (Tensor xyzs, Tensor dirs, Tensor deltas, Tensor rays) marchRaysTrain(
            Tensor raysOrigin,
            Tensor raysDirection,
            float bound,
            Tensor densityBitfield,
            uint C,
            uint H,
            Tensor nears,
            Tensor fars,
            Tensor stepCounter,
            int meanCount = -1,
            bool perturb = false,
            int align = -1,
            bool forceAllRays = false,
            float gammaGrad = 0f,
            uint maxSteps = 1024
            )
        {
            if (!raysOrigin.is_cuda)
                raysOrigin.to(torch.CUDA);
            if (!raysDirection.is_cuda)
                raysDirection.to(torch.CUDA);
            if (!densityBitfield.is_cuda)
                densityBitfield.to(torch.CUDA);

            raysOrigin = raysOrigin.contiguous().view(-1, 3);
            raysDirection = raysDirection.contiguous().view(-1, 3);
            densityBitfield = densityBitfield.contiguous();

            Device device = raysOrigin.device;
            ScalarType type = raysDirection.dtype;
            Console.WriteLine("  ");
            Console.WriteLine("DEVICE: " + device);
            Console.WriteLine("bound: " + bound);
            Console.WriteLine("  ");
            long N = raysOrigin.shape[0];
            long M = N * maxSteps;

            // running average based on previous epoch (mimics `measured_batch_size_before_compaction` in instant-ngp)
            // It estimates the max points number to enable faster training, but will lead to random ignored rays if underestimated.
            if (!forceAllRays && meanCount > 0)
            {
                if (align > 0)
                    meanCount += align - meanCount % align;
                M = meanCount;
            }
            //Console.WriteLine("M raymarch:" + M);
            //Console.WriteLine("N raymarch:" + N);

            Tensor xyzs = torch.zeros(new long[] { M, 3 }, dtype: type, device: device, requires_grad: true);
            Tensor dirs = torch.zeros(new long[] { M, 3 }, dtype: type, device: device, requires_grad: true);
            Tensor deltas = torch.zeros(new long[] { M, 2 }, dtype: type, device: device);
            Tensor rays = torch.zeros(new long[] { N, 3 }, dtype: torch.int32, device: device);

            Tensor noises = perturb ? torch.rand(N, type, device) : torch.zeros(N, type, device);

            stepCounter = stepCounter.numel() == 0 ? torch.zeros(2, type, device) : stepCounter;

            Utils.printDims(stepCounter, "counter");

            raymarchTrainApi(
                raysOrigin.Handle,
                raysDirection.Handle,
                densityBitfield.Handle,
                bound,
                gammaGrad,
                maxSteps,
                Convert.ToUInt32(N),
                C,
                H,
                Convert.ToUInt32(M),
                nears.Handle,
                fars.Handle,
                xyzs.Handle,
                dirs.Handle,
                deltas.Handle,
                rays.Handle,
                stepCounter.Handle,
                noises.Handle
                );


            if (forceAllRays || meanCount <= 0)
            {
                int m = stepCounter[0].item<int>();
                if (align > 0)
                    m += align - m % align;
                xyzs = xyzs.slice(0, 0, m, 1);
                dirs = dirs.slice(0, 0, m, 1);
                deltas = deltas.slice(0, 0, m, 1);
            }
            //Console.WriteLine("xyzs:" + xyzs.size(0));
            //Console.WriteLine("dirs:" + dirs.size(0));
            //Console.WriteLine("rays:" + rays.size(0));
            return (xyzs, dirs, deltas, rays);
        }

        
        public static (Tensor xyzs, Tensor dirs, Tensor deltas) marchRays(
            long nAlive,
            long nStep,
            Tensor raysAlive,
            Tensor raysTerminated,
            Tensor raysOrigin,
            Tensor raysDirection,
            float bound,
            Tensor densityBitfield,
            uint C,
            uint H,
            Tensor nears,
            Tensor fars,
            bool perturb = false,
            int align = -1,
            float gammaGrad = 0f,
            uint maxSteps = 1024
            )
        {
            if (!raysOrigin.is_cuda)
                raysOrigin.to(torch.CUDA);
            if (!raysDirection.is_cuda)
                raysDirection.to(torch.CUDA);
            if (!densityBitfield.is_cuda)
                densityBitfield.to(torch.CUDA);

            raysOrigin = raysOrigin.contiguous().view(-1, 3);
            raysDirection = raysDirection.contiguous().view(-1, 3);
            densityBitfield = densityBitfield.contiguous();
            Device device = raysOrigin.device;
            ScalarType type = raysDirection.dtype;

            long M = nAlive * nStep;

            if (align > 0)
                M += align - (M % align);

            Tensor xyzs = torch.zeros(new long[] { M, 3 }, type, device);
            Tensor dirs = torch.zeros(new long[] { M, 3 }, type, device);
            Tensor deltas = torch.zeros(new long[] { M, 2 }, type, device);

            Tensor noises = perturb ? torch.rand(nAlive, type, device) : torch.zeros(nAlive, type, device);


            raymarchApi(
                Convert.ToUInt32(nAlive),
                Convert.ToUInt32(nStep),
                raysAlive.Handle,
                raysTerminated.Handle,
                raysOrigin.Handle,
                raysDirection.Handle,
                bound,
                gammaGrad,
                maxSteps,
                C,
                H,
                densityBitfield.Handle,
                nears.Handle,
                fars.Handle,
                xyzs.Handle,
                dirs.Handle,
                deltas.Handle,
                noises.Handle
                );

            return (xyzs, dirs, deltas);
        }



    }
}
