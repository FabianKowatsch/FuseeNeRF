using TorchSharp;
using static TorchSharp.torch;
using Modules;
using TorchSharp.Modules;
using System.Reflection.Metadata.Ecma335;
using InstantNeRF;

namespace InstantNeRF
{
    public class NerfRenderer : nn.Module
    {
        public MLP mlp;
        public float meanDensity;
        public int meanCount;
        private float bound;
        private float densityScale;
        private float densityThreshhold;
        private float minNear;
        private long cascade;
        private long gridSize;
        private int iterDensity;
        private int localStep;
        private Tensor densityGrid;
        private Tensor densityBitfield;
        private Tensor aabbTrain;
        private Tensor aabbInference;
        private Tensor stepCounter;
        public NerfRenderer(string name, float bound = 1.0f, float densityScale = 1f, float densityThreshhold = 0.01f, float minNear = 0.2f) : base(name)
        {
            this.bound = bound;
            this.densityScale = densityScale;
            this.densityThreshhold = densityThreshhold;
            this.gridSize = 64L;
            this.minNear = minNear;
            cascade = 1 + Convert.ToInt64(Math.Ceiling(Math.Log2(Convert.ToDouble(bound))));
            this.aabbTrain = torch.FloatTensor(torch.from_array(new float[] { -bound, -bound, -bound, bound, bound, bound }));
            this.aabbInference = aabbTrain.clone();
            long gridSize3D = Convert.ToInt64(Math.Pow((double)gridSize, 3));
            this.densityGrid = torch.zeros(new long[] { cascade, gridSize3D }, dtype: torch.half);
            decimal sizeDecimal = cascade * gridSize3D / 8L;
            this.densityBitfield = torch.zeros(new long[] { Convert.ToInt64(Math.Floor(sizeDecimal)) }, dtype: ScalarType.Byte);
            this.stepCounter = torch.zeros(new long[] { 16, 2 }, dtype: ScalarType.Int32);
            meanDensity = 0f;
            iterDensity = 0;
            meanCount = 0;
            localStep = 0;
            this.register_buffer("aabbTrain", aabbTrain);
            this.register_buffer("aabbInference", aabbInference);
            this.register_buffer("densityGrid", densityGrid);
            this.register_buffer("densityBitfield", densityBitfield);
            this.register_buffer("stepCounter", stepCounter);
            this.mlp = new MLP("MLP", this.bound);
        }

        public (Tensor weightsSum, Tensor depth, Tensor image) runNerfInference(
            Tensor raysOrigin,
            Tensor raysDirection,
            float gammaGrad = 0f,
            uint maxSteps = 1024,
            double tThresshhold = 1e-4
            )
        {
            long[] shape = raysOrigin.shape;
            long[] prefix = new long[shape.Length - 1];
            Array.Copy(shape, prefix, prefix.Length);

            raysOrigin = raysOrigin.contiguous().view(-1, 3);
            raysDirection = raysDirection.contiguous().view(-1, 3);
            long N = raysOrigin.shape[0];
            Device device = raysOrigin.device;
            Tensor aabb = this.aabbInference;


            (Tensor nears, Tensor fars) = RaymarchUtils.nearFarFromAabb(raysOrigin, raysDirection, aabb);
            ScalarType type = torch.float32;

            Tensor weightsSum = torch.zeros(N, type, device);
            Tensor depth = torch.zeros(N, type, device);
            Tensor image = torch.zeros(N, type, device);

            long nAlive = N;
            Tensor raysAlive = torch.arange(nAlive, torch.int32, device);
            Tensor raysTerminated = nears.clone();

            uint step = 0;

            while (step < maxSteps)
            {
                nAlive = raysAlive.shape[0];

                if (nAlive <= 0)
                    break;

                long nStep = Math.Max(Math.Min((N / nAlive), 8L), 1L);
                bool perturbFirst = (step == 0);
                (Tensor xyzs, Tensor dirs, Tensor deltas) = RaymarchUtils.marchRays(nAlive,
                    nStep,
                    raysAlive,
                    raysTerminated,
                    raysOrigin,
                    raysDirection,
                    bound,
                    densityBitfield,
                    Convert.ToUInt32(cascade),
                    Convert.ToUInt32(gridSize),
                    nears,
                    fars,
                    perturbFirst,
                    128,
                    gammaGrad,
                    maxSteps
                    );

                (Tensor sigmas, Tensor rgbs) = this.mlp.forward(xyzs, dirs);
                sigmas = densityScale * sigmas;

                VolumeRendering.compositeRays(nAlive, nStep, raysAlive, raysTerminated, sigmas, rgbs, deltas, weightsSum, depth, image, tThresshhold);
                step += Convert.ToUInt32(nStep);
            }
            return (weightsSum, depth, image);
        }
        

        public (Tensor weightsSum, Tensor depth, Tensor image) runNerfTrain(
            VolumeRendering.CompositeRaysTrain compositeRaysTrain,
            Tensor raysOrigin,
            Tensor raysDirection,
            float backgroundColor,
            float gammaGrad = 0f,
            bool perturb = false,
            bool forceAllRays = false,
            uint maxSteps = 1024,
            double tThresshhold = 1e-4
            )
        {
            long[] shape = raysOrigin.shape;
            long[] prefix = new long[shape.Length - 1];
            Array.Copy(shape, prefix, prefix.Length);

            raysOrigin = raysOrigin.contiguous().view(-1, 3);
            raysDirection = raysDirection.contiguous().view(-1, 3);
            long N = raysOrigin.shape[0];
            Device device = raysOrigin.device;
            Tensor aabb = this.aabbTrain;

            (Tensor nears, Tensor fars) = RaymarchUtils.nearFarFromAabb(raysOrigin, raysDirection, aabb);

            //Console.WriteLine("raysO");
            //Utils.printMean(raysOrigin);
            //Console.WriteLine("raysDir");
            //Utils.printMean(raysDirection);

            //Console.WriteLine("stepCounter: " + stepCounter.dtype);
            Tensor counter = stepCounter[localStep % 16];
            counter.zero_();
            localStep += 1;

            (Tensor xyzs, Tensor dirs, Tensor deltas, Tensor rays) = RaymarchUtils.marchRaysTrain(raysOrigin,
                raysDirection,
                bound,
                densityBitfield,
                Convert.ToUInt32(cascade),
                Convert.ToUInt32(gridSize),
                nears,
                fars,
                counter,
                meanCount,
                perturb,
                128,
                forceAllRays,
                gammaGrad,
                maxSteps
                );
            /*
            Console.WriteLine("xyzs");
            Utils.printMean(xyzs);
            Console.WriteLine("deltas");
            Utils.printMean(deltas);
            Console.WriteLine("dirs");
            Utils.printMean(dirs);
            Console.WriteLine("rays");
            Utils.printMean(rays);
            */
            (Tensor sigmas, Tensor rgbs) = this.mlp.forward(xyzs, dirs);

            sigmas = densityScale * sigmas;


            (Tensor weightsSum, Tensor depth, Tensor image) = compositeRaysTrain.Forward(sigmas, rgbs, deltas, rays, tThresshhold);
            long lastDim = image.dim() - 1L;
            image = image + (1 - weightsSum).unsqueeze(lastDim) * backgroundColor;
            depth = torch.clamp(depth - nears, min: 0f) / (fars - nears);

            shape[prefix.Length] = 3;
            image = image.view(shape);
            depth = depth.view(prefix);
            return (weightsSum, depth, image);
        }
        public void markUntrainedGrid(Tensor poses, Tensor intrinsics, long split = 32)
        {
            using (torch.no_grad())
            {
                using (var d = torch.NewDisposeScope())
                {
                    long BATCH = poses.shape[0];
                    Tensor focalX = intrinsics[0];
                    Tensor focalY = intrinsics[1];
                    Tensor centerX = intrinsics[2];
                    Tensor centerY = intrinsics[3];

                    Tensor[] X = torch.arange(this.gridSize, dtype: ScalarType.Int32, device: densityBitfield.device).split(split);
                    Tensor[] Y = torch.arange(this.gridSize, dtype: ScalarType.Int32, device: densityBitfield.device).split(split);
                    Tensor[] Z = torch.arange(this.gridSize, dtype: ScalarType.Int32, device: densityBitfield.device).split(split);
                    Tensor count = torch.zeros_like(densityGrid);
                    poses = poses.to(count.device);
                    foreach (Tensor x in X)
                    {
                        foreach (Tensor y in Y)
                        {
                            foreach (Tensor z in Z)
                            {
                                Tensor[] grid = torch.meshgrid(new List<Tensor>() { x, y, z });
                                Tensor coords = torch.cat(new List<Tensor>() { grid[0].reshape(-1, 1), grid[1].reshape(-1, 1), grid[2].reshape(-1, 1) }, dim: -1);
                                Tensor indices = torch.LongTensor(RaymarchUtils.morton3D(coords));
                                Tensor xyzsWorld = (2 * torch.FloatTensor(coords) / (gridSize - 1) - 1).unsqueeze(0).to_type(torch.half);


                                //cascading
                                for (int cas = 0; cas < Convert.ToInt32(cascade); cas++)
                                {
                                    float currentBound = Convert.ToSingle(Math.Min(Math.Pow(2, (float)cas), this.bound));
                                    float halfGridSize = currentBound / this.gridSize;
                                    Tensor cascadeXyzsWorld = (xyzsWorld * (currentBound - halfGridSize)).to_type(torch.half);
                                    cascadeXyzsWorld += (torch.rand_like(cascadeXyzsWorld) * 2 - 1) * halfGridSize;

                                    long head = 0;

                                    while (head < BATCH)
                                    {
                                        long tail = Math.Min(head + split, BATCH);
                                        Tensor camXYZS = cascadeXyzsWorld - poses.slice(0, head, tail, 1).slice(1, 0, 3, 1).select(2, 3).unsqueeze(1);

                                        camXYZS = camXYZS.matmul(poses.slice(0, head, tail, 1).slice(1, 0, 3, 1).slice(2, 0, 3, 1));
                                        Tensor maskZ = camXYZS.select(2, 2) > 0;
                                        Tensor maskX = torch.abs(camXYZS.select(2, 0) < centerX / focalX * camXYZS.select(2, 2) + halfGridSize * 2);
                                        Tensor maskY = torch.abs(camXYZS.select(2, 1) < centerY / focalY * camXYZS.select(2, 2) + halfGridSize * 2);
                                        Tensor mask = (maskZ & maskX & maskY).sum(0).reshape(-1);
                                        count[cas, indices] += mask;
                                        head += split;
                                        maskX.Dispose();
                                        maskZ.Dispose();
                                        maskY.Dispose();
                                        mask.Dispose();
                                        camXYZS.Dispose();
                                    }
                                    cascadeXyzsWorld.Dispose();
                                }
                                //Console.WriteLine(d.DisposablesCount);
                                grid[1].Dispose();
                                grid[2].Dispose();
                                coords.Dispose();
                                indices.Dispose();
                                xyzsWorld.Dispose();
                                //Console.WriteLine(d.DisposablesCount);
                            }
                        }

                        densityGrid[count == 0] = -1;
                    }
                    d.DisposeEverythingBut(count);
                }
            }

        }
        public void resetExtraState()
        {
            meanDensity = 0f;
            iterDensity = 0;
            meanCount = 0;
            localStep = 0;
        }
        public void updateExtraState(float decay = 0.95f, long split = 64)
        {
            using (torch.no_grad())
            {
                Tensor temporaryGrid = torch.ones_like(densityGrid);
                if (this.iterDensity < 16)
                {

                    Tensor[] X = torch.arange(this.gridSize, dtype: ScalarType.Int32, device: densityBitfield.device).split(split);
                    Tensor[] Y = torch.arange(this.gridSize, dtype: ScalarType.Int32, device: densityBitfield.device).split(split);
                    Tensor[] Z = torch.arange(this.gridSize, dtype: ScalarType.Int32, device: densityBitfield.device).split(split);

                    foreach (Tensor x in X)
                    {
                        foreach (Tensor y in Y)
                        {
                            foreach (Tensor z in Z)
                            {
                                //construct points
                                Tensor[] grid = torch.meshgrid(new List<Tensor>() { x, y, z });
                                Tensor xx = grid[0].reshape(-1,1);
                                Tensor yy = grid[1].reshape(-1,1);
                                Tensor zz = grid[2].reshape(-1, 1);
                                Tensor coords = torch.cat(new List<Tensor>() { xx, yy, zz }, dim: -1);
                                Tensor indices = torch.LongTensor(RaymarchUtils.morton3D(coords));
                                Tensor xyzs = 2 * torch.FloatTensor(coords).to_type(torch.half) / (gridSize - 1) - 1;

                                //cascading
                                for (int cas = 0; cas < Convert.ToInt32(cascade); cas++)
                                {
                                    float currentBound = Convert.ToSingle(Math.Min(Math.Pow(2, cas), this.bound));
                                    float halfGridSize = currentBound / this.gridSize;
                                    Tensor cascadeXyzs = xyzs * (currentBound - halfGridSize);
                                    cascadeXyzs += (torch.rand_like(cascadeXyzs) * 2 - 1) * halfGridSize;
                                    Tensor sigmas = mlp.density(cascadeXyzs).sigmas.reshape(-1L).detach();
                                    sigmas *= densityScale;
                                    temporaryGrid[cas, indices] = sigmas;

                                }
                            }
                        }
                    }
                }
                else
                {

                    long N = (long)Math.Floor(Math.Pow(gridSize, 3) / 4);
                    for (int cas = 0; cas < Convert.ToInt32(cascade); cas++)
                    {
                        Tensor coords = torch.randint(0L, gridSize, new long[] { N, 3 }, dtype: torch.@long, device: densityBitfield.device);
                        Tensor indices = torch.LongTensor(RaymarchUtils.morton3D(coords));
                        Tensor occupiedIndices = torch.nonzero(densityGrid[cas] > 0).squeeze(-1L);
                        Tensor randomMask = torch.randint(0, occupiedIndices.shape[0], N, dtype: torch.@long, device: densityBitfield.device);
                        occupiedIndices = occupiedIndices[randomMask];
                        Tensor occupiedCoords = RaymarchUtils.morton3DInvert(occupiedIndices);
                        indices = torch.cat(new Tensor[] { indices, occupiedIndices });
                        coords = torch.cat(new Tensor[] { coords, occupiedCoords });
                        Tensor xyzs = 2 * torch.FloatTensor(coords).to_type(torch.half) / (gridSize - 1) - 1;
                        float currentBound = Convert.ToSingle(Math.Min(Math.Pow(2, cas), this.bound));
                        float halfGridSize = currentBound / this.gridSize;
                        Tensor cascadeXyzs = xyzs * (currentBound - halfGridSize);
                        cascadeXyzs += (torch.rand_like(cascadeXyzs) * 2 - 1) * halfGridSize;
                        Tensor sigmas = mlp.density(cascadeXyzs).sigmas.reshape(-1L).detach();
                        sigmas *= densityScale;
                        temporaryGrid[cas, indices] = sigmas;
                    }

                }
                Tensor validMask = (densityGrid >= 0) & (temporaryGrid >= 0);
                densityGrid[validMask] = torch.maximum(densityGrid[validMask] * decay, temporaryGrid[validMask]);
                meanDensity = torch.mean(densityGrid.clamp(min: 0)).item<float>();
                iterDensity += 1;

                densityThreshhold = Math.Min(meanDensity, densityThreshhold);
                densityBitfield = RaymarchUtils.packbits(densityGrid, densityThreshhold, densityBitfield);
                int totalStep = Math.Min(16, localStep);
                if (totalStep > 0)
                {
                    int stepCounterItem = stepCounter.slice(0, 0, totalStep, 1).select(1, 0).sum().item<int>();
                    meanCount = stepCounterItem / totalStep;
                }

                localStep = 0;
            }

        }
    }
}
