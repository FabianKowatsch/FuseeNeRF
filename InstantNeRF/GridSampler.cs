using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace InstantNeRF
{
    public class GridSampler : nn.Module
    {
        private int updateGridFrequency;
        private int nRays;
        private int nElementsCoords;
        private int nElementsDensity;
        private int nElementsBitfield;
        private int sizeIncludingMips;
        private float nearDistance;
        private float[] bgColor;
        public int densityActivation;
        public int rgbActivation;
        private int targetBatchSize;
        private Tensor measuredBatchSize;
        private int iteration;
        private int emaStep;
        private Tensor temporaryGrid;
        public Tensor densityBitfield;
        public Tensor densityMean;
        private Tensor densityGrid;
        public DataInfo? dataInfo;
        private readonly int UPDATE_BLOCK_SIZE = 5000000;
        private readonly float CONE_ANGLE_CONST = 0.00390625f;
        private readonly int PADDED_OUTPUT_WIDTH = 1;
        private readonly int CASCADES = 4;
        private readonly int GRIDSIZE = 64;
        private readonly float NEAR = 0.05f;
        private readonly float EMA_DECAY = 0.95f;
        private readonly int MAX_STEPS = 1024;
        private readonly float MIN_OPTICAL_THICKNESS = 0.01f;
        private readonly int N_THREADS_LINEAR = 128;
        public GridSampler(DataProvider dataProvider, int updateGridFrequency = 16, float nearDistance = 0.2f, int densityActivation = 2, int rgbActivation = 3, int targetBatchSize = 1 << 18) : base("GridSampler")
        {
            this.updateGridFrequency = updateGridFrequency;
            this.nRays = dataProvider.numRays;
            this.nearDistance = nearDistance;
            this.densityActivation = densityActivation;
            this.rgbActivation = rgbActivation;
            this.targetBatchSize = targetBatchSize;
            this.nElementsCoords = nRays * MAX_STEPS;
            this.nElementsBitfield = GRIDSIZE * GRIDSIZE * GRIDSIZE;
            this.nElementsDensity = nElementsBitfield * CASCADES;
            this.sizeIncludingMips = nElementsBitfield * CASCADES / 8;
            
            // Density Grid and Bitfield
            this.temporaryGrid = torch.zeros(nElementsDensity, torch.float32);

            double result = (double)nElementsBitfield / N_THREADS_LINEAR;
            int roundedUpResult = (int)Math.Ceiling(result);  
            this.densityMean = torch.zeros(roundedUpResult, torch.float32);
            this.densityBitfield = torch.zeros(nElementsBitfield, torch.uint8);
            this.register_buffer("densityBitfield", densityBitfield);
            this.densityGrid = torch.empty(nElementsDensity, torch.float32);

            this.measuredBatchSize = torch.zeros(1, torch.int32);

            this.iteration = 0;
            this.emaStep = 0;
            this.bgColor = dataProvider.bgColor;
            long nImages = dataProvider.images.size(0);
            Tensor focalLengths = dataProvider.focals;
            Tensor metaData = torch.from_array(new float[] { 0f, 0f, 0f, 0f, 0.5f, 0.5f, focalLengths[0].item<float>(), focalLengths[1].item<float>(), 0f, 0f, 0f });
            focalLengths = focalLengths.unsqueeze(0).repeat(nImages, 1);
            this.dataInfo = new DataInfo(dataProvider.width, dataProvider.height, dataProvider.poses, focalLengths, dataProvider.aabbScale, dataProvider.aabbMin, dataProvider.aabbMax, metaData);
        }

        public DataInfo getData() 
        {
            if(this.dataInfo == null)
            {
                throw new Exception("no datainfo available");
            }
            else { return this.dataInfo; }
        }

        public void updateDensityGrid(MLP mlp)
        {
            int nCascades = CASCADES + 1;
            int M = nElementsBitfield * nCascades;

            if(this.iteration < 256)
            {
                updateDensityGridBasedOnIteration(M, 0, mlp);
            }
            else
            {
                updateDensityGridBasedOnIteration(M/4, M/4, mlp);
            }
        }

        private void updateDensityGridBasedOnIteration(int nUniformSamples, int nNonUniformSamples, MLP mlp)
        {
            if(this.dataInfo != null)
            {
                int totalSamples = nUniformSamples + nNonUniformSamples;

                if (this.iteration == 0)
                {
                    this.densityGrid = RaymarchApi.markUntrainedGrid(dataInfo.focalLengths, dataInfo.transforms, nElementsDensity, dataInfo.nImages, dataInfo.width, dataInfo.height);
                }
                (Tensor positionsUniform, Tensor indicesUniform) = RaymarchApi.generateGridSamples(
                    densityGrid,
                    emaStep,
                    nUniformSamples,
                    dataInfo.maxCascades,
                    -MIN_OPTICAL_THICKNESS,
                    dataInfo.aabbMin,
                    dataInfo.aabbMax);
                (Tensor positionsNonUniform, Tensor indicesNonUniform) = RaymarchApi.generateGridSamples(
                    densityGrid,
                    emaStep,
                    nNonUniformSamples,
                    dataInfo.maxCascades,
                    MIN_OPTICAL_THICKNESS,
                    dataInfo.aabbMin,
                    dataInfo.aabbMax);

                Tensor positions = torch.cat(new List<Tensor>() { positionsUniform, positionsNonUniform });
                Tensor indices = torch.cat(new List<Tensor>() { indicesUniform, indicesNonUniform });

                Tensor density;
                using (torch.no_grad())
                {
                    List<Tensor> results = new List<Tensor>();
                    for (int i = 0; i < positions.size(0); i = i + UPDATE_BLOCK_SIZE)
                    {
                        Tensor positionsFlat = positions.slice(0, i, i + UPDATE_BLOCK_SIZE, 1);
                        results.Add(mlp.density(positionsFlat));

                    }
                    density = torch.cat(results, 0);
                }
                this.temporaryGrid = RaymarchApi.splatGridSamplesNerfMaxNearestNeighbour(density, indices, PADDED_OUTPUT_WIDTH, totalSamples, temporaryGrid);

                this.densityGrid = RaymarchApi.sampleDensityGridEma(temporaryGrid, nElementsDensity, EMA_DECAY, densityGrid);

                this.densityGrid.detach_();
                this.emaStep++;

                (this.densityBitfield, this.densityMean) = RaymarchApi.updateBitfield(densityGrid, densityMean, densityBitfield);
            }

            
        }

        public Dictionary<string, Tensor> Sample(Dictionary<string, Tensor> data, MLP mlp)
        {
            if (dataInfo == null) { throw new Exception("no datainfo provided."); }

            prepareData();
            if (this.training)
            {
                if(this.iteration % this.updateGridFrequency == 0)
                {
                    updateDensityGrid(mlp);
                }
            }

            Tensor raysOrigin = data["raysOrigin"].contiguous();
            Tensor raysDirection = data["raysDirection"].contiguous();

            Tensor imageIndices = data["imageIndices"].to(torch.int32).contiguous();
            data["bgColor"] = data["bgColor"].to(torch.float32).contiguous();

            Tensor[] sampledResults = RaymarchApi.sampleRays(
                raysOrigin, 
                raysDirection, 
                densityBitfield, 
                dataInfo.metadata, 
                imageIndices, 
                dataInfo.transforms, 
                dataInfo.aabbMin, 
                dataInfo.aabbMax, 
                NEAR,
                CONE_ANGLE_CONST,
                nElementsCoords
                );
            Tensor coords = sampledResults[0];
            Tensor positions = coords.slice(1, 0, 3, 1).detach();
            Tensor directions = coords.slice(1, 4, coords.size(1), 1).detach();
            Tensor rayIndices = sampledResults[1];
            Tensor rayNumsteps = sampledResults[2];
            Tensor rayCounter = sampledResults[3];

            if (!this.training)
            {
                rayNumsteps.detach_();
                coords.detach_();
                data.Add("positions", positions);
                data.Add("directions", directions);
                return data;
            }
            Dictionary<string, Tensor> dataForDensity = new Dictionary<string, Tensor>
            {
                { "positions", positions },
                { "directions", directions }
            };
            Tensor nerfOutputs = mlp.forward(dataForDensity)["raw"].detach().to(torch.float32);
            Tensor[] compactedResults = RaymarchApi.compactedCoords(nerfOutputs, coords, rayNumsteps, this.targetBatchSize, rgbActivation, densityActivation, dataInfo.aabbMin, dataInfo.aabbMax, this.bgColor);

            Tensor compactedCoords = compactedResults[0];
            Tensor rayNumstepsCompacted = compactedResults[1];
            Tensor rayCounterCompacted = compactedResults[2];
            this.measuredBatchSize += rayCounterCompacted;
            if(this.training)
            {
                updateRayBatchsize();
            }
            data.Add("coords", compactedCoords.detach());
            data.Add("rayNumsteps", rayNumsteps.detach());
            data.Add("rayNumstepsCompacted", rayNumstepsCompacted.detach());
            data.Add("positions", compactedCoords.slice(-1, 0, 3, 1));
            data.Add("directions", compactedCoords.slice(-1, 4, compactedCoords.size(-1), 1));

            this.iteration++;
            return data;
        }
        private void updateRayBatchsize()
        {
            if(iteration % updateGridFrequency == (updateGridFrequency - 1))
            {
                int measuredBatchSize = Math.Max(this.measuredBatchSize.item<int>() / 16, 1);
                int RaysPerBatch = nRays * targetBatchSize / measuredBatchSize;

                double result = (double)nRays / 128;
                int roundedUpResult = (int)Math.Ceiling(result);

                this.nRays = Math.Min(roundedUpResult, targetBatchSize);
                this.measuredBatchSize.zero_();
            }
        }
        private void prepareData()
        {
            if(dataInfo == null) {  return; }
            if(this.temporaryGrid.device != CUDA)
            {
                this.densityBitfield = this.densityBitfield.to(CUDA).contiguous();
                this.densityGrid = this.densityGrid.to(CUDA).contiguous();
                this.temporaryGrid = this.temporaryGrid.to(CUDA).contiguous();
                this.densityMean = this.densityMean.to(CUDA).contiguous();
                this.measuredBatchSize = this.measuredBatchSize.to(CUDA).contiguous();
                dataInfo.focalLengths = dataInfo.focalLengths.to(CUDA).contiguous();
                dataInfo.transforms = dataInfo.transforms.to(CUDA).contiguous();
                dataInfo.metadata = dataInfo.metadata.to(CUDA).contiguous();
            }

        }

        public void printDensity() 
        {
            Utils.printDims(this.densityBitfield, "densityBitfield");
            Utils.printFirstNValues(this.densityBitfield, 3, "densityBitfield");
            Utils.printDims(this.densityGrid, "densityGrid");
            Utils.printFirstNValues(this.densityGrid, 3, "densityGrid");
            Utils.printDims(this.temporaryGrid, "temporaryGrid");
            Utils.printFirstNValues(this.temporaryGrid, 3, "temporaryGrid");
        }
    }

    public class DataInfo
    {
        public int width;
        public int height;
        public Tensor transforms;
        public Tensor focalLengths;
        public float aabbScale;
        public float aabbMin;
        public float aabbMax;
        public Tensor metadata;
        public int nImages;
        public int maxCascades;

        public DataInfo(int width, int height, Tensor transforms, Tensor focalLengths, float aabbScale, float aabbMin, float aabbMax, Tensor metadata)
        {
            this.width = width;
            this.height = height;
            this.transforms = transforms;
            this.focalLengths = focalLengths;
            this.aabbScale = aabbScale;
            this.aabbMin = aabbMin;
            this.aabbMax = aabbMax;
            this.metadata = metadata;
            this.nImages = (int)this.transforms.size(0);
            while ((1 << this.maxCascades) < this.aabbScale)
            {
                this.maxCascades++;
            }
        }
    }
}
