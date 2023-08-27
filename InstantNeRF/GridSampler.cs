/*
  Copyright 2022 XRNerf Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

// custom implementation in C# based on https://github.com/openxrlab/xrnerf/blob/main/xrnerf/models/samplers/ngp_grid_sampler.py

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
        private int gridsize3D;
        private int sizeIncludingMips;
        public int densityActivation;
        public int rgbActivation;
        private int targetBatchSize;
        private int iteration;
        private int emaStep;
        public Tensor temporaryGrid;
        public Tensor densityBitfield;
        public Tensor densityMean;
        public Tensor densityGrid;
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
            this.densityActivation = densityActivation;
            this.rgbActivation = rgbActivation;
            this.targetBatchSize = targetBatchSize;
            this.nElementsCoords = nRays * MAX_STEPS;
            this.gridsize3D = GRIDSIZE * GRIDSIZE * GRIDSIZE;
            this.nElementsDensity = gridsize3D * CASCADES;
            this.sizeIncludingMips = gridsize3D * CASCADES / 8;

            // Density Grid and Bitfield
            this.temporaryGrid = torch.zeros(nElementsDensity, torch.float32, CUDA).contiguous();

            double result = (double)gridsize3D / N_THREADS_LINEAR;
            int roundedUpResult = (int)Math.Ceiling(result);
            this.densityMean = torch.zeros(roundedUpResult, torch.float32, CUDA).contiguous();
            this.densityBitfield = torch.zeros(sizeIncludingMips, torch.uint8, CUDA).contiguous();
            this.densityGrid = torch.zeros(nElementsDensity, torch.float32, CUDA).contiguous();
            this.register_buffer("densityBitfield", densityBitfield);
            this.register_buffer("densityMean", densityMean);
            this.register_buffer("densityGrid", densityGrid);
            this.register_buffer("temoraryGrid", temporaryGrid);

            this.iteration = 0;
            this.emaStep = 0;
            long nImages = dataProvider.images.size(0);
            Tensor focalLengths = dataProvider.focals;
            Tensor metaData = torch.from_array(new float[] { 0f, 0f, 0f, 0f, 0.5f, 0.5f, focalLengths[0].item<float>(), focalLengths[1].item<float>(), 0f, 0f, 0f });
            focalLengths = focalLengths.unsqueeze(0).repeat(nImages, 1);
            this.dataInfo = new DataInfo(dataProvider.width,
                dataProvider.height,
                dataProvider.poses.to(CUDA).contiguous(),
                focalLengths.to(CUDA).contiguous(),
                dataProvider.aabbScale,
                dataProvider.aabbMin,
                dataProvider.aabbMax,
                metaData.to(CUDA).contiguous());
        }

        public DataInfo getData()
        {
            if (this.dataInfo == null)
            {
                throw new Exception("no datainfo available");
            }
            else { return this.dataInfo; }
        }

        public void updateDensityGrid(MLP mlp)
        {
            int nCascades = CASCADES + 1;
            int M = gridsize3D * nCascades;

            if (this.iteration < 256)
            {
                updateDensityGridBasedOnIteration(M, 0, mlp);
            }
            else
            {
                updateDensityGridBasedOnIteration(M / 4, M / 4, mlp);
            }
        }

        private void updateDensityGridBasedOnIteration(int nUniformSamples, int nNonUniformSamples, MLP mlp)
        {
            if (this.dataInfo != null)
            {
                int totalSamples = nUniformSamples + nNonUniformSamples;

                if (this.iteration == 0)
                {
                    RaymarchApi.markUntrainedGrid(this.densityGrid, dataInfo.focalLengths, dataInfo.transforms, nElementsDensity, dataInfo.nImages, dataInfo.width, dataInfo.height);
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
                RaymarchApi.splatGridSamplesNerfMaxNearestNeighbour(density, indices, PADDED_OUTPUT_WIDTH, totalSamples, temporaryGrid);

                RaymarchApi.sampleDensityGridEma(temporaryGrid, nElementsDensity, EMA_DECAY, densityGrid);

                this.densityGrid.detach_();
                this.emaStep++;

                RaymarchApi.updateBitfield(densityGrid, densityMean, densityBitfield);
            }


        }

        public Dictionary<string, Tensor> Sample(Dictionary<string, Tensor> data, MLP mlp)
        {
            if (dataInfo == null) { throw new Exception("no datainfo provided."); }

            Tensor transforms;
            Tensor imageIndices;
            Tensor raysOrigin = data["raysOrigin"].contiguous();
            Tensor raysDirection = data["raysDirection"].contiguous();
           


            if (this.training)
            {
                if (this.iteration % this.updateGridFrequency == 0)
                {
                    updateDensityGrid(mlp);
                }
                data["bgColor"] = data["bgColor"].to(torch.float32).contiguous();
                imageIndices = data["imageIndices"].to(torch.int32).contiguous();
                transforms = dataInfo.transforms;
            }
            else
            {
                transforms = data["pose"].unsqueeze(0);
                imageIndices = torch.zeros(raysOrigin.size(0));
            }

            Tensor[] sampledResults = RaymarchApi.sampleRays(
                raysOrigin,
                raysDirection,
                densityBitfield,
                dataInfo.metadata,
                imageIndices,
                transforms,
                dataInfo.aabbMin,
                dataInfo.aabbMax,
                NEAR,
                CONE_ANGLE_CONST,
                nElementsCoords
                );
            Tensor coords = sampledResults[0];
            Tensor positions = coords.slice(-1, 0, 3, 1).detach();
            Tensor directions = coords.slice(-1, 4, coords.size(-1), 1).detach();
            Tensor rayIndices = sampledResults[1];
            Tensor rayNumsteps = sampledResults[2];
            Tensor rayCounter = sampledResults[3];

            if (!this.training)
            {
                data.Add("coords", coords.detach());
                data.Add("rayNumsteps", rayNumsteps.detach());
                data.Add("positions", positions);
                data.Add("directions", directions);
                return data;
            }
            Tensor sigmaOutput = mlp.density(positions).detach().to(torch.float32);
            Tensor[] compactedResults = RaymarchApi.compactedCoords(sigmaOutput, coords, rayNumsteps, this.targetBatchSize, densityActivation, dataInfo.aabbMin, dataInfo.aabbMax);

            Tensor compactedCoords = compactedResults[0];
            Tensor rayNumstepsCompacted = compactedResults[1];
            Tensor rayCounterCompacted = compactedResults[2];
            Tensor measuredBatchSize = torch.zeros(1, torch.int32, CUDA).contiguous();
            measuredBatchSize += rayCounterCompacted;
            if (this.training)
            {
                updateRayBatchsize(measuredBatchSize);
            }
            data.Add("coords", compactedCoords.detach());
            data.Add("rayNumsteps", rayNumsteps.detach());
            data.Add("rayNumstepsCompacted", rayNumstepsCompacted.detach());
            data.Add("positions", compactedCoords.slice(-1, 0, 3, 1));
            data.Add("directions", compactedCoords.slice(-1, 4, compactedCoords.size(-1), 1));

            this.iteration++;
            return data;
        }
        private void updateRayBatchsize(Tensor measured)
        {
            if (iteration % updateGridFrequency == (updateGridFrequency - 1))
            {
                int measuredBatchSize = Math.Max(measured.item<int>() / 16, 1);
                int measuredRaysPerBatch = nRays * targetBatchSize / measuredBatchSize;

                double result = (double)measuredRaysPerBatch / 128;
                int roundedUpResult = (int)Math.Ceiling(result);

                this.nRays = Math.Min(roundedUpResult * 128, targetBatchSize);
            }
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