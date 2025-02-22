﻿using InstantNeRF;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Xml.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


namespace InstantNeRF
{
    public static class RaymarchApi
    {
        [DllImport("RaymarchApi.dll")]
        private static extern void generateGridSamplesApi(IntPtr density_grid, int density_grid_ema_step, int n_elements, int max_cascade,
    float thresh, float aabb0, float aabb1,
    IntPtr density_grid_positions_uniform,
    IntPtr density_grid_indices_uniform);

        [DllImport("RaymarchApi.dll")]
        private static extern void markUntrainedGridApi(IntPtr focal_lengths,
    IntPtr transforms,
    int n_elements, int n_images,
    int img_resolution0, int img_resolution1,
    IntPtr density_grid);

        [DllImport("RaymarchApi.dll")]
        private static extern void splatGridSamplesNerfMaxNearestNeighbourApi(IntPtr mlp_out,
    IntPtr density_grid_indices, int padded_output_width,
    int n_density_grid_samples, IntPtr density_grid_tmp);

        [DllImport("RaymarchApi.dll")]
        private static extern void emaGridSamplesNerfApi(IntPtr density_grid_tmp, int n_elements,
    float decay, IntPtr density_grid);

        [DllImport("RaymarchApi.dll")]
        private static extern void updateBitfieldApi(IntPtr density_grid,
    IntPtr density_grid_mean, IntPtr density_grid_bitfield);

        [DllImport("RaymarchApi.dll")]
        private static extern void sampleRaysApi(
    IntPtr rays_o,
    IntPtr rays_d,
    IntPtr density_grid_bitfield,
    IntPtr metadata,
    IntPtr imgs_id,
    IntPtr xforms,
    float aabb0,
    float aabb1,
    float near_distance,
    float cone_angle_constant,
    IntPtr coords_out,
    IntPtr rays_index,
    IntPtr rays_numsteps,
    IntPtr ray_numstep_counter);

        [DllImport("RaymarchApi.dll")]
        private static extern void compactedCoordsApi(
            IntPtr network_output,
    IntPtr coords_in,
    IntPtr rays_numsteps,
    int density_activation_i,
    float aabb0,
    float aabb1,
    IntPtr coords_out,
    IntPtr rays_numsteps_compacted,
    IntPtr compacted_rays_counter,
    IntPtr compacted_numstep_counter);



        public static (Tensor densityGridPositions, Tensor densityGridIndices) generateGridSamples(Tensor densityGrid, int emaStep, int nElements, int maxCascade, float densityThreshold, float aabb0, float aabb1)
        {
            Device device = densityGrid.device;
            Tensor densityGridPositions = torch.empty(new long[] { nElements, 3 }, torch.float32, device);
            Tensor densityGridIndices = torch.empty(new long[] { nElements }, torch.int32, device);
            generateGridSamplesApi(densityGrid.Handle, emaStep, nElements, maxCascade, densityThreshold, aabb0, aabb1, densityGridPositions.Handle, densityGridIndices.Handle);

            return (densityGridPositions, densityGridIndices);
        }

        public static void markUntrainedGrid(Tensor densityGrid, Tensor focalLengths, Tensor transforms, int nElements, int nImages, int width, int height)
        {
            Device device = focalLengths.device;
            densityGrid.to(device);
            markUntrainedGridApi(focalLengths.Handle, transforms.Handle, nElements, nImages, width, height, densityGrid.Handle);
        }

        public static void splatGridSamplesNerfMaxNearestNeighbour(Tensor densityOutput, Tensor indices, int paddedOutputWidth, int nSamples, Tensor temporaryGrid)
        {
            temporaryGrid.to(densityOutput.device);
            temporaryGrid.zero_();
            splatGridSamplesNerfMaxNearestNeighbourApi(densityOutput.Handle, indices.Handle, paddedOutputWidth, nSamples, temporaryGrid.Handle);
        }

        public static void sampleDensityGridEma(Tensor temporaryGrid, int nElements, float decay, Tensor densityGrid)
        {
            emaGridSamplesNerfApi(temporaryGrid.Handle, nElements, decay, densityGrid.Handle);
        }

        public static void updateBitfield(Tensor densityGrid, Tensor densityMean, Tensor densityBitfield)
        {
            densityBitfield.to(densityGrid.device);
            densityMean.to(densityGrid.device);
            densityMean.zero_();
            updateBitfieldApi(densityGrid.Handle, densityMean.Handle, densityBitfield.Handle);
        }

        public static Tensor[] sampleRays(
            Tensor raysOrigin,
            Tensor raysDirection,
            Tensor densityBitfield,
            Tensor metadata,
            Tensor imagesIndices,
            Tensor transforms,
            float aabb0,
            float aabb1,
            float nearDistance,
            float coneAngleConstant,
            int nElements)
        {
            Device device = transforms.device;

            long nRaysPerBatch = raysOrigin.size(0);
            Tensor coordsOut = torch.zeros(new long[] { nElements, 7 }, torch.float32, device);
            Tensor rayIndices = torch.zeros(new long[] { nRaysPerBatch, 1 }, torch.int32, device);
            Tensor rayNumsteps = torch.zeros(new long[] { nRaysPerBatch, 2 }, torch.int32, device);
            Tensor rayCounter = torch.zeros(new long[] { 2 }, torch.int32, device);

            sampleRaysApi(raysOrigin.Handle,
                raysDirection.Handle,
                densityBitfield.Handle,
                metadata.Handle,
                imagesIndices.Handle,
                transforms.Handle,
                aabb0,
                aabb1,
                nearDistance,
                coneAngleConstant,
                coordsOut.Handle,
                rayIndices.Handle,
                rayNumsteps.Handle,
                rayCounter.Handle);

            coordsOut.detach_();
            rayIndices.detach_();
            rayNumsteps.detach_();
            rayCounter.detach_();



            int samples = rayCounter[1].item<int>();
            coordsOut = coordsOut.slice(0, 0, Convert.ToInt64(samples), 1);

            return new Tensor[] { coordsOut, rayIndices, rayNumsteps, rayCounter };

        }

        public static Tensor[] compactedCoords(
            Tensor sigmaOutput,
            Tensor positionsIn,
            Tensor rayNumsteps,
            long nElementsCompacted,
            int densityActivation,
            float aabb0,
            float aabb1
            )
        {
            Device device = sigmaOutput.device;
            Tensor positionsOut = torch.zeros(new long[] { nElementsCompacted, 7 }, torch.float32, device);
            Tensor rayNumstepsCompacted = torch.zeros_like(rayNumsteps, torch.int32, device);
            Tensor rayCounterCompacted = torch.zeros(new long[] { 1 }, torch.int32, device);
            Tensor numstepsCounterCompacted = torch.zeros(new long[] { 1 }, torch.int32, device);

            compactedCoordsApi(sigmaOutput.Handle,
                positionsIn.Handle,
                rayNumsteps.Handle,
                densityActivation,
                aabb0,
                aabb1,
                positionsOut.Handle,
                rayNumstepsCompacted.Handle,
                rayCounterCompacted.Handle,
                numstepsCounterCompacted.Handle);

            return new Tensor[] { positionsOut, rayNumstepsCompacted, rayCounterCompacted };
        }




    }
}