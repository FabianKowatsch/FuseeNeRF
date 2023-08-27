/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// custom implementation in C# based on https://github.com/NVlabs/tiny-cuda-nn/blob/master/bindings/torch/tinycudann/modules.py
using static TorchSharp.torch;
using TorchSharp;
using static InstantNeRF.TcnnWrapper;
using TorchSharp.Modules;

namespace InstantNeRF
{
    public class TcnnNerfModule : nn.Module<Tensor, Tensor>
    {
        private NerfModuleWrapper nativeTcnnMLP;
        private ulong seed;
        private ScalarType dtype;
        private Parameter param;
        private Parameter paramFP;
        public uint outputDims;
        public uint outputDimsDensity;
        private float lossScale;
        private AutogradFunctions.ModuleFunction? gradFnc;
        public TcnnNerfModule(string name, uint posInputDims, uint dirInputDims, uint extraInputDims, uint dirOffset, string posEncoding, string dirEncoding, string sigmaNet, string colorNet, ulong seed = 1337) : base(name)
        {
            this.nativeTcnnMLP = new NerfModuleWrapper(moduleRef: createNerfNetwork(posInputDims, dirInputDims, extraInputDims, dirOffset, posEncoding, dirEncoding, sigmaNet, colorNet));
            this.seed = seed;

            this.dtype = nativeTcnnMLP.paramPrecision();
            this.outputDims = nativeTcnnMLP.nOutputDims();
            this.outputDimsDensity = nativeTcnnMLP.nOutputDimsDensity();
            Tensor paramsFullPrecision = nativeTcnnMLP.initialParams(this.seed);
            Tensor paramsHalfPrecision = paramsFullPrecision.to_type(this.dtype, copy: true);
            paramFP = torch.nn.Parameter(paramsFullPrecision, requires_grad: true);
            param = torch.nn.Parameter(paramsHalfPrecision, requires_grad: true);
            this.register_parameter("param", param);
            this.register_parameter("paramFP", paramFP);
            if (this.dtype == torch.half)
            {
                //this.lossScale = 128.0f;
                this.lossScale = 1.0f;
            }
            else
            {
                this.lossScale = 1.0f;
            }
        }
        public override Tensor forward(Tensor input)
        {
            if (!input.is_cuda)
            {
                Console.WriteLine("input must be a CUDA tensor");
                input = input.cuda();
            }
            long batchSize = input.shape[0];
            long batchSizeGranularity = Convert.ToInt64(TcnnWrapper.batchSizeGranularity());
            long paddedBatchSize = (batchSize + batchSizeGranularity - 1) / batchSizeGranularity * batchSizeGranularity;
            Tensor inputPadded = (batchSize == paddedBatchSize) ? input : torch.nn.functional.pad(input, new long[4] { 0L, 0L, 0L, paddedBatchSize - batchSize });

            this.gradFnc = new AutogradFunctions.ModuleFunction(this.nativeTcnnMLP, this.lossScale);

            Tensor output = this.gradFnc.Forward(inputPadded.to_type(torch.float32).contiguous(), param.contiguous().nan_to_num());
            output = output.slice(0L, 0L, batchSize, 1L);
            output = output.slice(1L, 0L, outputDims, 1L);
            output = FloatTensor(output);

            return output;
        }
        public Tensor backward(float gradScale)
        {
            if (this.gradFnc != null)
            {
               return this.gradFnc.Backward(gradScale, param.contiguous().nan_to_num());
            }
            else
            {
                throw new Exception("must run forward pass before backward pass!");
            }
        }
        public Tensor density(Tensor input)
        {
            long batchSize = input.shape[0];
            long batchSizeGranularity = Convert.ToInt64(TcnnWrapper.batchSizeGranularity());
            long paddedBatchSize = (batchSize + batchSizeGranularity - 1) / batchSizeGranularity * batchSizeGranularity;
            Tensor inputPadded = (batchSize == paddedBatchSize) ? input : torch.nn.functional.pad(input, new long[4] { 0L, 0L, 0L, paddedBatchSize - batchSize });

            Tensor output = this.nativeTcnnMLP.density(inputPadded.to_type(torch.float32).contiguous(), param.contiguous().nan_to_num());
            output = output.slice(0L, 0L, batchSize, 1L);
            output = output.slice(1L, 0L, outputDimsDensity, 1L);
            output = FloatTensor(output);

            return output;
        }

        public Tensor inference(Tensor input)
        {
            long batchSize = input.shape[0];
            long batchSizeGranularity = Convert.ToInt64(TcnnWrapper.batchSizeGranularity());
            long paddedBatchSize = (batchSize + batchSizeGranularity - 1) / batchSizeGranularity * batchSizeGranularity;
            Tensor inputPadded = (batchSize == paddedBatchSize) ? input : torch.nn.functional.pad(input, new long[4] { 0L, 0L, 0L, paddedBatchSize - batchSize });

            Tensor output = this.nativeTcnnMLP.inference(inputPadded.to_type(torch.float32).contiguous(), param.contiguous().nan_to_num());
            output = output.slice(0L, 0L, batchSize, 1L);
            output = output.slice(1L, 0L, outputDims, 1L);
            output = FloatTensor(output);

            return output;
        }

        public IntPtr getHandle()
        {
            return nativeTcnnMLP.getHandle();
        }
        public (Parameter param, Parameter paramFP) getParameters()
        {
            return (param, paramFP);
        }
    }

}