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
        public TcnnNerfModule(string name, uint posInputDims, uint dirInputDims, uint extraInputDims, uint dirOffset, string posEncoding, string dirEncoding, string sigmaNet, string colorNet, ulong seed = 1234) : base(name)
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
                this.lossScale = 128.0f;
            }
            else
            {
                this.lossScale = 1.0f;
            }
        }
        public override Tensor forward(Tensor x)
        {
            if (!x.is_cuda)
            {
                Console.WriteLine("input must be a CUDA tensor");
                x = x.cuda();
            }
            long batchSize = x.shape[0];
            long batchSizeGranularity = Convert.ToInt64(TcnnWrapper.batchSizeGranularity());
            long paddedBatchSize = (batchSize + batchSizeGranularity - 1) / batchSizeGranularity * batchSizeGranularity;
            Tensor xPadded = (batchSize == paddedBatchSize) ? x : torch.nn.functional.pad(x, new long[4] { 0L, 0L, 0L, paddedBatchSize - batchSize });

            this.gradFnc = new AutogradFunctions.ModuleFunction(this.nativeTcnnMLP, this.lossScale);

            Tensor output = this.gradFnc.Forward(
                xPadded.to_type(torch.float32).contiguous(),
                param.contiguous().nan_to_num());
            output = output.slice(0L, 0L, batchSize, 1L);
            output = output.slice(1L, 0L, outputDims, 1L);
            output = FloatTensor(output);
            return output;
        }
        public Tensor backward(float gradScale)
        {
            if (this.gradFnc != null)
            {
               return this.gradFnc.Backward(gradScale);
            }
            else
            {
                throw new Exception("must run forward pass before backward pass!");
            }
        }
        public Tensor density(Tensor x)
        {
            long batchSize = x.shape[0];
            long batchSizeGranularity = Convert.ToInt64(TcnnWrapper.batchSizeGranularity());
            long paddedBatchSize = (batchSize + batchSizeGranularity - 1) / batchSizeGranularity * batchSizeGranularity;
            Tensor xPadded = (batchSize == paddedBatchSize) ? x : torch.nn.functional.pad(x, new long[4] { 0L, 0L, 0L, paddedBatchSize - batchSize });
            Tensor output = this.nativeTcnnMLP.density(
                xPadded.to_type(torch.float32).contiguous(),
                param.contiguous().nan_to_num());

            output = output.slice(0L, 0L, batchSize, 1L);
            output = output.slice(1L, 0L, outputDimsDensity, 1L);
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