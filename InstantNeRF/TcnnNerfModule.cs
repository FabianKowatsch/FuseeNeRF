using static TorchSharp.torch;
using TorchSharp;
using static InstantNeRF.TcnnWrapper;

namespace InstantNeRF
{
    public class TcnnNerfModule : nn.Module<Tensor, Tensor>
    {
        private NerfModuleWrapper nativeTcnnMLP;
        private ulong seed;
        private ScalarType dtype;
        private TorchSharp.Modules.Parameter param;
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
            Tensor initialParams = nativeTcnnMLP.initialParams(this.seed).to_type(this.dtype);
            param = torch.nn.Parameter(initialParams, requires_grad: true);
            this.register_parameter("params", param);
            if (this.dtype == torch.half)
            {
                this.lossScale = 128.0f;
            }
            else
            {
                this.lossScale = 1.0f;
            }
            Console.WriteLine(nativeTcnnMLP.hyperparams());
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
        public void backward(float gradScale)
        {
            if (this.gradFnc != null)
            {
                this.gradFnc.Backward(gradScale);
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
        public TorchSharp.Modules.Parameter getParameters()
        {
            return param;
        }
    }

}