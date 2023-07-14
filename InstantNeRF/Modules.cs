using TorchSharp;
using static TorchSharp.torch;
using static InstantNeRF.TcnnWrapper;
using InstantNeRF;
using System.Reflection.Metadata;
using System.Text;

namespace Modules
{
    public class Module : nn.Module<Tensor, Tensor>
    {
        private ModuleWrapper tcnnModule;
        private ulong seed;
        private float lossScale;
        private ScalarType dtype;
        private TorchSharp.Modules.Parameter param;
        public uint outputDims;
        private AutogradFunctions.ModuleFunction? gradFnc;
        public Module(string name, ModuleWrapper nativeTcnnModule, uint? outputDims, ulong seed = 1337) : base(name)
        {
            this.tcnnModule = nativeTcnnModule;
            this.seed = seed;
            this.dtype = tcnnModule.paramPrecision();

            if (outputDims != null)
            {
                this.outputDims = outputDims.Value;
            }
            else
            {
                this.outputDims = tcnnModule.nOutputDims();
            }
            Console.WriteLine("----" + name);
            if (tcnnModule.nParams() > 0)
            {
                Tensor initialParams = tcnnModule.initialParams(this.seed).to_type(this.dtype);
                param = torch.nn.Parameter(initialParams, requires_grad: true);
                this.register_parameter("params", param);
            }
            else
            {
                param = torch.nn.Parameter(torch.zeros(0, dtype: this.dtype, device: CUDA), requires_grad: true);
            }

            

            if (this.tcnnModule.paramPrecision() == torch.half)
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

            this.gradFnc = new AutogradFunctions.ModuleFunction();

            Utils.printFirstNValues(param, 3, " -- parameters --");

            Tensor output = this.gradFnc.ApplyFwd(
                this.tcnnModule,
                xPadded.to_type(torch.float32).contiguous(),
                param.contiguous().nan_to_num(),
                this.lossScale);
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
    }
    public static class ModuleFactory
    {
        public static Module NetworkWithInputEncoding(string name,uint inputDims, uint outputDims, string encodingCfg, string networkCfg)
        {
            ModuleWrapper tcnnModule = new ModuleWrapper(moduleRef: TcnnWrapper.createNetworkWithInputEncoding(inputDims, outputDims, encodingCfg, networkCfg));
            return new Module(name, tcnnModule, outputDims);
        }
        public static Module Network(string name, uint inputDims, uint outputDims, string networkCfg)
        {
            ModuleWrapper tcnnModule = new ModuleWrapper(moduleRef: TcnnWrapper.createNetwork(inputDims, outputDims, networkCfg));
            return new Module(name, tcnnModule, outputDims);
        }
        public static Module Encoding(string name, uint inputDims, string encodingCfg, int precision)
        {
            ModuleWrapper tcnnModule = new ModuleWrapper(moduleRef: TcnnWrapper.createEncoding(inputDims, encodingCfg, precision));
            return new Module(name, tcnnModule, null);
        }
    }


   
}
