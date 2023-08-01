using System.IO;
using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace InstantNeRF
{
    public class Optimizer
    {
        private IntPtr handle;

        private Parameter param;
        private Parameter paramFP;
        public Optimizer(string config, MLP mlp)
        {
            handle = TcnnWrapper.createOptimizer(config);

            (param, paramFP) = mlp.getParameters();

            allocate(mlp.getHandle());
        }

        public void step()
        {
            float lossScale = param.dtype == float16 ? 128.0f : 1.0f;
            Tensor gradients = param.grad() ?? torch.empty_like(param);
            TcnnWrapper.step(handle, lossScale, param.Handle, paramFP.Handle, gradients.Handle);
        }

        public string hyperparams()
        {
            return TcnnWrapper.hyperparams(handle);
        }
        private void allocate(IntPtr module)
        {
            TcnnWrapper.allocate(handle, module);
        }

        public Parameter getParameter()
        {
            return param;
        }

    }
}