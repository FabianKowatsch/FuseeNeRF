using System.IO;
using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;

namespace InstantNeRF
{
    public class Optimizer
    {
        private IntPtr handle;

        private TorchSharp.Modules.Parameter parameters;
        public Optimizer(string config, MLP mlp)
        {
            handle = TcnnWrapper.createOptimizer(config);

            parameters = mlp.getParameters();

            allocate(mlp.getHandle());
        }

        public void step()
        {
            float lossScale = parameters.dtype == float16 ? 128.0f : 1.0f;
            Tensor gradients = parameters.grad() ?? torch.empty_like(parameters);
            TcnnWrapper.step(handle, lossScale, parameters.Handle, gradients.Handle);
        }

        public string hyperparams()
        {
            return TcnnWrapper.hyperparams(handle);
        }
        private void allocate(IntPtr module)
        {
            TcnnWrapper.allocate(handle, module);
        }

        public TorchSharp.Modules.Parameter getParameters()
        {
            return parameters;
        }

    }
}