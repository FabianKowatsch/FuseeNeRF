using Modules;
using System.IO;
using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;

namespace InstantNeRF
{
    public static class TcnnWrapper
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct Handle2D
        {
            public IntPtr handle1;
            public IntPtr handle2;
        }
        [StructLayout(LayoutKind.Sequential)]
        public struct Handle3D
        {
            public IntPtr handle1;
            public IntPtr handle2;
            public IntPtr handle3;
        }

        [DllImport("TcnnApi.dll")]
        public static extern Handle2D forward(IntPtr module, IntPtr input, IntPtr parameters);

        [DllImport("TcnnApi.dll")]
        public static extern Handle2D backward(IntPtr module, IntPtr ctx, IntPtr input, IntPtr parameters, IntPtr output, IntPtr outputGrad);

        [DllImport("TcnnApi.dll")]
        public static extern Handle3D backwardBackwardInput(IntPtr module, IntPtr ctx, IntPtr input, IntPtr parameters, IntPtr inputGradDl, IntPtr outputGrad);

        [DllImport("TcnnApi.dll")]
        public static extern IntPtr initialParams(IntPtr module, ulong seed);

        [DllImport("TcnnApi.dll")]
        public static extern uint nInputDims(IntPtr module);

        [DllImport("TcnnApi.dll")]
        public static extern uint nParams(IntPtr module);

        [DllImport("TcnnApi.dll")]
        public static extern uint nOutputDims(IntPtr module);

        [DllImport("TcnnApi.dll")]
        public static extern int paramPrecision(IntPtr module);

        [DllImport("TcnnApi.dll")]
        public static extern int outputPrecision(IntPtr module);

        [DllImport("TcnnApi.dll")]
        [return: MarshalAs(UnmanagedType.BStr)]
        public static extern string hyperparams(IntPtr module);

        [DllImport("TcnnApi.dll")]
        [return: MarshalAs(UnmanagedType.BStr)]
        public static extern string name(IntPtr module);

        [DllImport("TcnnApi.dll")]
        public static extern IntPtr createNetwork(uint inputDims, uint outputDims, string network);


        [DllImport("TcnnApi.dll")]
        public static extern IntPtr createEncoding(uint inputDims, string encoding, int precision);


        [DllImport("TcnnApi.dll")]
        public static extern IntPtr createNetworkWithInputEncoding(uint inputDims, uint outputDims, string encoding, string network);

        [DllImport("TcnnApi.dll")]
        public static extern void deleteModule(IntPtr module);

        [DllImport("TcnnApi.dll")]
        public static extern uint batchSizeGranularity();

        [DllImport("TcnnApi.dll")]
        public static extern void freeTemporaryMemory();

        [DllImport("TcnnApi.dll")]
        public static extern int cudaDevice();

        [DllImport("TcnnApi.dll")]
        public static extern void setCudaDevice(int device);
        [DllImport("TcnnApi.dll")]
        public static extern int preferredPrecision();


        public class ModuleWrapper
        {

            private IntPtr handle;
            public ModuleWrapper(IntPtr moduleRef)
            {
                handle = moduleRef;
            }
            ~ModuleWrapper()
            {
                deleteModule(handle);
                freeTemporaryMemory();
            }

            public (IntPtr ctx, Tensor output) forward(Tensor input, Tensor parameters)
            {
                Handle2D tuple = TcnnWrapper.forward(handle, input.Handle, parameters.Handle);
                Tensor output = Tensor.UnsafeCreateTensor(tuple.handle2);
                return (tuple.handle1, output);
            }
            public (Tensor inputGrad, Tensor paramsGrad) backward(IntPtr ctx, Tensor input, Tensor parameters, Tensor output, Tensor outputGrad)
            {

                Handle2D tuple = TcnnWrapper.backward(handle, ctx, input.Handle, parameters.Handle, output.Handle, outputGrad.Handle);
                Tensor inputGrad = Tensor.UnsafeCreateTensor(tuple.handle1);
                Tensor paramsGrad = Tensor.UnsafeCreateTensor(tuple.handle2);
                return (inputGrad, paramsGrad);
            }
            public (Tensor outputGradDl, Tensor paramsGrad, Tensor inputGrad) backwardBackwardInput(IntPtr ctx, Tensor input, Tensor parameters, Tensor inputGradDl, Tensor outputGrad)
            {
                Handle3D tuple = TcnnWrapper.backwardBackwardInput(handle, ctx, input.Handle, parameters.Handle, inputGradDl.Handle, outputGrad.Handle);
                Tensor outputGradDl = Tensor.UnsafeCreateTensor(tuple.handle1);
                Tensor paramsGrad = Tensor.UnsafeCreateTensor(tuple.handle2);
                Tensor inputGrad = Tensor.UnsafeCreateTensor(tuple.handle3);
                return (outputGradDl, paramsGrad, inputGrad);
            }
            public Tensor initialParams(ulong seed)
            {

                Tensor t = Tensor.UnsafeCreateTensor(TcnnWrapper.initialParams(handle, seed));

                return t;
            }
            public uint nInputDims()
            {
                uint result = TcnnWrapper.nInputDims(handle);
                return result;
            }
            public uint nParams()
            {
                uint result = TcnnWrapper.nParams(handle);
                return result;
            }
            public uint nOutputDims()
            {
                uint result = TcnnWrapper.nOutputDims(handle);
                return result;
            }
            public ScalarType paramPrecision()
            {
                int p = TcnnWrapper.paramPrecision(handle);
                switch (p)
                {
                    case 0: return torch.float32;
                    case 1: return torch.half;
                    default: throw new Exception("Unknown precision");
                }
            }
            public ScalarType outputPrecision()
            {
                int p = TcnnWrapper.outputPrecision(handle);
                switch (p)
                {
                    case 0: return torch.float32;
                    case 1: return torch.half;
                    default: throw new Exception("Unknown precision");
                }
            }
            public string hyperparams()
            {
                string result = TcnnWrapper.hyperparams(handle);
                return result;
            }
            public string name()
            {
                string result = TcnnWrapper.name(handle);
                return result;
            }
        }
    }
}
