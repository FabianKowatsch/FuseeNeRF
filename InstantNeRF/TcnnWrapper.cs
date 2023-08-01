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

        [DllImport("TcnnNerfApi.dll")]
        public static extern Handle2D forward(IntPtr module, IntPtr input, IntPtr parameters);

        [DllImport("TcnnNerfApi.dll")]
        public static extern Handle2D backward(IntPtr module, IntPtr ctx, IntPtr input, IntPtr parameters, IntPtr output, IntPtr outputGrad);

        [DllImport("TcnnNerfApi.dll")]
        public static extern IntPtr density(IntPtr module, IntPtr input, IntPtr parameters);

        [DllImport("TcnnNerfApi.dll")]
        public static extern IntPtr initialParams(IntPtr module, ulong seed);

        [DllImport("TcnnNerfApi.dll")]
        public static extern uint nInputDims(IntPtr module);

        [DllImport("TcnnNerfApi.dll")]
        public static extern uint nInputDimsDensity(IntPtr module);

        [DllImport("TcnnNerfApi.dll")]
        public static extern uint nParams(IntPtr module);

        [DllImport("TcnnNerfApi.dll")]
        public static extern uint nOutputDims(IntPtr module);

        [DllImport("TcnnNerfApi.dll")]
        public static extern uint nOutputDimsDensity(IntPtr module);

        [DllImport("TcnnNerfApi.dll")]
        public static extern int paramPrecision(IntPtr module);

        [DllImport("TcnnNerfApi.dll")]
        public static extern int outputPrecision(IntPtr module);

        [DllImport("TcnnNerfApi.dll")]
        [return: MarshalAs(UnmanagedType.BStr)]
        public static extern string hyperparams(IntPtr module);

        [DllImport("TcnnNerfApi.dll")]
        [return: MarshalAs(UnmanagedType.BStr)]
        public static extern string name(IntPtr module);


        [DllImport("TcnnNerfApi.dll")]
        public static extern IntPtr createNerfNetwork(uint posInputDims, uint dirInputDims, uint extraInputDims, uint dirOffset, string posEncoding, string dirEncoding, string sigmaNet, string colorNet);

        [DllImport("TcnnNerfApi.dll")]
        public static extern void deleteModule(IntPtr module);

        [DllImport("TcnnNerfApi.dll")]
        public static extern uint batchSizeGranularity();

        [DllImport("TcnnNerfApi.dll")]
        public static extern void freeTemporaryMemory();

        [DllImport("TcnnNerfApi.dll")]
        public static extern int cudaDevice();

        [DllImport("TcnnNerfApi.dll")]
        public static extern void setCudaDevice(int device);
        [DllImport("TcnnNerfApi.dll")]
        public static extern int preferredPrecision();

        [DllImport("TcnnNerfApi.dll")]
        public static extern IntPtr createOptimizer(string config);

        [DllImport("TcnnNerfApi.dll")]
        public static extern void step(IntPtr optimizer, float lossScale, IntPtr parameters, IntPtr parametersFullPrecision, IntPtr gradients);

        [DllImport("TcnnNerfApi.dll")]
        public static extern void allocate(IntPtr optimizer, IntPtr module);

        [DllImport("TcnnNerfApi.dll")]
        [return: MarshalAs(UnmanagedType.BStr)]
        public static extern string optimizer_hyperparams(IntPtr optimizer);

        public class NerfModuleWrapper
        {

            private IntPtr handle;
            public NerfModuleWrapper(IntPtr moduleRef)
            {
                handle = moduleRef;
            }
            ~NerfModuleWrapper()
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
            public Tensor density(Tensor input, Tensor parameters)
            {
                return Tensor.UnsafeCreateTensor(TcnnWrapper.density(handle, input.Handle, parameters.Handle));
            }
            public Tensor initialParams(ulong seed)
            {

                Tensor t = Tensor.UnsafeCreateTensor(TcnnWrapper.initialParams(handle, seed));

                return t;
            }
            public ScalarType paramPrecision()
            {
                switch (TcnnWrapper.paramPrecision(handle))
                {
                    case 0: return torch.float32;
                    case 1: return torch.float16;
                    default: return torch.float32;
                }
            }
            public uint nInputDims()
            {
                uint result = TcnnWrapper.nInputDims(handle);
                return result;
            }
            public uint nInputDimsDensity()
            {
                uint result = TcnnWrapper.nInputDimsDensity(handle);
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
            public uint nOutputDimsDensity()
            {
                uint result = TcnnWrapper.nOutputDimsDensity(handle);
                return result;
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
            public IntPtr getHandle()
            {
                return handle;
            }
        }
    }
}