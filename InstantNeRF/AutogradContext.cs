using static InstantNeRF.TcnnWrapper;
using static TorchSharp.torch;

namespace InstantNeRF
{
    // Context classes store results for the backward pass to compute gradients
    class AutogradContext : BasicContext
    {
        public AutogradContext(NerfModuleWrapper module, float lossScale) : base()
        {
            this.tcnnModule = module;
            this.nativeCtx = IntPtr.Zero;
            this.lossScale = lossScale;
        }
        public NerfModuleWrapper tcnnModule;
        public IntPtr nativeCtx;
        public float lossScale;
    }

    class VolumeRenderingContext : BasicContext
    {
        public VolumeRenderingContext(int rgbActivation, int densityActivation, float aabbMin, float aabbMax) : base()
        {
            this.aabbMin = aabbMin;
            this.aabbMax = aabbMax;
            this.rgbActivation = rgbActivation;
            this.densityActivation = densityActivation;
        }
        public int rgbActivation;
        public int densityActivation;
        public float aabbMin;
        public float aabbMax;
    }

    class BasicContext
    {
        public BasicContext() { }
        public List<Tensor> savedTensors = new List<Tensor>();
        public void saveForBackward(List<Tensor> tensors)
        {
            tensors.ForEach(t =>
            {
                savedTensors.Add(t);
            });
        }
    }
}