﻿using static InstantNeRF.TcnnWrapper;
using static TorchSharp.torch;

namespace InstantNeRF
{

    class AutogradContext
    {
        public AutogradContext(ModuleWrapper module, float lossScale)
        {
            this.lossScale = lossScale;
            this.tcnnModule = module;
            this.nativeCtx = IntPtr.Zero;
        }
        public List<Tensor> savedTensors = new List<Tensor>();
        public ModuleWrapper tcnnModule;
        public AutogradContext? forwardCtx;
        public IntPtr nativeCtx;
        public float lossScale;

        public void saveForBackward(List<Tensor> tensors)
        {
            tensors.ForEach(t =>
            {
                savedTensors.Add(t);
            });
        }
    }

    class RaymarchContext
    {
        public RaymarchContext(long M, long N, double tThreshhold)
        {
            this.M = M;
            this.N = N;
            this.tThreshhold = tThreshhold;
        }
        public List<Tensor> savedTensors = new List<Tensor>();
        public long M;
        public long N;
        public double tThreshhold;
        public void saveForBackward(List<Tensor> tensors)
        {
            tensors.ForEach(t =>
            {
                savedTensors.Add(t);
            });
        }

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