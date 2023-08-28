using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace InstantNeRF
{
    public class Network : nn.Module
    {
        public GridSampler sampler;
        public MLP mlp;
        public GradScaler scaler; 
        private VolumeRenderer renderer;
        public Network(GridSampler sampler, float gradScale, float[] bgColor, string encodingPos, string encodingDir, string networkSigma, string networkColor) : base("NerfNetwork") {
            this.sampler = sampler;
            this.mlp = new MLP(encodingPos, encodingDir, networkSigma, networkColor);
            this.renderer = new VolumeRenderer(bgColor);
            this.scaler = new GradScaler(gradScale);
        }
        public Dictionary<string, Tensor> forward( Dictionary<string, Tensor> data) 
        {
            data = this.sampler.Sample(data, this.mlp);

            data = this.mlp.forward(data, !mlp.training);

            data = this.renderer.forward(sampler, data);

            return data;
        }

        public Tensor trainStep( Dictionary<string, Tensor> data, Optimizer optimizer)
        {
            sampler.train();
            mlp.train();
            renderer.train();
            Dictionary<string, Tensor> result = forward(data);
            data["alpha"].detach_();

            Tensor loss = renderer.backward(data["gt"]).mean();

            Tensor gradients = mlp.backward(scaler.getScale());

            scaler.step(optimizer, gradients);
            return loss;
        }
        public (Tensor rgb, Tensor alpha) testStep(Dictionary<string, Tensor> data)
        {
            sampler.eval();
            mlp.eval();
            renderer.eval();

            Dictionary<string, Tensor> result = this.forward(data);

            return (result["rgb"], result["alpha"]);
        }      
    }
}
