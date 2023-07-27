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
        private readonly int wantedBatchSize = 1024;
        private readonly int chunkSIze = 128;
        public Network(GridSampler sampler, float gradScale, float[] bgColor) : base("NerfNetwork") {
            this.sampler = sampler;

            float halfBound = 1.0f;

            if(sampler.dataInfo != null )
            {
                halfBound = (sampler.dataInfo.aabbMax - sampler.dataInfo.aabbMin) / 2;
            }
            this.mlp = new MLP(halfBound);
            this.renderer = new VolumeRenderer(bgColor);
            this.scaler = new GradScaler(gradScale);
        }
        public Dictionary<string, Tensor> forward( Dictionary<string, Tensor> data) 
        {

            Console.WriteLine("-:-:- before sampling -:-:-");

            data = this.sampler.Sample(data, this.mlp);

            Console.WriteLine("-:-:- after sampling -:-:-");

            data = this.mlp.forward(data);
            
            Console.WriteLine("-:-:- after mlp -:-:-");

            data = this.renderer.forward(sampler, data);

            Console.WriteLine("-:-:- after renderer -:-:-");

            return data;
        }

        public Tensor trainStep( Dictionary<string, Tensor> data)
        {
            sampler.train();
            mlp.train();
            renderer.train();
            
            Dictionary<string, Tensor> result = forward(data);

            data["alpha"].detach_();
            Tensor loss = Metrics.HuberLoss(result["rgb"], data["gt"]);

            Console.WriteLine("------------------------");
            Console.WriteLine("LOSS");
            Console.WriteLine("------------------------");
            loss.print();

            scaler.scale(loss).backward();
            renderer.backward();
            mlp.backward(scaler.getScale());


            return loss;
        }
        public Tensor testStep(Dictionary<string, Tensor> data)
        {
            sampler.eval();
            mlp.eval();
            renderer.eval();

            Dictionary<string, Tensor> result = this.forward(data);

            return result["rgb"];
        }
        public Dictionary<string, Tensor> batchifyForward( Dictionary<string, Tensor> data) //smaller batches to avoid memory problems
        {
            Dictionary<string, Tensor> totalData = new Dictionary<string, Tensor>();
            for ( var i = 0; i < wantedBatchSize; i = i + chunkSIze ) 
            {
                Dictionary<string, Tensor> chunk = new Dictionary<string, Tensor>();

                foreach ( var key in data.Keys ) 
                {
                    if (data[key].size(0) == wantedBatchSize)
                    {
                        chunk[key] = data[key].slice(0, i, i+chunkSIze, 1);
                    }
                    else
                    {
                        chunk[key] = data[key];
                    }
                }

                Dictionary<string, Tensor> result = this.forward(chunk);

                foreach (var key in result.Keys)
                {
                    totalData.Add(key, result[key]);
                }
            }
            return totalData;

        }

        public Tensor unfoldData( Tensor data)
        {
            if(data.Dimensions > 1)
            {
                long batchSize = data.size(0);
                
                List<Tensor> batch = new List<Tensor>();
                for (int i = 0; i < batchSize; i++)
                {
                    batch.Add(data[i]);
                }
                data = torch.cat(batch, 0);
            }
            return data;
        } 


    }
}
