using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace InstantNeRF
{
    public class MLP : nn.Module
    {
        private TcnnNerfModule tcnnMLP;
        private float bound;
        public MLP(float bound) : base("MLP")
        {
            this.bound = bound;
            //uint featureDims = 15u;
            uint neuronsPerLayer = 64u;
            uint nLayersSigma = 1u;
            uint nLayersColor = 2u;
            float perLevelScale = Convert.ToSingle(Math.Pow(2, Math.Log2(2048 * this.bound / 16) / (16 - 1)));

            string encodingPosCfg = "{\"otype\": \"HashGrid\", \"n_levels\": 16, \"n_features_per_level\": 2, \"log2_hashmap_size\": 19, \"base_resolution\": 16, \"per_level_scale\": " + perLevelScale.ToString(System.Globalization.CultureInfo.InvariantCulture) + "}";
            string networkSigmaCfg = "{\"otype\": \"FullyFusedMLP\", \"n_neurons\": " + neuronsPerLayer + ", \"n_hidden_layers\": " + nLayersSigma + ", \"activation\": \"ReLU\", \"output_activation\": \"None\"}";
            string encodingDirCfg = "{\"n_dims_to_encode\": 3, \"otype\": \"SphericalHarmonics\", \"degree\": 4}";
            string networkColorCfg = "{\"otype\": \"FullyFusedMLP\", \"n_neurons\": " + neuronsPerLayer + ", \"n_hidden_layers\": " + nLayersColor + ", \"activation\": \"ReLU\", \"output_activation\": \"None\"}";

            uint positionDims = 3;
            uint directionDims = 3;
            uint extraDims = 0;
            uint offsetStartToDirection = positionDims;
            tcnnMLP = new TcnnNerfModule("TcnnMLP", positionDims, directionDims, extraDims, offsetStartToDirection, encodingPosCfg, encodingDirCfg, networkSigmaCfg, networkColorCfg);

        }

        public Dictionary<string, Tensor> forward(Dictionary<string, Tensor> data)
        {
            long batchSize = data["positions"].size(0);

            Tensor outputsFlat;
            if (this.training)
            {
               outputsFlat = this.runMlpForward(data["positions"], data["directions"]);
            }
            else
            {
                outputsFlat = this.runMlpInference(data["positions"], data["directions"]);

            }
            
            data.Add("raw", outputsFlat.reshape(batchSize, outputsFlat.size(-1)));
            return data;
        }

        public Tensor runMlpForward(Tensor positions, Tensor directions)
        {
            Console.WriteLine("F O R W A R D");

            Tensor input = torch.cat(new List<Tensor>() { positions, directions }, dim: 1);
            Tensor output = tcnnMLP.forward(input);
            Tensor color = output.slice(1, 0, 3, 1);
            Tensor sigma = output.slice(1, 3, 4, 1);

            Tensor result = torch.cat(new List<Tensor>() { color, sigma }, -1).to(torch.float32).contiguous();
            return result;
        }

        public Tensor runMlpInference(Tensor positions, Tensor directions)
        {
            Console.WriteLine("I N F E R E N C E");
            Tensor input = torch.cat(new List<Tensor>() { positions, directions }, dim: 1);
            Tensor output = tcnnMLP.forward(input);
            Tensor color = output.slice(1, 0, 3, 1);
            Tensor sigma = output.slice(1, 3, 4, 1);

            Tensor result = torch.cat(new List<Tensor>() { color, sigma }, -1).to(torch.float32).contiguous();

            return result;
        }

        public Tensor density(Tensor positionsFlat)
        {
            Console.WriteLine("D E N S I T Y");
            Tensor output = tcnnMLP.density(positionsFlat);
            long lastDim = output.dim() - 1L;
            Tensor sigma = output.slice(lastDim, 0L, 1L, 1L).squeeze(lastDim);
            return sigma;
        }

        public Tensor backward(float gradScale)
        {
            Console.WriteLine("- - BACKWARD - -");
            return tcnnMLP.backward(gradScale);
        }

        public List<Parameter> getParams()
        {
            List<Parameter> paramsList = new List<Parameter>();
            paramsList.AddRange(tcnnMLP.parameters());
            return paramsList;
        }

        public IntPtr getHandle()
        {
            return tcnnMLP.getHandle();
        }

        public (Parameter param, Parameter paramFP) getParameters()
        {
            return tcnnMLP.getParameters();
        }
    }
}
