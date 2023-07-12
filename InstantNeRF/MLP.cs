using TorchSharp;
using static TorchSharp.torch;
using Modules;

namespace InstantNeRF
{
    public class MLP : nn.Module
    {
        private Module encoderPos;
        private Module sigmaNet;
        private Module colorNet;
        private Module encoderDir;
        private float bound;
        private AutogradFunctions.TruncExp trunxExp;
        public MLP(string name, float bound) : base(name)
        {
            this.bound = bound;
            uint featureDims = 15u;
            uint neuronsPerLayer = 64u;
            uint nLayersSigma = 1u;
            uint nLayersColor = 2u;
            float perLevelScale = Convert.ToSingle(Math.Pow(2, Math.Log2(2048 * this.bound / 16) / (16 - 1)));

            string encodingPosCfg = "{\"otype\": \"HashGrid\", \"n_levels\": 16, \"n_features_per_level\": 2, \"log2_hashmap_size\": 19, \"base_resolution\": 16, \"per_level_scale\": " + perLevelScale.ToString(System.Globalization.CultureInfo.InvariantCulture) +"}";
            string networkSigmaCfg = "{\"otype\": \"FullyFusedMLP\", \"n_neurons\": " + neuronsPerLayer +", \"n_hidden_layers\": " + nLayersSigma +", \"activation\": \"ReLU\", \"output_activation\": \"None\"}";
            string encodingDirCfg = "{\"n_dims_to_encode\": 3, \"otype\": \"SphericalHarmonics\", \"degree\": 4}";
            string networkColorCfg = "{\"otype\": \"FullyFusedMLP\", \"n_neurons\": " + neuronsPerLayer +", \"n_hidden_layers\": " + nLayersColor +", \"activation\": \"ReLU\", \"output_activation\": \"None\"}";
            

            encoderPos = ModuleFactory.Encoding("PosEncoding", 3, encodingPosCfg, 0);
            sigmaNet = ModuleFactory.Network("SigmaNet", 32, featureDims + 1, networkSigmaCfg);
            encoderDir = ModuleFactory.Encoding("DirEncoding", 3, encodingDirCfg, 0);
            uint colorInputDims = encoderDir.outputDims + featureDims;
            colorNet = ModuleFactory.Network("ColorNet", colorInputDims, 3, networkColorCfg);
        }

        public (Tensor sigma, Tensor color) forward(Tensor positions, Tensor directions)
        {
            //Sigma
            positions = (positions + this.bound) / (2 * this.bound);


            Console.WriteLine("posEnc---------------------------");
            positions = encoderPos.forward(positions);


            Console.WriteLine("sigma---------------------------");
            Tensor sigmaOut = sigmaNet.forward(positions);


            //long lastDim = h.Dimensions - 1L;

            Tensor sigmaRawSliced = sigmaOut.slice(-1, 0L, 1L, 1L).squeeze(-1);

            Utils.printFirstNValues(sigmaRawSliced, 4, "sigmaRaw before exp");

            //Tensor sigma = torch.exp(FloatTensor(hSliced));
            this.trunxExp = new AutogradFunctions.TruncExp();
            Tensor sigma = trunxExp.forward(sigmaRawSliced);
            Utils.printFirstNValues(sigma, 4, "sigma after exp");
            Tensor geometryFeatures = sigmaOut.slice(-1, 1L, sigmaOut.size((int)-1), 1L);

            //Color

            //Spherical Harmonics requires the input to be in the range of [0,1]
            directions = (directions + 1) / 2;
            Console.WriteLine("dirEnc---------------------------");
            directions = encoderDir.forward(directions);


            sigmaOut = torch.cat(new List<Tensor>() { directions, geometryFeatures }, -1);

            Console.WriteLine("color---------------------------");
            sigmaOut = colorNet.forward(sigmaOut);
            Tensor color = torch.sigmoid(sigmaOut);
            Utils.printFirstNValues(color, 4, "color output");
            return (sigma, color);
        }
        public void backward(float gradScale)
        {
            Console.WriteLine("==>color--bwd");
            colorNet.backward(gradScale);
            Console.WriteLine("==>dirEnc--bwd");
            encoderDir.backward(gradScale);

            if(this.trunxExp != null)
            {
                trunxExp.backward();
            }

            Console.WriteLine("==>sigma--bwd");
            sigmaNet.backward(gradScale);
            Console.WriteLine("==>posEnc--bwd");
            encoderPos.backward(gradScale);
        }
        public (Tensor sigmas, Tensor geometryFeatures) density(Tensor input)
        {
            input = (input + bound) / (2 * bound);
            Console.WriteLine("posEnc|Dens---------------------------");
            input = encoderPos.forward(input);
            Console.WriteLine("sigma|Dens---------------------------");
            Tensor h = sigmaNet.forward(input);
            long lastDim = h.dim() - 1L;
            Tensor hSliced = h.slice(lastDim, 0L, 1L, 1L).squeeze(lastDim);
            Tensor sigma = torch.exp(hSliced);
            Tensor geometryFeatures = h.slice(lastDim, 1L, lastDim, 1L);
            return (sigma, geometryFeatures);
        }
        public List<TorchSharp.Modules.Parameter> getParams()
        {
            List<TorchSharp.Modules.Parameter> paramsList = new List<TorchSharp.Modules.Parameter>();
            paramsList.AddRange(encoderPos.parameters());
            paramsList.AddRange(sigmaNet.parameters());
            //paramsList.AddRange(encoderDir.parameters());
            paramsList.AddRange(colorNet.parameters());
            return paramsList;
        }

    }
}
