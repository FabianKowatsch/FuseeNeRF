using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;
using Modules;
using static InstantNeRF.AutogradFunctions;

namespace InstantNeRF
{
    public class MLP : nn.Module
    {
        private float bound;
        private Module positionalEnc;
        private Module sigmaNet;
        private Module colorNet;
        private Module directionalEnc;
        public MLP(float bound = 1f) : base("MLP")
        {
            this.bound = bound;
            uint featureDims = 15u;
            uint neuronsPerLayer = 64u;
            uint nLayersSigma = 1u;
            uint nLayersColor = 2u;
            float perLevelScale = Convert.ToSingle(Math.Pow(2, Math.Log2(2048 * this.bound / 16) / (16 - 1)));

            string encodingPosCfg = "{\"otype\": \"HashGrid\", \"n_levels\": 16, \"n_features_per_level\": 2, \"log2_hashmap_size\": 19, \"base_resolution\": 16, \"per_level_scale\": " + perLevelScale.ToString(System.Globalization.CultureInfo.InvariantCulture) + "}";
            string networkSigmaCfg = "{\"otype\": \"FullyFusedMLP\", \"n_neurons\": " + neuronsPerLayer + ", \"n_hidden_layers\": " + nLayersSigma + ", \"activation\": \"ReLU\", \"output_activation\": \"None\"}";
            string encodingDirCfg = "{\"n_dims_to_encode\": 3, \"otype\": \"SphericalHarmonics\", \"degree\": 4}";
            string networkColorCfg = "{\"otype\": \"FullyFusedMLP\", \"n_neurons\": " + neuronsPerLayer + ", \"n_hidden_layers\": " + nLayersColor + ", \"activation\": \"ReLU\", \"output_activation\": \"None\"}";


            positionalEnc = ModuleFactory.Encoding("PosEncoding", 3, encodingPosCfg, 0);
            sigmaNet = ModuleFactory.Network("SigmaNet", 32, featureDims + 1, networkSigmaCfg);
            directionalEnc = ModuleFactory.Encoding("DirEncoding", 3, encodingDirCfg, 0);
            uint colorInputDims = directionalEnc.outputDims + featureDims;
            colorNet = ModuleFactory.Network("ColorNet", colorInputDims, 3, networkColorCfg);
        }

        public Dictionary<string, Tensor> forward(Dictionary<string, Tensor> data)
        {
            long[] shape = data["positions"].shape;
            long[] unflattenedShape = shape.Take(shape.Length -1).ToArray();
            Tensor outputsFlat = this.runMLP(data["positions"], data["directions"]);

            data.Add("raw", outputsFlat.reshape(unflattenedShape));
            return data;
        }

        public Tensor runMLP(Tensor positions, Tensor directions)
        {
            Utils.printDims(positions, "positions");
            Utils.printDims(directions, "directions");
            // Input Encoding
            Console.WriteLine("-:-:- before encoding pos -:-:-");
            Tensor positionsFlat = torch.reshape(positions, -1, positions.size(-1)).detach();
            Tensor positionsEncoded = this.positionalEnc.forward(positionsFlat);

            if (positions.Dimensions > directions.Dimensions)
                directions = directions.slice(0, 0, directions.size(0), 1).expand(positions.shape);

            Tensor directionsFlat = torch.reshape(directions, -1, directions.size(-1)).detach();

            Console.WriteLine("-:-:- before encoding dir -:-:-");

            Tensor directionsEncoded = this.directionalEnc.forward(directionsFlat);

            // Networks
            Console.WriteLine("-:-:- before sigma -:-:-");
            Tensor densityOut = this.sigmaNet.forward(positionsEncoded);
            Tensor sigma = densityOut.slice(-1, 0, 1, 1).squeeze(-1);
            Tensor geometryFeatures = densityOut.slice(-1, 1, densityOut.size(-1), 1);

            Tensor colorNetIn = torch.cat(new List<Tensor>() { geometryFeatures, directionsEncoded }, -1);

            Console.WriteLine("-:-:- before color -:-:-");
            Tensor colorOutput = this.colorNet.forward(colorNetIn);

            return torch.cat(new List<Tensor>() { colorOutput, sigma });
        }

        public Tensor density(Tensor positionsFlat)
        {
            Utils.printDims(positionsFlat, "positionsFlat");
            Tensor positionsEncoded = this.positionalEnc.forward(positionsFlat);
            Tensor densityOutput = this.sigmaNet.forward(positionsEncoded);
            Tensor density = densityOutput.slice(1, 0, 1, 1).to(torch.float32);
            return density;
        }

        public void backward(float gradScale)
        {
            colorNet.backward(gradScale);

            directionalEnc.backward(gradScale);

            sigmaNet.backward(gradScale);

            positionalEnc.backward(gradScale);
        }

        public List<TorchSharp.Modules.Parameter> getParams()
        {
            List<TorchSharp.Modules.Parameter> paramsList = new List<TorchSharp.Modules.Parameter>();
            paramsList.AddRange(positionalEnc.parameters());
            paramsList.AddRange(sigmaNet.parameters());
            //paramsList.AddRange(encoderDir.parameters());
            paramsList.AddRange(colorNet.parameters());
            return paramsList;
        }
    }
}
