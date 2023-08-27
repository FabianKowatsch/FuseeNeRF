using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace InstantNeRF
{
    public class MLP : nn.Module
    {
        private TcnnNerfModule tcnnMLP;
        public MLP(string encodingPos, string encodingDir, string networkSigma, string networkColor) : base("MLP")
        {
            uint positionDims = 3;
            uint directionDims = 3;
            uint extraDims = 0;
            uint offsetStartToDirection = positionDims + extraDims;
            tcnnMLP = new TcnnNerfModule("TcnnMLP", positionDims, directionDims, extraDims, offsetStartToDirection, encodingPos, encodingDir, networkSigma, networkColor);
        }

        public Dictionary<string, Tensor> forward(Dictionary<string, Tensor> data, bool inference = true)
        {
            // format the input data
            Tensor positions = data["positions"];
            Tensor directions = data["directions"];
            long batchSize = positions.size(0);
            Tensor input = torch.cat(new List<Tensor>() { positions, directions }, dim: 1);

            Tensor output;
            if (inference)
            {
                Console.WriteLine("I N F E R E N C E");
                output = tcnnMLP.inference(input);
            }
            else
            {
                Console.WriteLine("F O R W A R D");
                output = tcnnMLP.forward(input);
            }

            // remove the padding and flatten the output
            Tensor color = output.slice(1, 0, 3, 1);
            Tensor sigma = output.slice(1, 3, 4, 1);
            Tensor outputsFlat = torch.cat(new List<Tensor>() { color, sigma }, -1).to(torch.float32).contiguous();

            data.Add("raw", outputsFlat.reshape(batchSize, outputsFlat.size(-1)));
            return data;
        }

        public Tensor density(Tensor positionsFlat)
        {
            Console.WriteLine("D E N S I T Y");
            Tensor output = tcnnMLP.density(positionsFlat);

            //remove the feature vectors to extract sigma
            long lastDim = output.dim() - 1L;
            Tensor sigma = output.slice(lastDim, 0L, 1L, 1L).squeeze(lastDim);
            return sigma;
        }

        public Tensor backward(float gradScale)
        {
            Console.WriteLine("B A C K W A R D");
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
