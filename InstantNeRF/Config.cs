using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using static Tensorboard.TensorShapeProto.Types;
using static TorchSharp.torch.distributions;

namespace InstantNeRF
{
    public class Config
    {
        public string dataPath;
        public string trainDataFilename;
        public string evalDataFilename;
        public string datasetType;
        public string posEncodingCfg;
        public string dirEncodingCfg;
        public string sigmaNetCfg;
        public string colorNetCfg;
        public float aabbMin;
        public float aabbMax;
        public float[] offset;
        public float aabbScale;
        public float imageDownscale;
        public int nRays;
        public float learningRate;
        public float[] bgColor;
        private readonly string configPath = "../../../../Config/config.json";

        public Config()
        {
            string pathToConfig = Path.Combine(Environment.CurrentDirectory, configPath);
            Console.WriteLine(pathToConfig);
            if (File.Exists(pathToConfig))
            {
                string jsonContent = File.ReadAllText(pathToConfig);
                JsonDocument json = JsonDocument.Parse(jsonContent);

                if (json.RootElement.TryGetProperty("dataPath", out JsonElement path))
                {
                    this.dataPath = path.GetString() ?? throw new Exception("wrong format for dataPath");
                }
                else throw new Exception("couldnt find dataPath");

                if (json.RootElement.TryGetProperty("trainDataFilename", out JsonElement trainFilename))
                {
                    this.trainDataFilename = trainFilename.GetString() ?? throw new Exception("wrong format for trainDataFilename");
                }
                else throw new Exception("couldnt find trainDataFilename");

                if (json.RootElement.TryGetProperty("evalDataFilename", out JsonElement evalFilename))
                {
                    this.evalDataFilename = evalFilename.GetString() ?? throw new Exception("wrong format for evalDataFilename");
                }
                else throw new Exception("couldnt find evalDataFilename");

                if (json.RootElement.TryGetProperty("datasetType", out JsonElement datasetTypeElement))
                {
                    this.datasetType = datasetTypeElement.GetString() ?? throw new Exception("wrong format for datasetType");
                }
                else throw new Exception("couldnt find datasetType");

                if (json.RootElement.TryGetProperty("posEncodingCfg", out JsonElement posEncoding))
                {
                    this.posEncodingCfg = posEncoding.GetRawText();
                }
                else throw new Exception("couldnt find posEncodingCfg");

                if (json.RootElement.TryGetProperty("dirEncodingCfg", out JsonElement dirEncoding))
                {
                    this.dirEncodingCfg = dirEncoding.GetRawText();
                }
                else throw new Exception("couldnt find dirEncodingCfg");

                if (json.RootElement.TryGetProperty("sigmaNetCfg", out JsonElement sigmaNet))
                {
                    this.sigmaNetCfg = sigmaNet.GetRawText();
                }
                else throw new Exception("couldnt find sigmaNetCfg");

                if (json.RootElement.TryGetProperty("colorNetCfg", out JsonElement colorNet))
                {
                    this.colorNetCfg = colorNet.GetRawText();
                }
                else throw new Exception("couldnt find colorNetCfg");

                if (json.RootElement.TryGetProperty("aabbMin", out JsonElement aabbMinELement))
                {
                    this.aabbMin = aabbMinELement.GetSingle();
                }
                else throw new Exception("couldnt find aabbMin");

                if (json.RootElement.TryGetProperty("aabbMax", out JsonElement aabbMaxELement))
                {
                    this.aabbMax = aabbMaxELement.GetSingle();
                }
                else throw new Exception("couldnt find aabbMax");

                if (json.RootElement.TryGetProperty("offset", out JsonElement offsetELement))
                {
                    this.offset = new float[3];
                    int index = 0;
                    foreach (var numberElement in offsetELement.EnumerateArray())
                    {
                        offset[index] = numberElement.GetSingle();
                        index++;
                    }

                }
                else throw new Exception("couldnt find offset");

                if (json.RootElement.TryGetProperty("aabbScale", out JsonElement aabbScaleElement))
                {
                    this.aabbScale = aabbScaleElement.GetSingle();
                }
                else throw new Exception("couldnt find aabbScaleh");

                if (json.RootElement.TryGetProperty("imageDownscale", out JsonElement imageDownscaleElement))
                {
                    this.imageDownscale = imageDownscaleElement.GetSingle();
                }
                else throw new Exception("couldnt find imageDownscale");

                if (json.RootElement.TryGetProperty("nRays", out JsonElement nRaysElement))
                {
                    this.nRays = nRaysElement.GetInt32();
                }
                else throw new Exception("couldnt find nRays");

                if (json.RootElement.TryGetProperty("learningRate", out JsonElement learningRateElement))
                {
                    this.learningRate= learningRateElement.GetSingle();
                }
                else throw new Exception("couldnt find learningRate");

                if (json.RootElement.TryGetProperty("bgColor", out JsonElement bgColorElement))
                {
                    int index = 0;
                    this.bgColor = new float[3];
                    foreach (var numberElement in bgColorElement.EnumerateArray())
                    {
                        bgColor[index] = numberElement.GetSingle();
                        index++;
                    }
                }
                else throw new Exception("couldnt find bgColor");

            }
            else
            {
                throw new Exception("Could not find config.json");
            }

        }

    }
}
