/*
Copyright (c) 2022 hawkey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

//custom implementation in C# based on https://github.com/ashawkey/torch-ngp/blob/main/nerf/utils.py
using TorchSharp;
using static TorchSharp.torch;
using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Formats.Bmp;

namespace InstantNeRF
{
    public class Trainer
    {
        private string name;
        private Optimizer optimizer;
        private int nIterations;
        private int lastIter;
        private int epoch;
        private int globalStep;
        private int localStep;
        private string savePath;
        private string checkpointPath;
        private string logPath;
        private Stats stats;
        private Network network;


        public Trainer(
            string name,
            Optimizer optimizer,
            Network network,
            int nIterations = 30000,
            string subdirectoryName = "workspace",
            bool loadCheckpoint = false)
        {
            this.name = name;
            this.nIterations = nIterations;
            this.optimizer = optimizer;
            this.network = network;
            this.savePath = createDirectory(subdirectoryName);
            this.checkpointPath = createDirectory(subdirectoryName + "\\checkpoints");
            this.logPath = createDirectory(subdirectoryName + "\\logs");
            this.stats = new Stats();
            if (loadCheckpoint)
            {
                this.loadCheckpoint();
            }
        }
        public (Tensor predictedRgb, Tensor psnr) evalStep(DataProvider dataProvider)
        {
            long batchSize = dataProvider.images.size(0);
            int randomIndex = torch.randint(high: batchSize, size: new long[] { 1L }).item<int>();
            Tensor gtImage = dataProvider.images[randomIndex];
            Tensor pose = dataProvider.poses[randomIndex];
            (Tensor raysO, Tensor raysDir) = Utils.getRaysFromPose((int)gtImage.size(0), (int)gtImage.size(1), dataProvider.intrinsics, pose);
            //(Tensor raysO, Tensor raysDir) = Utils.getRaysFromPose(width, height, intrinsics, pose);

            raysO = raysO.reshape(-1, 3).to(CUDA); //from [H,W,3] to [H*W,3]
            raysDir = raysDir.reshape(-1, 3).to(CUDA); //from [H,W,3] to [H*W,3]
            Dictionary<string, Tensor> data = new Dictionary<string, Tensor>() { { "raysOrigin", raysO }, { "raysDirection", raysDir }, { "pose", pose.to(CUDA).contiguous() } };

            Tensor rgbOut = this.network.testStep(data).rgb;
            Tensor predictedRGB = Utils.linearToSrgb(rgbOut);
            Tensor psnr = Metrics.PSNR(predictedRGB, gtImage);

            return (predictedRGB, psnr);
        }


        public byte[] inferenceStep(Tensor pose, Tensor intrinsics, int height, int width, DataProvider dataProvider)
        {

            using (var d = torch.NewDisposeScope())
            {
                using (torch.no_grad())
                {
                    //This is currently not working correctly due to wrong pose transformations. To see results (that are also not correct), use the code below

                    (Tensor raysO, Tensor raysDir) = Utils.getRaysFromPose(width, height, intrinsics, pose);

                    //(Tensor raysO, Tensor raysDir) = Utils.getRaysFromPose(width, height, dataProvider.intrinsics, dataProvider.poses[0]);

                    //flatten the iamge

                    long[] imageShape = raysO.shape;
                    raysO = raysO.reshape(-1, 3).to(CUDA); //from [H,W,3] to [H*W,3]
                    raysDir = raysDir.reshape(-1, 3).to(CUDA); //from [H,W,3] to [H*W,3]

                    //inference pass

                    Dictionary<string, Tensor> data = new Dictionary<string, Tensor>() { { "raysOrigin", raysO }, { "raysDirection", raysDir }, { "pose", pose.to(CUDA).contiguous() } };

                    Tensor rgbOut = this.network.testStep(data).rgb; // [H*W,3]

                    // Transform the result into a byte buffer that can be used to update the texture

                    rgbOut = rgbOut.reshape(imageShape).swapaxes(0, 1).flip(0); // [H,W,3]
                    Tensor imageLinear = rgbOut.flatten(); // [H * W * 3]
                    Tensor image = ByteTensor(imageLinear * 255f).to(CPU);
                    byte[] buffer = image.data<byte>().ToArray();


                    // Optional: Save an iamge every 1000 steps in a subdirectory
                    if (this.globalStep % 1000 == 0)
                    {
                        using (Image<Rgb24> imageRaw = new Image<Rgb24>(Configuration.Default, width, height))
                        {
                            int index = 0;
                            for (int y = 0; y < height; y++)
                            {
                                for (int x = 0; x < width; x++)
                                {
                                    byte r = buffer[index++];
                                    byte g = buffer[index++];
                                    byte b = buffer[index++];

                                    imageRaw[x, y] = new Rgb24(r, g, b);
                                }
                            }
                            string outputPath = Path.Combine(this.savePath, "output_rgb_image_ " + this.globalStep + ".jpg");
                            imageRaw.Save(outputPath);
                        }
                    }

                    //Temporary Memory cleanup
                    d.DisposeEverything();
                    TcnnWrapper.freeTemporaryMemory();

                    return buffer;
                }
            }
        }

        public float trainStep(int step, DataProvider dataProvider)
        {
            float lossValue = 0;

            using (var d = torch.NewDisposeScope())
            {
                Dictionary<string, Tensor> data = dataProvider.getTrainData();

                Tensor loss = this.network.trainStep(data, optimizer);

                lossValue = loss.item<float>();


                //Temporary Memory cleanup
                d.DisposeEverything();
                TcnnWrapper.freeTemporaryMemory();

                this.globalStep++;
            }
            return lossValue;
        }

        private string createDirectory(string subdir)
        {
            string subdirectoryPath = Path.Combine(Environment.CurrentDirectory, subdir);

            if (Directory.Exists(subdirectoryPath))
            {
                return subdirectoryPath;
            }
            else
            {
                try
                {
                    Directory.CreateDirectory(subdirectoryPath);

                }
                catch (Exception ex)
                {
                    Console.WriteLine("Error creating subdirectory: " + ex.Message);
                }
                return subdirectoryPath;
            }
        }


        // Saving and loading a model is currently not supported
        public void saveCheckpoint(string? name)
        {
            string checkpointName = name ?? "" + this.name + "_ep" + this.epoch.ToString("D4");

            State state = new State();
            state.epoch = this.epoch;
            state.globalStep = this.globalStep;
            state.lastIteration = lastIter;
            string pathToFile = "" + this.checkpointPath + "/" + checkpointName + ".json";
            this.stats.checkpoints.Append(pathToFile);
            state.stats = this.stats;

            string jsonString = JsonSerializer.Serialize(state);
            File.WriteAllText(pathToFile, jsonString);
        }
        public void loadCheckpoint()
        {
            string pattern = $"{name}_ep*.json";

            string[] checkpointList = Directory.GetFiles(checkpointPath, pattern);
            Array.Sort(checkpointList);
            string checkpoint = checkpointList[checkpointList.Length - 1];

            string jsonString = File.ReadAllText(checkpoint);
            var state = JsonSerializer.Deserialize<State>(jsonString);
            this.stats = state.stats;
            this.epoch = state.epoch;
            this.globalStep = state.globalStep;
        }

    }

    public struct State
    {
        public int epoch;
        public int globalStep;
        public Stats stats;
        public Dictionary<string, Tensor> network;
        public optim.Optimizer.StateDictionary optimizer;
        public Dictionary<string, dynamic> ema;
        public int lastIteration;
        public int meanCount;
        public float meanDensity;

    }
    public struct Stats
    {
        public List<float> loss;
        public List<float> validLoss;
        public List<float> results;
        public List<string> checkpoints;
    }
}
