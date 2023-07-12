﻿using TorchSharp;
using static TorchSharp.torch;
using System.Text.Json;
using System.Collections.Generic;
using System.Xml.Schema;
using System;
using InstantNeRF;
using TorchSharp.Modules;

namespace InstantNeRF
{
    public class Trainer
    {
        private string name;
        private optim.lr_scheduler.LRScheduler scheduler;
        private TorchSharp.Modules.Adam optimizer;
        private Device device;
        private int evalEveryNEpochs;
        private bool updateSchedulerEveryStep;
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
            TorchSharp.Modules.Adam optimizerAdam,
            Network network,
            int evalEveryNEpochs = 50,
            bool updateSchedulerEveryStep = true,
            int nIterations = 30000,
            string subdirectoryName = "workspace",
            bool loadCheckpoint = false)
        {
            this.name = name;
            this.nIterations = nIterations;
            this.device = cuda.is_available() ? CUDA : CPU;
            this.optimizer = optimizerAdam;
            this.network = network;
            this.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda: iter => lambdaLR(iter));
            this.evalEveryNEpochs = evalEveryNEpochs;
            this.updateSchedulerEveryStep = updateSchedulerEveryStep;
            this.savePath = createDirectory(subdirectoryName);
            this.checkpointPath = createDirectory(subdirectoryName + "\\checkpoints");
            this.logPath = createDirectory(subdirectoryName + "\\logs");
            this.stats = new Stats();
            if (loadCheckpoint)
            {
                this.loadCheckpoint();
            }


        }

        public (Tensor groundTruth, Tensor predictedRgb, Tensor loss) evalStep(Dictionary<string, Tensor> data)
        {
            Tensor raysOrigin = data["raysOrigin"].to_type(torch.half);
            Tensor raysDirection = data["raysDirection"].to_type(torch.half);
            Tensor groundTruthImages = data["groundTruthImages"].to_type(torch.half);
            long BATCH = groundTruthImages.shape[0];
            long HEIGHT = groundTruthImages.shape[1];
            long WIDTH = groundTruthImages.shape[2];
            long COLOR = groundTruthImages.shape[3];

            Tensor images = groundTruthImages.slice(-1, 0, 3, 1);

            images = Utils.srgbToLinear(images);
            Tensor bgColor = (COLOR == 3) ? 1.0f : torch.rand_like(images);

            Tensor gtRGB;

            if (COLOR == 4)
            {
                gtRGB = images * groundTruthImages.slice(-1, 3, -1, 1) + bgColor * (1 - groundTruthImages.slice(-1, 3, -1, 1));
            }
            else gtRGB = images;
            Tensor rgbOut = this.network.testStep(data);

            Tensor predictedRGB = rgbOut.reshape(BATCH, HEIGHT, WIDTH, 3);
            //Tensor predictedDepth = depthOut.reshape(BATCH, HEIGHT, WIDTH);

            Tensor loss = Metrics.PSNR(predictedRGB, gtRGB);

            return (gtRGB, predictedRGB, loss);
        }

        public Tensor testStep(Dictionary<string, Tensor> data, long height, long width, float bgColor = 1.0f, bool perturb = false)
        {

            Tensor rgbOut = this.network.testStep(data);
            Tensor predictedRGB = rgbOut.reshape(-1, height, width, 3);
            //Tensor predictedDepth = depthOut.reshape(-1, height, width, 3);

            return predictedRGB;
        }

        public Tensor testInference(Tensor pose, Tensor intrinsics, int height, int width, bool depth = false, int downScale = 1)
        {
            int renderWidth = width * downScale;
            int renderHeight = height * downScale;
            intrinsics = intrinsics * (float)downScale;

            Tensor image;

            using(var d = torch.NewDisposeScope())
            {
                using (torch.no_grad())
                {
                    Dictionary<string, Tensor> data = Utils.getRays(pose, intrinsics, renderHeight, renderWidth, torch.empty(0));

                    if (depth)
                    {
                        image = testStep(data, Convert.ToInt64(height), Convert.ToInt64(width));
                    }
                    else
                    {
                        image = testStep(data, Convert.ToInt64(height), Convert.ToInt64(width));
                    }
                }

                if (downScale != 1)
                {
                    if (depth)
                    {
                        image = torch.nn.functional.interpolate(image.permute(0, 3, 1, 2), size: new long[] { height, width }, mode: InterpolationMode.Nearest).permute(0, 2, 3, 1).contiguous();
                    }
                    else
                    {
                        image = torch.nn.functional.interpolate(image.unsqueeze(1), size: new long[] { height, width }, mode: InterpolationMode.Nearest).squeeze(1);
                    }
                }
                d.DisposeEverything();
            }


            return Utils.linearToSrgb(image[0].detach());
        }

        public float trainStepRT(int step, DataProvider dataProvider)
        {
            float totalLoss = 0f;

            using (var d = torch.NewDisposeScope())
            {
                var data = dataProvider.getTrainData();
                this.globalStep++;

                optimizer.zero_grad();

                Tensor loss= this.network.trainStep(dataProvider.getTrainData());

                torch.nn.utils.clip_grad_norm_(network.mlp.parameters(), 2.0);

                this.network.scaler.step(optimizer);

                if (this.updateSchedulerEveryStep)
                {
                    scheduler.step();
                }
                else if (this.globalStep % 5 == 0)
                {
                    scheduler.step();
                }
                float lossValue = loss.item<float>();
                totalLoss += lossValue;
                d.DisposeEverything();
            }

            return totalLoss;         
        }


        private void evaluateOneEpoch(DataProvider validLoader)
        {
            Console.WriteLine("==> Evaluating Epoch: " + this.epoch);

            float totalLoss = 0;
            using (torch.no_grad())
            {
                this.localStep = 0;
                for (int index = 0; index < validLoader.batchSize; index++)
                {

                    Dictionary<string, Tensor> data = validLoader.getTrainData();
                    (Tensor gtImage, Tensor predictedRGB, Tensor loss) = this.evalStep(data);

                    float lossValue = loss.item<float>();
                    totalLoss += lossValue;
                }
            }
            float averageLoss = totalLoss / this.localStep;
            stats.validLoss.Append(averageLoss);
            stats.results.Append(averageLoss);

        }

        private double lambdaLR(int iter)
        {
            lastIter = iter;
            return Convert.ToDouble(Math.Pow(0.1d, Math.Min(iter / nIterations, 1)));
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

        public void saveCheckpoint(string? name)
        {
            string checkpointName = name ?? "" + this.name + "_ep" + this.epoch.ToString("D4");

            State state = new State();
            state.epoch = this.epoch;
            state.globalStep = this.globalStep;
            //state.network = this.nerfRenderer.state_dict();
            //state.meanCount = this.nerfRenderer.meanCount;
            //state.meanDensity = this.nerfRenderer.meanDensity;
            state.optimizer = this.optimizer.state_dict();
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
            //this.nerfRenderer.load_state_dict(state.network, strict: false);
            //this.nerfRenderer.meanCount = state.meanCount;
            //this.nerfRenderer.meanDensity = state.meanDensity;
            this.stats = state.stats;
            this.epoch = state.epoch;
            this.globalStep = state.globalStep;
            this.optimizer.load_state_dict(state.optimizer);
            this.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda: iter => lambdaLR(state.lastIteration + iter));
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
