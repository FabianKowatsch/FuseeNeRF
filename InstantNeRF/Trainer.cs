using TorchSharp;
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
        private NerfRenderer nerfRenderer;
        private optim.lr_scheduler.LRScheduler scheduler;
        private Adam optimizer;
        private Device device;
        private Loss<Tensor, Tensor, Tensor> criterion;
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
        private ColorSpace colorSpace;
        private int patchSize;
        private float emaDecay;
        private ExponentialMovingAverage ema;
        private GradScaler scaler;
        private Tensor errorMap;
        private bool useEMA;
        private float[] bgColor;

        public Trainer(
            string name,
            NerfRenderer renderer,
            Adam optimizerAdam,
            Loss<Tensor, Tensor, Tensor> criterion,
            int patchSize,
            float[] bgColor,
            float emaDecay = 0.99f,
            int evalEveryNEpochs = 50,
            bool updateSchedulerEveryStep = true,
            int nIterations = 30000,
            string subdirectoryName = "workspace",
        bool loadCheckpoint = false,
        ColorSpace colorSpace = ColorSpace.Linear,
            bool useEMA = false
            )
        {
            this.name = name;
            //this.criterion = torch.nn.HuberLoss(reduction: nn.Reduction.None, delta: 0.1);
            //this.criterion = torch.nn.MSELoss(reduction: nn.Reduction.None);
            this.nIterations = nIterations;
            this.criterion = criterion;
            this.device = cuda.is_available() ? CUDA : CPU;
            this.nerfRenderer = renderer.cuda<NerfRenderer>(device.index);
            this.optimizer = optimizerAdam;
            this.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda: iter => lambdaLR(iter));
            this.evalEveryNEpochs = evalEveryNEpochs;
            this.updateSchedulerEveryStep = updateSchedulerEveryStep;
            this.savePath = createDirectory(subdirectoryName);
            this.checkpointPath = createDirectory(subdirectoryName + "\\checkpoints");
            this.logPath = createDirectory(subdirectoryName + "\\logs");
            this.stats = new Stats();
            this.colorSpace = colorSpace;
            this.patchSize = patchSize;
            this.emaDecay = emaDecay;
            this.ema = new ExponentialMovingAverage(this.emaDecay, renderer.mlp.getParams());
            this.useEMA = useEMA;
            this.errorMap = torch.empty(0);
            this.scaler = new GradScaler();
            this.bgColor = bgColor;
            if (loadCheckpoint)
            {
                this.loadCheckpoint();
            }


        }
        public (Tensor groundTruth, Tensor predicted, Tensor loss) trainStep(Dictionary<string, Tensor> data)
        {
            this.nerfRenderer.mlp.train();
            VolumeRendering.CompositeRaysTrain compositeRaysTrain = new VolumeRendering.CompositeRaysTrain();
            Tensor raysOrigin = data["raysOrigin"];
            Tensor raysDirection = data["raysDirection"];
            Tensor groundTruthImages = data["groundTruthImages"];
            long B = groundTruthImages.shape[0];
            long N = groundTruthImages.shape[1];
            long C = groundTruthImages.shape[2];

            Tensor images = groundTruthImages.slice(-1, 0, 3, 1);
            if (this.colorSpace == ColorSpace.Linear)
            {
                images = Utils.srgbToLinear(images);
            }

            Tensor bgColor = (C == 3) ? this.bgColor[0] : torch.rand_like(images);

            Tensor gtRGB;

            if (C == 4)
            {
                gtRGB = images * groundTruthImages.slice(-1, 3, -1, 1) + bgColor * (1 - groundTruthImages.slice(-1, 3, -1, 1));
            }
            else gtRGB = images;


            bool forceAllRays = (patchSize > 1);

            (Tensor weightsSum, Tensor depth, Tensor predictedRGB) = this.nerfRenderer.runNerfTrain(compositeRaysTrain, raysOrigin, raysDirection, bgColor.item<float>(), forceAllRays: forceAllRays, gammaGrad: 0f, perturb: true);


            Tensor loss = this.criterion.call(predictedRGB, gtRGB).mean(dimensions: new long[] { -1 });

            if (patchSize > 1)
            {
                gtRGB = gtRGB.view(-1, patchSize, patchSize, 3).permute(0, 3, 1, 2).contiguous();
                predictedRGB = predictedRGB.view(-1, patchSize, patchSize, 3).permute(0, 3, 1, 2).contiguous();
            }


            if (errorMap.numel() > 0)
            {
                Tensor index = data["index"];
                Tensor errorMap = this.errorMap[index].unsqueeze(0);
                Tensor indicesCoarse = data["indicesCoarse"].to(errorMap.device);

                Tensor error = loss.detach().to(errorMap.device);
                Tensor emaError = 0.1 * errorMap.gather(1, indicesCoarse) + 0.9 * error;
                errorMap = errorMap.scatter_(1, indicesCoarse, emaError);
                this.errorMap[index] = errorMap;

            }

            Console.WriteLine("loss");
            loss = loss.mean();
            loss.print();

            Console.WriteLine("=====before loss bwd");

            scaler.scale(loss).backward();

            /*
            Tensor weightsSumGrad = weightsSum.grad() ?? torch.empty(0);
            Tensor depthGrad = depth.grad() ?? torch.empty(0);
            Tensor imageGrad = predictedRGB.grad() ?? torch.empty(0);
            */

            Console.WriteLine("=====before volume rendering bwd");
            compositeRaysTrain.Backward();
            Console.WriteLine("=====before mlp bwd");
            this.nerfRenderer.mlp.backward(scaler.getScale());
            return (gtRGB, predictedRGB, loss);

        }

        public (Tensor groundTruth, Tensor predictedRgb, Tensor predictedDepth, Tensor loss) evalStep(Dictionary<string, Tensor> data)
        {
            this.nerfRenderer.mlp.eval();
            Tensor raysOrigin = data["raysOrigin"].to_type(torch.float32);
            Tensor raysDirection = data["raysDirection"].to_type(torch.float32);
            Tensor groundTruthImages = data["groundTruthImages"].to_type(torch.float32);
            long BATCH = groundTruthImages.shape[0];
            long HEIGHT = groundTruthImages.shape[1];
            long WIDTH = groundTruthImages.shape[2];
            long COLOR = groundTruthImages.shape[3];

            Tensor images = groundTruthImages.slice(-1, 0, 3, 1);
            if (this.colorSpace == ColorSpace.Linear)
            {
                images = Utils.srgbToLinear(images);
            }
            Tensor bgColor = (COLOR == 3) ? 1.0f : torch.rand_like(images);

            Tensor gtRGB;

            if (COLOR == 4)
            {
                gtRGB = images * groundTruthImages.slice(-1, 3, -1, 1) + bgColor * (1 - groundTruthImages.slice(-1, 3, -1, 1));
            }
            else gtRGB = images;
            (Tensor weightsSum, Tensor depthOut, Tensor rgbOut) = this.nerfRenderer.runNerfInference(raysOrigin, raysDirection, gammaGrad: 0f);

            Tensor predictedRGB = rgbOut.reshape(BATCH, HEIGHT, WIDTH, 3);
            Tensor predictedDepth = depthOut.reshape(BATCH, HEIGHT, WIDTH);

            Tensor loss = this.criterion.call(predictedRGB, gtRGB).mean();

            return (gtRGB, predictedRGB, predictedDepth, loss);
        }

        public (Tensor predictedRGB, Tensor predictedDepth) testStep(Dictionary<string, Tensor> data, long height, long width, float bgColor = 1.0f, bool perturb = false)
        {
            this.nerfRenderer.mlp.eval();
            Tensor raysOrigin = data["raysOrigin"];
            Tensor raysDirection = data["raysDirection"];

            (Tensor weightsSum, Tensor depthOut, Tensor rgbOut) = this.nerfRenderer.runNerfInference(raysOrigin, raysDirection, gammaGrad: 0f);
            Tensor predictedRGB = rgbOut.reshape(-1, height, width, 3);
            Tensor predictedDepth = depthOut.reshape(-1, height, width, 3);

            return (predictedRGB, predictedDepth);
        }

        public void train(DataProvider trainLoader, DataProvider validLoader, int maxEpochs)
        {
            this.nerfRenderer.markUntrainedGrid(trainLoader.poses, trainLoader.intrinsics);
            this.errorMap = trainLoader.errorMap;

            for (int epoch = 0; epoch < maxEpochs; epoch++)
            {
                this.epoch = epoch;

                trainOneEpoch(trainLoader);


                if (this.epoch % this.evalEveryNEpochs == 0)
                {
                    this.evaluateOneEpoch(validLoader);
                }
                if (this.savePath != null)
                {
                    this.saveCheckpoint(this.savePath);
                }
            }
        }

        public void evaluate(DataProvider validLoader)
        {
            evaluateOneEpoch(validLoader);
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

                    if (this.useEMA)
                    {
                        this.ema.store();
                        this.ema.copyTo();
                    }

                    if (depth)
                    {
                        image = testStep(data, Convert.ToInt64(height), Convert.ToInt64(width)).predictedDepth;
                    }
                    else
                    {
                        image = testStep(data, Convert.ToInt64(height), Convert.ToInt64(width)).predictedRGB;
                    }
                    if (this.useEMA)
                    {
                        ema.reStore();
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
            this.nerfRenderer.mlp.train();
            float totalLoss = 0f;

            using (var d = torch.NewDisposeScope())
            {
                var data = dataProvider.collate(step);
                if (this.globalStep % 16 == 0)
                {
                    this.nerfRenderer.updateExtraState();
                }
                this.globalStep++;

                optimizer.zero_grad();

                (Tensor gt, Tensor rgb, Tensor loss) = this.trainStep(data);

                torch.nn.utils.clip_grad_norm_(nerfRenderer.mlp.parameters(), 2.0);

                this.scaler.step(optimizer);

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
                if (this.useEMA)
                {
                    this.ema.update();
                }
                d.DisposeEverything();
            }
            Console.ReadLine();
            return totalLoss;         
        }

        public void trainOneEpoch(DataProvider trainLoader)
        {
            Console.WriteLine("==> Started Training Epoch: " + this.epoch + ", lr: " + this.optimizer.ParamGroups.First().LearningRate.ToString());
            float totalLoss = 0;
            this.localStep = 0;
            //torch.autograd.detect_anomaly(true);

            for (int index = 0; index < trainLoader.batchSize; index++)
            {
                if (index % 2 == 0)
                {
                    string? pause = Console.ReadLine();
                    Console.WriteLine("paused " + pause);
                }

                using (var d = torch.NewDisposeScope())
                {
                    if (this.globalStep % 16 == 0)
                    {
                        this.nerfRenderer.updateExtraState();
                    }
                    this.localStep++;
                    this.globalStep++;

                    this.optimizer.zero_grad();
                    Dictionary<string, Tensor> data = trainLoader.collate(index);


                    Console.WriteLine("==> Training Step - - - - - " + index + " of " + trainLoader.batchSize);


                    (Tensor gtImage, Tensor predicted, Tensor loss) = this.trainStep(data);


                    Console.WriteLine("==> Step done - - - - - " + index + " of " + trainLoader.batchSize);


                    torch.nn.utils.clip_grad_norm_(nerfRenderer.mlp.parameters(), 2.0);
                    scaler.step(optimizer);
                    scaler.update();

                    if (this.updateSchedulerEveryStep)
                    {
                        scheduler.step();
                    }
                    float lossValue = loss.item<float>();
                    totalLoss += lossValue;

                    d.DisposeEverything();
                    Console.WriteLine("Tensors to handle: " + d.DisposablesCount);

                }



            }
            if (this.useEMA)
            {
                this.ema.update();
            }
            float averageLoss = totalLoss / this.localStep;
            stats.loss.Append(averageLoss);

            if (!this.updateSchedulerEveryStep)
            {
                this.scheduler.step();
            }
            Console.WriteLine("==> Finished Training Epoch: " + this.epoch);

        }

        private void evaluateOneEpoch(DataProvider validLoader)
        {
            Console.WriteLine("==> Evaluating Epoch: " + this.epoch);

            float totalLoss = 0;
            this.nerfRenderer.mlp.eval();
            if (this.useEMA)
            {
                this.ema.store();
                this.ema.copyTo();
            }
            using (torch.no_grad())
            {
                this.localStep = 0;
                for (int index = 0; index < validLoader.batchSize; index++)
                {

                    Dictionary<string, Tensor> data = validLoader.collate(index);
                    (Tensor gtImage, Tensor predictedRGB, Tensor predictedDepth, Tensor loss) = this.evalStep(data);

                    float lossValue = loss.item<float>();
                    totalLoss += lossValue;
                }
            }
            float averageLoss = totalLoss / this.localStep;
            stats.validLoss.Append(averageLoss);
            stats.results.Append(averageLoss);
            if (this.useEMA)
            {
                ema.reStore();
            }

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
            state.network = this.nerfRenderer.state_dict();
            state.meanCount = this.nerfRenderer.meanCount;
            state.meanDensity = this.nerfRenderer.meanDensity;
            state.optimizer = this.optimizer.state_dict();
            state.ema = this.ema.state_dict();
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
            this.nerfRenderer.load_state_dict(state.network, strict: false);
            this.nerfRenderer.meanCount = state.meanCount;
            this.nerfRenderer.meanDensity = state.meanDensity;
            this.stats = state.stats;
            this.epoch = state.epoch;
            this.globalStep = state.globalStep;
            this.optimizer.load_state_dict(state.optimizer);
            this.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda: iter => lambdaLR(state.lastIteration + iter));
            this.ema.load_state_dict(state.ema);
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
