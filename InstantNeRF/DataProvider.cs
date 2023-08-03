using TorchSharp;
using static TorchSharp.torch;
using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SkiaSharp;
using System.Numerics;

namespace InstantNeRF
{
    public class DataProvider
    {
        private Device device;
        private NerfMode mode;
        private float downscale;
        private float[] offset;
        public float[] bgColor;
        private bool useRandomBgColor;
        private float[,] initialPose;
        public int numRays;
        private int currentIndex;
        private string dataPath;
        private JsonDocument transforms;
        public float aabbMax;
        public float aabbMin;
        public float aabbScale;
        public int width;
        public int height;
        public Tensor focals;
        public Tensor poses;
        public Tensor images;
        public Tensor intrinsics;
        public Tensor raysAndRGBS;
        public long batchSize;

        public DataProvider(Device device, string dataPath, string jsonName, string mode, float downScale, float aabbScale, float aabbMin, float aabbMax, float[] offset, float[]bgColor, int numRays, bool preload, string datasetType, bool useRandomBgColor)
        {
            this.device = device;
            this.mode = Utils.modeFromString(mode);
            this.downscale = downScale;
            this.aabbScale = aabbScale;
            this.offset = offset;
            this.bgColor = bgColor;
            this.aabbMin = aabbMin;
            this.aabbMax = aabbMax;
            this.numRays = numRays;
            this.dataPath = dataPath;
            this.useRandomBgColor = useRandomBgColor;
            string pathToTransforms = Path.Combine(dataPath, jsonName + ".json");
            if (File.Exists(pathToTransforms))
            {
                string jsonContent = File.ReadAllText(pathToTransforms);
                transforms = JsonDocument.Parse(jsonContent);
            }
            else
            {
                throw new Exception("Could not find transforms.json, must run colmap on dataset first");
            }
            if (datasetType == "real")
            {
                if (transforms.RootElement.TryGetProperty("w", out JsonElement wElement))
                    width = (int)Math.Floor(wElement.GetInt32() / downscale);
                else
                    width = 0;

                if (transforms.RootElement.TryGetProperty("h", out JsonElement hElement))
                    height = (int)Math.Floor(hElement.GetInt32() / downscale);
                else
                    height = 0;

                (this.poses, this.images) = extractImageDataFromJSON(false);

            }
            else if (datasetType == "synthetic")
            {
                (this.poses, this.images) = extractImageDataFromJSON(true);
            }
            else
            {
                throw new Exception("unknown dataset type");
            }

            float[] rawIntrinsics = extractIntrinsics();

            float[,] intrinsicsArray = new float[,] { 
                { rawIntrinsics[0], 0f, rawIntrinsics[2] },
                {0f, rawIntrinsics[1], rawIntrinsics[3] }, 
                {0f, 0f, 1f }
            };
            this.intrinsics = torch.from_array(intrinsicsArray);
            this.focals = torch.from_array(new float[] { rawIntrinsics[0], rawIntrinsics[1] });

            if (preload)
            {
                this.poses = this.poses.to(this.device);
                this.images = this.images.to(this.device);
            }
            this.batchSize = images.size(0);
            Utils.printDims(this.images, "images");
            Console.WriteLine("test_alpha:" + this.images[0, 0, 0 ,3].item<float>());

            Tensor rays = Utils.loadRays(this.poses, this.images, this.intrinsics, width, height);


            if (this.mode == NerfMode.TRAIN)
            {
                Tensor indices = torch.randperm(rays.size(0));
                Tensor shuffledRays = rays[indices];
                this.raysAndRGBS = shuffledRays;
            }
            else
            {
                this.raysAndRGBS = rays;
            }
            this.currentIndex = 0;
        }
        private float[] extractIntrinsics()
        {
            float focusX;
            if (transforms.RootElement.TryGetProperty("fl_x", out JsonElement focusXElement))
                focusX = focusXElement.GetSingle() / downscale;
            else if (transforms.RootElement.TryGetProperty("camera_angle_x", out JsonElement camXElement))
                focusX = this.width / (2 * (float)Math.Tan(camXElement.GetSingle() / 2));
            else focusX = 0;

            float focusY;
            if (transforms.RootElement.TryGetProperty("fl_y", out JsonElement focusYElement))
                focusY = focusYElement.GetSingle() / downscale;
            else if (transforms.RootElement.TryGetProperty("camera_angle_y", out JsonElement camYElement))
                focusY = this.height / (2 * (float)Math.Tan(camYElement.GetSingle() / 2));
            else focusY = 0;

            focusY = focusY == 0 ? focusX : focusY;
            focusX = focusX == 0 ? focusX : focusY;

            float centerX;
            if (transforms.RootElement.TryGetProperty("cx", out JsonElement centerXElement))
                centerX = centerXElement.GetSingle() / downscale;
            else centerX = this.width / 2;

            float centerY;
            if (transforms.RootElement.TryGetProperty("cy", out JsonElement centerYElement))
                centerY = centerYElement.GetSingle() / downscale;
            else centerY = this.height / 2;
            return new float[4] { focusX, focusY, centerX, centerY };
        }

        private (Tensor poses, Tensor images) extractImageDataFromJSON(bool useSynthetic)
        {
            List<Tensor> posesList = new List<Tensor>();
            List<Tensor> imageList = new List<Tensor>();
            if (transforms.RootElement.TryGetProperty("frames", out JsonElement framesArray))
            {
                int counter = 0;
                foreach (var frameElement in framesArray.EnumerateArray())
                {

                    float[,] mtx = new float[4, 4];
                    if (frameElement.TryGetProperty("transform_matrix", out JsonElement matrixArray))
                    {
                        int dim = 0;
                        foreach (var matrixElement in matrixArray.EnumerateArray())
                        {
                            int index = 0;

                            foreach (var numberElement in matrixElement.EnumerateArray())
                            {
                                mtx[dim, index] = numberElement.GetSingle();

                                index++;
                            }
                            dim++;
                        }
                    }
                    string filePath = "";
                    if (frameElement.TryGetProperty("file_path", out JsonElement pathElement))
                    {
                        filePath = pathElement.GetString() ?? throw new Exception("wrong path");
                    }
                    Tensor image = useSynthetic ? getImageDataFromPNG(Path.Combine(dataPath, filePath + ".png")) : getImageDataFromJPG(Path.Combine(dataPath, filePath + ".jpg"));
                    image = Utils.srgbToLinear(image);

                    if(counter == 0)
                    {
                        initialPose = mtx;
                    }
                    Tensor transform = torch.from_array(mtx);
                    if (!useSynthetic)
                    {
                        switch (mode)
                        {
                            case NerfMode.TRAIN:
                                if (counter % 2 == 0)
                                {
                                    imageList.Add(image);

                                    posesList.Add(transform);
                                }
                                break;
                            case NerfMode.VAL:
                                if (counter % 2 != 0)
                                {
                                    imageList.Add(image);
                                    posesList.Add(transform);
                                }
                                break;
                            default:
                                imageList.Add(image);
                                posesList.Add(transform);
                                break;
                        }

                    }
                    else
                    {
                        imageList.Add(image);
                        posesList.Add(transform);
                    }
                    counter++;
                }
            }
            Tensor poses = torch.stack(posesList);
            poses = Utils.posesToNGP(poses, this.aabbScale, this.offset);
            Tensor images = torch.stack(imageList);
            return (poses, images);
        }

        private Tensor getImageDataFromJPG(string path)
        {
            using (Image<Rgb24> image = Image.Load<Rgb24>(path))
            {
                this.width = (int)(image.Width / this.downscale);
                this.height = (int)(image.Height / this.downscale);
                if (width != image.Width || height != image.Height)
                {
                    image.Mutate(x => x.Resize(new ResizeOptions
                    {
                        Size = new SixLabors.ImageSharp.Size(width, height),
                        Mode = ResizeMode.Stretch
                    }));
                }

                float[,,] floatArray = new float[width, height, 4];

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        Rgb24 pixel = image[x, y];

                        floatArray[x, y, 0] = pixel.R / 255f;
                        floatArray[x, y, 1] = pixel.G / 255f;
                        floatArray[x, y, 2] = pixel.B / 255f;
                        floatArray[x, y, 3] = 1f;
                    }
                }
                return torch.from_array(floatArray);
            }
        }
        private Tensor getImageDataFromPNG(string path)
        {
            using (Image<Rgba32> image = Image.Load<Rgba32>(path))
            {
                this.width = (int)(image.Width / this.downscale);
                this.height = (int)(image.Height / this.downscale);
                if (width != image.Width || height != image.Height)
                {
                    image.Mutate(x => x.Resize(new ResizeOptions
                    {
                        Size = new SixLabors.ImageSharp.Size(width, height),
                        Mode = ResizeMode.Stretch
                    }));
                }

                float[,,] floatArray = new float[width, height, 4];

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        Vector4 color = image[x, y].ToVector4();

                        floatArray[x, y, 0] = color.X;
                        floatArray[x, y, 1] = color.Y;
                        floatArray[x, y, 2] = color.Z;
                        floatArray[x, y, 3] = color.W;

                    }
                }
                return torch.from_array(floatArray);
            }
        }

        public Dictionary<string, Tensor> getTrainData() 
        {
            if(this.currentIndex + numRays > this.raysAndRGBS.size(0))
                this.currentIndex = 0;

            int startingIndex = this.currentIndex;
            int endIndex = this.currentIndex + numRays;

            Tensor batch = this.raysAndRGBS.slice(0, startingIndex, endIndex, 1);

            Tensor raysO = batch.slice(1, 0, 3, 1);
            Tensor raysD = batch.slice(1, 3, 6, 1);

            Tensor gtColors = batch.slice(1, 6, 9, 1);
            Tensor alpha = batch.slice(1, 9, 10, 1);
            Tensor indices = batch.slice(1, 10, batch.size(1), 1);

            //Color perturbation while training
            Tensor bgColor;
            if(useRandomBgColor)
            {
                bgColor = torch.rand(gtColors.shape, torch.float32);
            }
            else
            {
                bgColor = torch.ones_like(gtColors, torch.float32).multiply(torch.from_array(this.bgColor));
            }

            if (this.mode == NerfMode.TRAIN)
            {
                gtColors = gtColors * alpha + bgColor * (1 - alpha);
            }

            this.currentIndex += numRays;

            Dictionary<string, Tensor> results = new Dictionary<string, Tensor>() 
            {
                { "raysOrigin", raysO.to(CUDA) },
                {"raysDirection", raysD.to(CUDA)},
                {"gt", gtColors.to(float32).to(CUDA)},
                {"alpha", alpha.to(CUDA) },
                {"bgColor", bgColor.to(float32).to(CUDA) },
                {"imageIndices", indices }
            };

            return results;

        }

        public float[,] getStartingPose()
        {
            return this.initialPose;
        }

    }

}
