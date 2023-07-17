using TorchSharp;
using static TorchSharp.torch;
using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;

namespace InstantNeRF
{


    public class DataProvider
    {
        private Device device;
        private NerfMode mode;
        private float downscale;
        private float aabbScale;
        private float[] offset;
        private int numRays;
        private int height;
        private int width;
        private string dataPath;
        private JsonDocument transforms;
        public float aabbMin;
        public float aabbMax;
        public float[] bgColor;
        public Tensor poses;
        public Tensor images;
        public Tensor intrinsics;
        public Tensor errorMap;
        public float radius;
        public long batchSize;

        public DataProvider(Device device, string dataPath, string jsonName, string mode, float downScale, float aabbScale, float aabbMin, float aabbMax, float[] offset, float[] bgColor, int numRays, bool preload, string datasetType)
        {
            this.device = device;
            this.mode = Utils.modeFromString(mode);
            this.downscale = downScale;
            this.aabbScale = aabbScale;
            this.offset = offset;
            this.aabbMin = aabbMin;
            this.aabbMax = aabbMax;
            this.bgColor = bgColor;
            this.numRays = numRays;
            this.dataPath = dataPath;
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

                if (this.mode == NerfMode.TRAIN)
                    this.errorMap = torch.ones(this.images.shape[0], 128 * 128, dtype: torch.float32);
                else this.errorMap = torch.empty(0);

                this.radius = this.poses.slice(2, 0, 3, 1).select(2, 3).norm(-1).mean(new long[] { 0 }).item<float>();

            }
            else if (datasetType == "synthetic")
            {
                (this.poses, this.images) = extractImageDataFromJSON(true);
                if (this.mode == NerfMode.TRAIN)
                    this.errorMap = torch.ones(this.images.shape[0], 128 * 128, dtype: torch.float32);
                else this.errorMap = torch.empty(0);
                this.radius = this.poses.slice(1, 0, 3, 1).select(2, 3).norm(-1).mean(new long[] { 0 }).item<float>();
            }
            else
            {
                throw new Exception("unknown dataset type");
            }
            this.intrinsics = extractIntrinsics();

            if (preload)
            {
                this.poses = this.poses.to(this.device);
                this.images = this.images.to(this.device);
                this.errorMap = this.errorMap.to(this.device);
            }
            this.batchSize = images.size(0);
        }
        private Tensor extractIntrinsics()
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
            return torch.from_array(new float[4] { focusX, focusY, centerX, centerY });
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
                    Tensor transform = torch.from_array(Utils.arrayToNGPMatrix(mtx, this.aabbScale, this.offset));
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

                float[,,] floatArray = new float[width, height, 3];

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        Rgb24 pixel = image[x, y];

                        floatArray[x, y, 0] = pixel.R / 255f;
                        floatArray[x, y, 1] = pixel.G / 255f;
                        floatArray[x, y, 2] = pixel.B / 255f;
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

                float[,,] floatArray = new float[width, height, 3];

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        Rgba32 pixel = image[x, y];

                        floatArray[x, y, 0] = pixel.R / 255f;
                        floatArray[x, y, 1] = pixel.G / 255f;
                        floatArray[x, y, 2] = pixel.B / 255f;
                    }
                }
                return torch.from_array(floatArray);
            }
        }
        public Dictionary<string, Tensor> collate(int index)
        {
            Dictionary<string, Tensor> results = new Dictionary<string, Tensor>();
            Tensor poses = this.poses[index].to(device);
            Tensor errorMap = this.errorMap[index];

            Dictionary<string, Tensor> raysDict = Utils.getRays(poses, this.intrinsics, this.height, this.width, errorMap, this.numRays);

            Tensor images = this.images[index].to(device);
            if (this.mode == NerfMode.TRAIN)
            {
                images = images.unsqueeze(0);
                long colorDims = images.shape[images.ndim - 1];
                images = torch.gather(images.view(1, -1, colorDims), 1, torch.stack(new List<Tensor>() { raysDict["indices"], raysDict["indices"], raysDict["indices"] }, -1));
            }
            results["groundTruthImages"] = images;
            if (this.errorMap.numel() != 0)
            {
                results["index"] = torch.tensor(index);
                results["indicesCoarse"] = raysDict["indicesCoarse"];
            }
            results["H"] = height;
            results["W"] = width;
            results["raysOrigin"] = raysDict["raysOrigin"];
            results["raysDirection"] = raysDict["raysDirection"];

            return results;
        }

    }

}
