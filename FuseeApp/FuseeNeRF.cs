using Fusee.Engine.Common;
using Fusee.Engine.Core;
using Fusee.Engine.Core.Primitives;
using Fusee.Engine.Core.Scene;
using Fusee.Engine.Gui;
using Fusee.Math.Core;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using static Fusee.Engine.Core.Input;
using static Fusee.Engine.Core.Time;
using InstantNeRF;
using TorchSharp;
using static TorchSharp.torch;
using System;
using OpenTK.Mathematics;
using Fusee.Base.Core;
using Fusee.Engine.Core.Effects;
using OpenTK.Graphics.ES20;

namespace FuseeApp
{
    [FuseeApplication(Name = "FuseeTestApp", Description = "Yet another FUSEE App.")]
    public class FuseeNeRF : RenderCanvas
    {
        // angle variables
        private static float _angleHorz, _angleVert, _angleVelHorz, _angleVelVert;

        private const float RotationSpeed = 7;
        private const float Damping = 0.8f;

        private SceneContainer _camScene;
        private SceneRendererForward _sceneRenderer;

        private const float ZNear = 1f;
        private const float ZFar = 1000;
        private readonly float _fovy = M.PiOver4;

        private float _focalX;
        private float _focalY;
        private Texture texture;
        private int currentStep = 0;
        private readonly int stepsToTrain = 5;
        private DataProvider _dataProvider;
        private Config _config;
        private Transform _camPivotTransform;

        private Trainer _trainer;
        private Camera _camera;

        private bool _keys;

        private async Task Load()
        {
            //Simulate Camera to get the poses required for inference
            _camScene = new SceneContainer();
            _camPivotTransform = new Transform();
            _camera = new Camera(ProjectionMethod.Perspective, ZNear, ZFar, _fovy) { BackgroundColor = float4.One };
            var camNode = new SceneNode()
            {
                Name = "SimulatingCamNode",
                Children = new ChildList()
                {
                    new SceneNode()
                    {
                        Name = "SimulatingCam",
                        Components = new System.Collections.Generic.List<SceneComponent>()
                        {
                            new Transform() { Translation = new float3(0, 2, -10) },
                            _camera
                            
                        }
                    }
                },
                Components = new System.Collections.Generic.List<SceneComponent>()
                {
                    _camPivotTransform
                }
            };
            _camScene.Children.Add(camNode);

            //Actual Scene containing a Quad with a Texture

            SceneContainer textureScene = new SceneContainer();

            //Setup texture to write to
            var focalLengths = CalculateFocalLength(this.Width / 100, this.Height / 100, this._fovy);
            _focalX = focalLengths[0];
            _focalY = focalLengths[1];

            byte[] raw = new byte[this.Width * this.Height * 3];
            Fusee.Base.Common.ImagePixelFormat format = new Fusee.Base.Common.ImagePixelFormat(Fusee.Base.Common.ColorFormat.RGB);
            texture = new Texture(raw, this.Width, this.Height, format, false);

            var quad = new SceneNode()
            {
                Name = "Quad",
                Components = new List<SceneComponent>()
                {
                    new Plane(),
                    new Transform() { Translation = new float3(0, 0, 0) },
                    MakeEffect.FromUnlit(float4.One, texture)

                }
            };

            var textureCam = new SceneNode()
            {
                Name = "MainCamNode",
                Components = new List<SceneComponent>()
                {
                    new Transform() { Translation = new float3(0, 0, -10), Rotation = new float3(0, (float)Math.PI/2, 0) },
                    new Camera(ProjectionMethod.Orthographic, ZNear, 20f, _fovy) { BackgroundColor = float4.Zero }
                }

            };
            textureScene.Children.Add(textureCam);
            textureScene.Children.Add(quad);

            _sceneRenderer = new SceneRendererForward(textureScene);

        }

        public override async Task InitAsync()
        {
            await Load();
            await base.InitAsync();
        }


        // Init is called on startup.
        public override void Init()
        {
            try
            {
                string torchDir = Environment.GetEnvironmentVariable("TORCH_DIR") ?? throw new Exception("Could not find variable TORCH_DIR");
                string path = Path.Combine(torchDir, "lib/torch.dll");

                NativeLibrary.Load(path);
            }
            catch (Exception e)
            {
                Console.WriteLine("an error occured while loading a ´native library:" + e);
            }

            try
            {
                Console.WriteLine("torch version: " + __version__);
                Device device = cuda.is_available() ? CUDA : CPU;

                _config = new Config();

                DataProvider trainData = new DataProvider(device, _config.dataPath, _config.trainDataFilename, "train", _config.imageDownscale, _config.aabbScale, _config.aabbMin, _config.aabbMax, _config.offset, _config.bgColor, _config.nRays, preload: false, _config.datasetType);
                DataProvider evalData = new DataProvider(device, _config.dataPath, _config.evalDataFilename, "train", _config.imageDownscale, _config.aabbScale, _config.aabbMin, _config.aabbMax, _config.offset, _config.bgColor, _config.nRays, preload: false, _config.datasetType);
                Console.WriteLine("created datasets");

                GridSampler sampler = new GridSampler(trainData);
                Console.WriteLine("created gridsampler");

                Network network = new Network(sampler, _config.gradScale, _config.bgColor);
                Console.WriteLine("created net");

                TorchSharp.Modules.Adam optimizer = optim.Adam(network.mlp.getParams(), lr: _config.learningRate, beta1: _config.beta1, beta2: _config.beta2, eps: _config.epsilon, weight_decay: _config.weightDecay);
                Console.WriteLine("created optimizer");

                Loss<Tensor, Tensor, Tensor> criterion = torch.nn.MSELoss(reduction: nn.Reduction.None);
                Console.WriteLine("created loss");

                Trainer trainer = new Trainer("NGP001", optimizer, network, 1, subdirectoryName: "workspace_lego_synthetic");
                Console.WriteLine("created trainer");

                _trainer = trainer;
                _dataProvider = trainData;
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
            }

        }

        public override void Update()
        {
            _camPivotTransform.RotationQuaternion = QuaternionF.FromEuler(_angleVert, _angleHorz, 0);

            // Mouse and keyboard movement
            if (Keyboard.LeftRightAxis != 0 || Keyboard.UpDownAxis != 0)
            {
                _keys = true;
            }

            if (Mouse.LeftButton)
            {
                _keys = false;
                _angleVelHorz = RotationSpeed * Mouse.XVel * DeltaTimeUpdate * 0.0005f;
                _angleVelVert = RotationSpeed * Mouse.YVel * DeltaTimeUpdate * 0.0005f;
            }
            else if (Touch != null && Touch.GetTouchActive(TouchPoints.Touchpoint_0))
            {
                _keys = false;
                var touchVel = Touch.GetVelocity(TouchPoints.Touchpoint_0);
                _angleVelHorz = RotationSpeed * touchVel.x * DeltaTimeUpdate * 0.0005f;
                _angleVelVert = RotationSpeed * touchVel.y * DeltaTimeUpdate * 0.0005f;
            }
            else
            {
                if (_keys)
                {
                    _angleVelHorz = RotationSpeed * Keyboard.LeftRightAxis * DeltaTimeUpdate;
                    _angleVelVert = RotationSpeed * Keyboard.UpDownAxis * DeltaTimeUpdate;
                }
                else
                {
                    var curDamp = (float)System.Math.Exp(-Damping * DeltaTimeUpdate);
                    _angleVelHorz *= curDamp;
                    _angleVelVert *= curDamp;
                }
            }

            _angleHorz += _angleVelHorz;
            _angleVert += _angleVelVert;

            if(currentStep <= stepsToTrain)
            {
                TrainStep();
                if (currentStep == stepsToTrain)
                {
                    InferenceStep();
                    Console.ReadLine();
                }
            }


        }

        private void InferenceStep()
        {
            //pose
            
            //float4 viewport;
            //float4x4 matrix = _camera.GetProjectionMat(this.Width, this.Height, out viewport);

            float4x4 matrix = float4x4.Zero;
            matrix.Column1 = new float4(-0.999f , 0.004f , -0.013f, -0.05f);
            matrix.Column2 = new float4(-0.014f, -0.3f, 0.954f, 3.845f);
            matrix.Column3 = new float4(0f, 0.954f, 0.299f, 1.208f);
            matrix.Column4 = new float4(0f, 0f, 0f, 1f);

            Tensor pose = torch.from_array(matrix.ToArray()).reshape(4,4);

            Tensor poseConverted = Utils.matrixToNGP(pose, _config.aabbScale, _config.offset);

            //intrinsics
            float centerX = this.Width / 2;
            float centerY = this.Height / 2;

            float[,] intrinsicsArray = new float[,] {
                { _focalX, 0f, centerX },
                {0f, _focalY, centerY },
                {0f, 0f, 1f }
            };
            Tensor intrinsics = torch.from_array(intrinsicsArray);

            byte[] buffer = _trainer.inferenceStepRT(poseConverted, intrinsics, height: this.Height, width: this.Width, 1);
            Fusee.Base.Common.ImagePixelFormat format = new Fusee.Base.Common.ImagePixelFormat(Fusee.Base.Common.ColorFormat.RGB);
            ImageData data = new ImageData(buffer, this.Width, this.Height, format);
            texture.Blt(0, 0, data, width: this.Width, height: this.Height);
        }
        private void TrainStep()
        {
            Tensor loss = _trainer.trainStepRT(currentStep, _dataProvider);
            currentStep++;

        }

        // RenderAFrame is called once a frame
        public override void RenderAFrame()
        {
            _sceneRenderer.Render(RC);

            // Swap buffers: Show the contents of the backbuffer (containing the currently rendered frame) on the front buffer.
            Present();
        }


        private float[] CalculateFocalLength(float sensorWidth, float sensorHeight, float fov)
        {

            float focalLengthX = (sensorWidth / 2f) / Convert.ToSingle(MathHelper.Tan(Convert.ToDouble(fov / 2d)));
            float focalLengthY = (sensorHeight / 2f) / Convert.ToSingle(MathHelper.Tan(Convert.ToDouble(fov / 2d)));
            return new float[2] { focalLengthX, focalLengthY };
        }
    }
}
