﻿using Fusee.Engine.Common;
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
using Fusee.Base.Common;

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

        private const float ZNear = 0.1f;
        private const float ZFar = 10f;
        private readonly float _fovy = M.PiOver4;

        private float _focalX;
        private float _focalY;
        private float _centerX;
        private float _centerY;
        private int _renderWidth;
        private int _renderHeight;
        private Texture _texture;
        private int currentStep = 0;
        private readonly int stepsToTrain = 100;
        private DataProvider _dataProvider;
        private Config _config;
        private Transform _simulatingCamPivotTransform;
        private Transform _mainCamPivotTransform;

        private Trainer _trainer;
        private Camera _camera;

        private bool _keys;

        private async Task Load()
        {
            //Simulate Camera to get the poses required for inference
            _camScene = new SceneContainer();
            _simulatingCamPivotTransform = new Transform();
            _camera = new Camera(ProjectionMethod.Perspective, ZNear, ZFar, _fovy) { BackgroundColor = float4.Zero };
            var camNode = new SceneNode()
            {
                Name = "SimulatingCamNode",
                Children = new ChildList()
                {
                    new SceneNode()
                    {
                        Name = "SimulatingCam",
                        Components = new List<SceneComponent>()
                        {
                            new Transform() { Translation = new float3(0, 0, 0) },
                            _camera

                        }
                    }
                },
                Components = new List<SceneComponent>()
                {
                    _simulatingCamPivotTransform
                }
            };
            _camScene.Children.Add(camNode);

            //Actual Scene containing a Quad with a Texture

            SceneContainer textureScene = new SceneContainer();

            _renderWidth = 800 / (int)_config.imageDownscale;
            _renderHeight = 800 / (int)_config.imageDownscale;

            //Setup texture to write to
            UpdateIntrinsics((float)_renderWidth, (float)_renderHeight, this._fovy);

            byte[] raw = new byte[_renderWidth * _renderHeight * 3];
            ImagePixelFormat format = new ImagePixelFormat(ColorFormat.RGB);
            _texture = new Texture(raw, _renderWidth, _renderHeight, format, false, wrapMode: TextureWrapMode.ClampToEdge);



            // CANVAS
            /*
            float canvasHeight = Height;
            float canvasWidth = Width;
            var canvas = new CanvasNode(
                "Canvas",
                CanvasRenderMode.Screen,
                new MinMaxRect
                {
                    Min = new float2(-canvasWidth / 2f, -canvasHeight / 2f),
                    Max = new float2(canvasWidth / 2f, canvasHeight / 2f)
                })
            {
                Children = new ChildList()
                {
                TextureNode.Create(
                "Blt",
                _texture,
                GuiElementPosition.GetAnchors(AnchorPos.DownDownLeft),
                GuiElementPosition.CalcOffsets(AnchorPos.DownDownLeft, new float2(0, 0), canvasHeight, canvasWidth, new float2(4, 4)),
                float2.One)
                }
            };
            canvas.AddComponent(MakeEffect.FromDiffuseSpecular((float4)ColorUint.Red));
            canvas.AddComponent(new Plane());
            var canvasNode = new SceneNode()
            {
                Components = new List<SceneComponent>()
                        {
                            new Transform()
                            {
                                Translation = new float3(0, 0, 0),
                                Rotation = new float3(0, M.PiOver4, 0)
                            }
                        },
                Children = new ChildList()
                        {
                            canvas
                        }
            };
            */
            var quad = new SceneNode()
            {
                Name = "Quad",
                Components = new List<SceneComponent>()
                {
                    new Plane(),
                    new Transform() { Translation = new float3(0, 0, 0) },
                    //MakeEffect.FromUnlit(float4.One, _texture),
                    MakeEffect.Default()

                }
            };
            _mainCamPivotTransform = new Transform();
            var textureCam = new SceneNode()
            {
                Name = "MainCamNode",
                Children = new ChildList()
                {
                    new SceneNode()
                    {
                        Name = "MainCam",
                        Components = new List<SceneComponent>()
                        {
                            new Transform() { Translation = new float3(0, 0, -1) },
                            new Camera(ProjectionMethod.Perspective, ZNear, ZFar, _fovy) { BackgroundColor = float4.Zero }

                        }
                    }
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
                Device device = cuda.is_available() ? CUDA : CPU;

                _config = new Config();

                DataProvider trainData = new DataProvider(device, _config.dataPath, _config.trainDataFilename, "train", _config.imageDownscale, _config.aabbScale, _config.aabbMin, _config.aabbMax, _config.offset, _config.bgColor, _config.nRays, preload: false, _config.datasetType, _config.useRandomBgColor);
                DataProvider evalData = new DataProvider(device, _config.dataPath, _config.evalDataFilename, "val", _config.imageDownscale, _config.aabbScale, _config.aabbMin, _config.aabbMax, _config.offset, _config.bgColor, _config.nRays, preload: false, _config.datasetType, _config.useRandomBgColor);
                Console.WriteLine("created datasets");

                GridSampler sampler = new GridSampler(trainData);
                Console.WriteLine("created gridsampler");

                Network network = new Network(sampler, _config.gradScale, _config.bgColor);
                Console.WriteLine("created net");

                Optimizer optimizer = new Optimizer(_config.optimizerCfg, network.mlp);
                Console.WriteLine("created optimizer");

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

            if(currentStep == 0)
            {
                setInitialPose();
            }

            Controls();

            if (currentStep <= stepsToTrain)
            {
                TrainStep();
                if (currentStep % 2 == 0)
                {
                    InferenceStep();
                }
                if(currentStep == stepsToTrain)
                {
                    Console.ReadLine();
                }
            }

        }

        private void Controls()
        {
            //_simulatingCamPivotTransform.RotationQuaternion = QuaternionF.FromEuler(_angleVert, _angleHorz, 0);
            //_mainCamPivotTransform.RotationQuaternion = QuaternionF.FromEuler(_angleVert, _angleHorz, 0);

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
        }

        private void InferenceStep()
        {
            //pose

            float[] matrix = _simulatingCamPivotTransform.Matrix.ToArray();

            Tensor pose = torch.from_array(matrix).reshape(4, 4);

            Tensor poseConverted = Utils.matrixToNGP(pose, _config.aabbScale, _config.offset);

            //intrinsics


            float[,] intrinsicsArray = new float[,] {
                { _focalX, 0f, _centerX },
                {0f, _focalY, _centerY },
                {0f, 0f, 1f }
            };
            Tensor intrinsics = torch.from_array(intrinsicsArray);

            byte[] buffer = _trainer.inferenceStepRT(poseConverted, intrinsics, _renderHeight, _renderWidth);
            ImagePixelFormat format = new ImagePixelFormat(ColorFormat.RGB);
            ImageData data = new ImageData(buffer, _renderWidth, _renderHeight, format);
            _texture.Blt(0, 0, data, width: _renderWidth, height: _renderHeight);
        }
        private void TrainStep()
        {
            float loss = _trainer.trainStepRT(currentStep, _dataProvider);
            currentStep++;
        }

        private void setInitialPose()
        {
            float[,] startingPose = _dataProvider.getStartingPose();
            float4x4 matrix = float4x4.Zero;
            matrix.Row1 = new float4(startingPose[0,0], startingPose[0, 1], startingPose[0, 2], startingPose[0, 3]);
            matrix.Row2 = new float4(startingPose[1, 0], startingPose[1, 1], startingPose[1, 2], startingPose[1, 3]);
            matrix.Row3 = new float4(startingPose[2, 0], startingPose[2, 1], startingPose[2, 2], startingPose[2, 3]);
            matrix.Row4 = new float4(startingPose[3, 0], startingPose[3, 1], startingPose[3, 2], startingPose[3, 3]);
            _simulatingCamPivotTransform.Matrix = matrix;
        }

        // RenderAFrame is called once a frame
        public override void RenderAFrame()
        {
            _sceneRenderer.Render(RC);

            // Swap buffers: Show the contents of the backbuffer (containing the currently rendered frame) on the front buffer.
            Present();
        }


        private void UpdateIntrinsics(float sensorWidth, float sensorHeight, float fov)
        {
            _focalX = (sensorWidth / 2f) / Convert.ToSingle(MathHelper.Tan(Convert.ToDouble(fov / 2d)));
            _focalY = (sensorHeight / 2f) / Convert.ToSingle(MathHelper.Tan(Convert.ToDouble(fov / 2d)));
            _centerX = sensorWidth / 2;
            _centerY = sensorHeight / 2;
        }
    }
}
