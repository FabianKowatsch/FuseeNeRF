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
using Fusee.Base.Common;
using Fusee.Engine.Core.Effects;
using System.Xml.Serialization;
using TorchSharp.Modules;

namespace FuseeApp
{
    [FuseeApplication(Name = "FUSEE NeRF Viewer")]
    public class FuseeNeRF : RenderCanvas
    {
        // angle variables
        private static float _angleHorz, _angleVert, _angleVelHorz, _angleVelVert;
        private const float RotationSpeed = 7;
        private const float Damping = 0.8f;
        private SceneContainer _scene;
        private SceneRendererForward _sceneRenderer;
        private bool _keys;

        private Texture _bltDestinationTex;

        private Transform _simulatingCamPivotTransform;
        private SceneNode _simulatingCam;
        private SceneContainer _camScene;
        private const float ZNear = 0.1f;
        private const float ZFar = 10f;
        private readonly float _fovy = M.PiOver4;
        private Camera _camera;

        private float _focalX;
        private float _focalY;
        private float _centerX;
        private float _centerY;
        private int _renderWidth;
        private int _renderHeight;
        private int currentStep = 0;
        private int stepsToTrain;
        private DataProvider _dataProvider;
        private Config _config;
        private Trainer _trainer;

        private readonly Camera _uiCam = new(ProjectionMethod.Perspective, 0.1f, 1000, M.PiOver4)
        {
            BackgroundColor = float4.One
        };
        public CanvasRenderMode CanvasRenderMode
        {
            get
            {
                return _canvasRenderMode;
            }
            set
            {
                _canvasRenderMode = value;
                if (_canvasRenderMode == CanvasRenderMode.World)
                {
                    _uiCam.ProjectionMethod = ProjectionMethod.Perspective;
                }
                else
                {
                    _uiCam.ProjectionMethod = ProjectionMethod.Orthographic;
                }
            }
        }
        private CanvasRenderMode _canvasRenderMode;

        private float _initCanvasWidth;
        private float _initCanvasHeight;
        private float _canvasWidth = 16;
        private float _canvasHeight = 9;

        private Transform _camPivot;

        //Build a scene graph consisting of a canvas and other UI elements.
        private SceneContainer CreateScene()
        {


            var bltTextureNode = TextureNode.Create(
                "Blt",
                _bltDestinationTex,
                GuiElementPosition.GetAnchors(AnchorPos.Middle),
                GuiElementPosition.CalcOffsets(AnchorPos.Middle, new float2(0, 0), _initCanvasHeight, _initCanvasWidth, new float2(9, 9)),
                new float2(1, 1));


            var canvas = new CanvasNode(
                "Canvas",
                _canvasRenderMode,
                new MinMaxRect
                {
                    Min = new float2(-_canvasWidth / 2, -_canvasHeight / 2f),
                    Max = new float2(_canvasWidth / 2, _canvasHeight / 2f)
                })
            {
                Children = new ChildList()
                {
                    //Simple Texture Node, contains a Blt"ed" texture.
                    bltTextureNode
                }
            };

            canvas.AddComponent(MakeEffect.FromDiffuseSpecular((float4)ColorUint.Black));
            canvas.AddComponent(new Plane());

            _camPivot = new Transform()
            {
                Translation = new float3(0, 0, 0),
                Rotation = float3.Zero
            };

            return new SceneContainer
            {
                Children = new List<SceneNode>
                {
                    new SceneNode()
                    {
                        Name = "CamPivot",
                        Components = new List<SceneComponent>()
                        {
                            _camPivot
                        },
                        Children = new ChildList()
                        {
                            new SceneNode()
                            {
                                Name = "MainCam",
                                Components = new List<SceneComponent>()
                                {
                                    new Transform()
                                    {
                                        Translation = new float3(0, 0, -15),
                                        Rotation = float3.Zero
                                    },
                                    _uiCam
                                }
                            },
                        }
                    },
                    //Add canvas.
                    new SceneNode()
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
                    },

                }
            };
        }


        // Init is called on startup.
        public override void Init()
        {
            /*
             * NERF =====================================================================================================================================================
             */

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

                Network network = new Network(sampler, _config.gradScale, _config.bgColor, _config.dirEncodingCfg, _config.posEncodingCfg, _config.sigmaNetCfg, _config.colorNetCfg);
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

            /*
            * FUSEE =====================================================================================================================================================
            */
            CanvasRenderMode = CanvasRenderMode.Screen;

            if (_canvasRenderMode == CanvasRenderMode.Screen)
            {
                _initCanvasWidth = Width / 100f;
                _initCanvasHeight = Height / 100f;
            }
            else
            {
                _initCanvasWidth = 16;
                _initCanvasHeight = 9;
            }
            _canvasHeight = _initCanvasHeight;
            _canvasWidth = _initCanvasWidth;


            _renderWidth = 800 / (int)_config.imageDownscale;
            _renderHeight = 800 / (int)_config.imageDownscale;
            stepsToTrain = (int)_config.stepsToTrain;

            UpdateIntrinsics((float)_renderWidth, (float)_renderHeight, this._fovy);
            byte[] raw = new byte[_renderWidth * _renderHeight * 3];
            for (int i = 0; i < raw.Length; i++)
            {
                raw[i] = 0;
            }
            ImagePixelFormat format = new ImagePixelFormat(ColorFormat.RGB);
            _bltDestinationTex = new Texture(raw, _renderWidth, _renderHeight, format, false, wrapMode: TextureWrapMode.Repeat, filterMode: TextureFilterMode.Nearest);


            // Actual rendered scene
            _scene = CreateScene();

            //Simulating Camera to get the poses required for inference

            _camScene = new SceneContainer();
            _simulatingCamPivotTransform = new Transform();
            _camera = new Camera(ProjectionMethod.Perspective, ZNear, ZFar, _fovy) { BackgroundColor = float4.Zero };

            _simulatingCam = new SceneNode()
            {
                Name = "SimulatingCam",
                Components = new List<SceneComponent>()
                        {
                            new Transform() { Translation = new float3(0, 0, -4) },
                            _camera

                        }
            };
            var camNode = new SceneNode()
            {
                Name = "SimulatingCamNode",
                Children = new ChildList()
                {
                    _simulatingCam
                },
                Components = new List<SceneComponent>()
                {
                    _simulatingCamPivotTransform
                }
            };
            _camScene.Children.Add(camNode);
            _sceneRenderer = new SceneRendererForward(_scene);
        }
        public override void Update()
        {
            Controls();
            if (currentStep <= stepsToTrain)
            {
                TrainStep();

                if(currentStep == stepsToTrain)
                {
                    Console.ReadLine();
                }
            }
            InferenceStep();
        }

        private void InferenceStep()
        {
            //pose

            float[] matrix = _simulatingCam.GetGlobalTransformation().ToArray();
            Tensor pose = torch.from_array(matrix).reshape(4, 4);
            Tensor poseConverted = Utils.fuseeMatrixToNGP(pose, _config.aabbScale, _config.offset);

            //intrinsics

            float[,] intrinsicsArray = new float[,] {
                { _focalX, 0f, _centerX },
                {0f, _focalY, _centerY },
                {0f, 0f, 1f }
            };
            Tensor intrinsics = torch.from_array(intrinsicsArray);

            //inference pass

            byte[] buffer = _trainer.inferenceStep(poseConverted, intrinsics, _renderHeight, _renderWidth, _dataProvider);

            //update the texture

            ImagePixelFormat format = new ImagePixelFormat(ColorFormat.RGB);
            ImageData data = new ImageData(buffer, _renderWidth, _renderHeight, format);
            _bltDestinationTex.Blt(0, 0, data);
        }
        private void TrainStep()
        {
            float loss = _trainer.trainStep(currentStep, _dataProvider);
            currentStep++;
        }

        private void UpdateIntrinsics(float sensorWidth, float sensorHeight, float fov)
        {
            _focalX = (sensorWidth / 2f) / Convert.ToSingle(MathHelper.Tan(Convert.ToDouble(fov / 2d)));
            _focalY = (sensorHeight / 2f) / Convert.ToSingle(MathHelper.Tan(Convert.ToDouble(fov / 2d)));
            _centerX = sensorWidth / 2;
            _centerY = sensorHeight / 2;
        }
        private void Controls()
        {
            // Mouse and keyboard movement
            if (Input.Keyboard.LeftRightAxis != 0 || Input.Keyboard.UpDownAxis != 0)
            {
                _keys = true;
            }

            if (Input.Mouse.LeftButton)
            {
                _keys = false;
                _angleVelHorz = RotationSpeed * Input.Mouse.XVel * Time.DeltaTimeUpdate * 0.0005f;
                _angleVelVert = RotationSpeed * Input.Mouse.YVel * Time.DeltaTimeUpdate * 0.0005f;
            }
            else if (Input.Touch.GetTouchActive(TouchPoints.Touchpoint_0))
            {
                _keys = false;
                var touchVel = Input.Touch.GetVelocity(TouchPoints.Touchpoint_0);
                _angleVelHorz = RotationSpeed * touchVel.x * Time.DeltaTimeUpdate * 0.0005f;
                _angleVelVert = RotationSpeed * touchVel.y * Time.DeltaTimeUpdate * 0.0005f;
            }
            else
            {
                if (_keys)
                {
                    _angleVelHorz = RotationSpeed * Input.Keyboard.LeftRightAxis * Time.DeltaTimeUpdate;
                    _angleVelVert = RotationSpeed * Input.Keyboard.UpDownAxis * Time.DeltaTimeUpdate;
                }
                else
                {
                    var curDamp = (float)System.Math.Exp(-Damping * Time.DeltaTimeUpdate);
                    _angleVelHorz *= curDamp;
                    _angleVelVert *= curDamp;
                }
            }

            _angleHorz += _angleVelHorz;
            _angleVert += _angleVelVert;

            _simulatingCamPivotTransform.RotationQuaternion = QuaternionF.FromEuler(_angleVert, _angleHorz, 0);
        }



        // RenderAFrame is called once a frame
        public override void RenderAFrame()
        {
            Console.WriteLine("FPS: " + Time.FramesPerSecond.ToString("0.00"));
            _sceneRenderer.Render(RC);

            // Swap buffers: Show the contents of the back buffer (containing the currently rendered frame) on the front buffer.
            Present();
        }
    }
}
