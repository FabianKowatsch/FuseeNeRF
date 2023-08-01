using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace InstantNeRF
{
    public class GradScaler
    {
        private readonly float growthFactor = 2.0f;
        private readonly int growthInterval = 2000;
        private readonly float backoffFactor = 0.5f;
        private float scaleFactor;
        private bool wasNaN;
        private int growthTracker;

        public GradScaler(float initialScale)
        {
            wasNaN = false;
            growthTracker = 0;
            scaleFactor = initialScale;
        }

        public Tensor scale(Tensor x)
        {
            return x * this.scaleFactor;
        }

        public void step(Optimizer optimizer)
        {


            Parameter parameters = optimizer.getParameter();

            Console.WriteLine("PARAMS BEFORE STEP");
            parameters.print();
            Console.WriteLine(parameters.IsInvalid);
            Console.WriteLine(parameters.IsInvalid);
            Console.WriteLine(parameters.device);
            Console.WriteLine(parameters.dtype);
            Console.WriteLine(parameters.numel());
            growthTracker++;

            if (isInvalid(parameters.grad()!))
            {
                wasNaN = true;
                growthTracker = 0;
            }


            optimizer.step();

            Console.WriteLine("PARAMS AFTER STEP");
            Console.WriteLine(parameters.IsInvalid);
            Console.WriteLine(parameters.device);
            Console.WriteLine(parameters.dtype);
            Console.WriteLine(parameters.numel());
            parameters.print();
            growthTracker++;

            if (isInvalid(parameters))
            {
                wasNaN = true;
                growthTracker = 0;
            }

        }
        public void update()
        {
            if (wasNaN)
            {
                this.scaleFactor *= backoffFactor;
            }
            else if (growthTracker >= growthInterval)
            {
                this.scaleFactor *= growthFactor;
                growthTracker = 0;
            }
            wasNaN = false;
        }

        public float getScale() { return scaleFactor; }

        private bool isInvalid(Tensor parameter)
        {
           return(parameter.isinf().any().item<bool>() || parameter.isnan().any().item<bool>());
        }
    }
}