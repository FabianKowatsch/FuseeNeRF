using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices.Marshalling;
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

        public void step(Adam optimizer)
        {

            var paramsArray = optimizer.parameters().ToArray();

            for (int i = 0; i < paramsArray.Length; i++)
            {
                growthTracker++;

                if (paramsArray[i].numel() > 0)
                {
                    if (paramsArray[i].grad()!.isinf().any().item<bool>() || paramsArray[i].grad()!.isnan().any().item<bool>())
                    {
                        wasNaN = true;
                        growthTracker = 0;
                    }
                }

            }

            optimizer.step();

            for (int i = 0; i < paramsArray.Length; i++)
            {
                growthTracker++;

                if (paramsArray[i].isinf().any().item<bool>() || paramsArray[i].isnan().any().item<bool>())
                {
                    wasNaN = true;
                    growthTracker = 0;
                }
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
    }
}
