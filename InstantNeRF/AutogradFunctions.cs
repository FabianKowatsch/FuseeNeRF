using static InstantNeRF.TcnnWrapper;
using static TorchSharp.torch;
using TorchSharp;
using InstantNeRF;

namespace InstantNeRF
{
    static class AutogradFunctions
    {

        public class ModuleFunction
        {
            public ModuleFunction(NerfModuleWrapper tcnnModule, float lossScale)
            {
                this.context = new AutogradContext(tcnnModule, lossScale);
            }

            private AutogradContext context;
            public Tensor Forward(Tensor input, Tensor parameters)
            {
                (IntPtr nativeCtx, Tensor output) = this.context.tcnnModule.forward(input, parameters);
                this.context.saveForBackward(new List<Tensor> { input, parameters, output });
                this.context.nativeCtx = nativeCtx;
                return output;
            }

            public void Backward(float gradScale)
            {

                Tensor output = this.context.savedTensors[2];
                Tensor outputGrad = output.grad() ?? torch.empty(0);
                if (outputGrad.numel() == 0L)
                {
                    Console.WriteLine("output has no grad");
                    return;
                }
                else if (!outputGrad.is_cuda)
                {
                    Console.WriteLine("outputGrad must be a CUDA Tensor");
                    outputGrad = outputGrad.cuda();
                }
                Tensor input = this.context.savedTensors[0];
                Tensor parameters = this.context.savedTensors[1];
                Tensor inputGrad;
                Tensor paramsGrad;

                using (torch.no_grad())
                {
                    Tensor scaledGrad = outputGrad * this.context.lossScale;


                    (inputGrad, paramsGrad) = this.context.tcnnModule.backward(this.context.nativeCtx, input, parameters, output, scaledGrad);

                    if (!inputGrad.IsInvalid)
                    {
                        inputGrad = (inputGrad.numel() == 0L) ? inputGrad : inputGrad / this.context.lossScale;
                    }
                    paramsGrad = (paramsGrad.numel() == 0L) ? paramsGrad : paramsGrad / this.context.lossScale / gradScale;
                    //paramsGrad = (paramsGrad.numel() == 0L) ? paramsGrad : paramsGrad / this.context.lossScale / gradScale;

                }
                if (!inputGrad.IsInvalid)
                {
                    input.backward(new List<Tensor> { (inputGrad).nan_to_num() });
                }


                if (parameters.numel() > 0)
                {
                    parameters.backward(new List<Tensor> { (paramsGrad).nan_to_num() });
                }
            }

        }


    }
}