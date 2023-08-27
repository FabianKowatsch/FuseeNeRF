using static InstantNeRF.TcnnWrapper;
using static TorchSharp.torch;
using TorchSharp;
using InstantNeRF;

namespace InstantNeRF
{
    public class AutogradFunction
    {
        private AutogradContext context;
        public AutogradFunction(NerfModuleWrapper tcnnModule, float lossScale)
        {
            this.context = new AutogradContext(tcnnModule, lossScale);
        }


        public Tensor Forward(Tensor input, Tensor parameters)
        {
            (IntPtr nativeCtx, Tensor output) = this.context.tcnnModule.forward(input, parameters);
            this.context.saveForBackward(new List<Tensor> { input, output });
            this.context.nativeCtx = nativeCtx;
            return output;
        }

        public Tensor Backward(float gradScale, Tensor parameters)
        {

            Tensor output = this.context.savedTensors[1];
            Tensor outputGrad = output.grad() ?? torch.empty(0);
            if (outputGrad.numel() == 0L)
            {
                throw new Exception("output has no grad");
            }
            else if (!outputGrad.is_cuda)
            {
                Console.WriteLine("outputGrad must be a CUDA Tensor");
                outputGrad = outputGrad.cuda();
            }
            Tensor input = this.context.savedTensors[0];
            Tensor paramsGrad;

            using (torch.no_grad())
            {
                Tensor scaledGrad = outputGrad * this.context.lossScale;


                paramsGrad = this.context.tcnnModule.backward(this.context.nativeCtx, input, parameters, output, scaledGrad);

                //paramsGrad = (paramsGrad.numel() == 0L) ? paramsGrad : paramsGrad / gradScale;
                paramsGrad = (paramsGrad.numel() == 0L) ? paramsGrad : paramsGrad / this.context.lossScale / gradScale;

            }
            return paramsGrad;
        }

    }



}