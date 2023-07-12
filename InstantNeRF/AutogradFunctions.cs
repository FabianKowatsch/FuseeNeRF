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
            public ModuleFunction() { }

            private AutogradContext? context;
            public Tensor Forward(Tensor input, Tensor parameters)
            {
                if (this.context != null)
                {
                    (IntPtr nativeCtx, Tensor output) = this.context.tcnnModule.forward(input, parameters);
                    this.context.saveForBackward(new List<Tensor> { input, parameters, output });
                    this.context.nativeCtx = nativeCtx;
                    return output;
                }
                else
                {
                    throw new Exception("context not initialized");
                }
            }

            public void Backward(float gradScale)
            {
                if (this.context != null)
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


                    (Tensor inputGrad, Tensor paramsGrad) = new ModuleFunctionBackward().Apply(this.context, outputGrad, input, parameters, output);

                    input.backward(new List<Tensor> { (inputGrad).nan_to_num() });

                    if(parameters.numel() > 0)
                        parameters.backward(new List<Tensor> { (paramsGrad / gradScale).nan_to_num() });


                }
                else
                {
                    throw new Exception("must run forward pass first by using apply");
                }
            }
            public Tensor ApplyFwd(ModuleWrapper tcnnModule, Tensor input, Tensor parameters, float lossScale)
            {
                this.context = new AutogradContext(tcnnModule, lossScale);
                Tensor output = Forward(input, parameters);
                return output;
            }

        }

        class ModuleFunctionBackward
        {
            public ModuleFunctionBackward() { }

            private AutogradContext? context;
            public (Tensor inputGrad, Tensor paramsGrad) Forward(Tensor outputGrad, Tensor input, Tensor parameters, Tensor output)
            {
                if (this.context != null)
                {
                    this.context.saveForBackward(new List<Tensor> { input, parameters, outputGrad });

                    if (this.context.forwardCtx != null && this.context.forwardCtx.nativeCtx != IntPtr.Zero)
                    {
                        Tensor inputGrad;
                        Tensor paramsGrad;
                        using (torch.no_grad())
                        {
                            Tensor scaledGrad = outputGrad * this.context.forwardCtx.lossScale;

                            (inputGrad, paramsGrad) = this.context.forwardCtx.tcnnModule.backward(this.context.forwardCtx.nativeCtx, input, parameters, output, scaledGrad);

                            inputGrad = (inputGrad.numel() == 0L) ? inputGrad : inputGrad / this.context.forwardCtx.lossScale;
                            paramsGrad = (paramsGrad.numel() == 0L) ? paramsGrad : paramsGrad / this.context.forwardCtx.lossScale;


                        }
                        return (inputGrad, paramsGrad);
                    }
                    else
                    {
                        throw new Exception("Wrong context");
                    }
                }
                else
                {
                    throw new Exception("context not initialized");
                }
            }

            public void Backward(Tensor inputGradDl, Tensor paramsGradDl)
            {
                if (this.context != null && this.context.forwardCtx != null && this.context.forwardCtx.nativeCtx != IntPtr.Zero)
                {
                    Tensor input = this.context.savedTensors[0];
                    Tensor parameters = this.context.savedTensors[1];
                    Tensor outputGrad = this.context.savedTensors[2];

                    using (torch.enable_grad())
                    {
                        outputGrad = outputGrad * this.context.forwardCtx.lossScale;
                    }
                    Tensor inputGrad;
                    Tensor outputGradDl;
                    Tensor paramsGrad;
                    using (torch.no_grad())
                    {
                        ModuleWrapper tcnnModule = this.context.forwardCtx.tcnnModule;
                        (outputGradDl, paramsGrad, inputGrad) = tcnnModule.backwardBackwardInput(
                            this.context.forwardCtx.nativeCtx,
                            input,
                            parameters,
                            inputGradDl,
                            outputGrad
                            );
                        inputGrad = (inputGrad.numel() == 0L) ? inputGrad : inputGrad / this.context.forwardCtx.lossScale;
                        paramsGrad = (paramsGrad.numel() == 0L) ? paramsGrad : paramsGrad / this.context.forwardCtx.lossScale;

                    }

                    input.backward(grad_tensors: new List<Tensor> { inputGrad });
                    outputGrad.backward(grad_tensors: new List<Tensor> { outputGradDl });
                    parameters.backward(grad_tensors: new List<Tensor> { paramsGrad });
                }
                else
                {
                    throw new Exception("Wrong context");
                }
            }
            public (Tensor inputGrad, Tensor paramsGrad) Apply(AutogradContext forwardCtx, Tensor gradOutput, Tensor input, Tensor parameters, Tensor output)
            {
                this.context = new AutogradContext(forwardCtx.tcnnModule, forwardCtx.lossScale);
                this.context.forwardCtx = forwardCtx;
                (Tensor inputGrad, Tensor paramsGrad) = Forward(gradOutput, input, parameters, output);
                return (inputGrad, paramsGrad);
            }

        }

        public class TruncExp
        {
            private BasicContext context;

            public TruncExp()
            {
                this.context = new BasicContext();
            }
            public Tensor forward(Tensor x)
            {
                x = FloatTensor(x);
                Tensor leaf = x.clone().detach().requires_grad_(true);
                context.saveForBackward(new List<Tensor> { leaf});
                return torch.exp(leaf);
            }
            public void backward()
            {
                Tensor leaf = context.savedTensors[0];
                Tensor xGrad = leaf.grad() ?? torch.zeros_like(leaf);
                leaf.backward(new List<Tensor>() { xGrad * torch.exp(leaf.clamp(-15, 15)) });
            }
        }


    }
}