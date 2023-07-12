using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace InstantNeRF
{
    public static class Metrics
    {
        public static Tensor MSELoss(Tensor x, Tensor y)
        {
            Loss<Tensor, Tensor, Tensor> mse = torch.nn.MSELoss(reduction: nn.Reduction.None);
            return mse.call(x, y);
        }
        public static Tensor PSNR(Tensor x, Tensor y) 
        {
            Tensor mseLoss = MSELoss(x, y);
            return -10f * torch.log(mseLoss) / torch.log(torch.tensor(new float[] { 10f }).to(mseLoss.device));
        }

        public static Tensor HuberLoss(Tensor x, Tensor y, float delta = 0.1f, nn.Reduction reduction = nn.Reduction.Sum) 
        {
            Loss<Tensor, Tensor, Tensor> huber = torch.nn.HuberLoss(delta: delta, reduction: reduction);
            
            return huber.call(x, y);
        }
    }
}
