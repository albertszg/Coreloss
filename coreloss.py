import torch
import torch.nn as nn
import numpy as np
class AdaWeightedLoss(nn.Module):
    '''
    strategy: how fast the coefficient w2 shrink to 1.0
    strategy: coefficients with increase the of epoch from 0->1 ***Parameter:['exp' 'linear' 'log' 'nlog' 'quadratic']
    hard_ratio: sample_no_updating/sample_updating
    direct: if use w1 directly
    '''
    def __init__(self, strategy='linear', temp=1.0,hard_ratio=0.5,direct=False):
        super(AdaWeightedLoss, self).__init__()
        self.strategy = strategy
        self.loss_function='MSE'#maybe MAE

        self.u = 0.0
        self.sigma = 1.0
        # smoothness control
        self.temp = temp
        # will stop gradient!
        self.hard_ratio=hard_ratio
        self.direct=direct

        #moving average mode----EMA is waiting for implemented!
        self.MA='MA'
        self.alpha=0.1 #ratio for moving average
    def forward(self, input, target, global_step):
        """
        args:
            input: original values, [bsz,seq,x_dim]
            target: reconstructed values
            global_step: training global step, from 1 to +infinity
        return: loss enables backward()
        """
        bsz, x_dim, seq = target.size()
        base = seq * x_dim
        if self.loss_function=='MSE':
            errors = torch.sum((input - target) ** 2, dim=(-1, -2)) / base
        elif self.loss_function=='MAE':
            errors = torch.sum(torch.abs(input - target), dim=(-1, -2)) / base
        else:
            raise Exception('other loss function excpet MSE is not implemented!')

        with torch.no_grad():
            U = torch.mean(errors)  #[]
            Sigma = torch.std(errors)+ 1e-6 #[]

            self.u = self.alpha * U + (1.0 - self.alpha) * self.u
            self.sigma = self.alpha * Sigma + (1.0 - self.alpha) * self.sigma

            z_score = torch.abs(errors - self.u)/ self.sigma  # 以样本中心为加权中心
            w1 = torch.softmax(-self.temp*z_score, dim=-1)
            w1 = w1 * (1.0 / torch.mean(w1))
            _, indices = w1.topk(int(bsz * self.hard_ratio), largest=False)# 取出最小的 self.hard 比例的值
            w1[indices] = 0.0


            step_coeff = torch.ones(bsz, dtype=target.dtype, device=target.device)
            if self.strategy == 'exp':
                step_coeff *= np.exp((global_step-1)/10)#/10.0
            elif self.strategy == 'log':
                step_coeff *= np.log(global_step + np.e - 1)
            elif self.strategy == 'linear':#10epoch=2100
                step_coeff *= global_step
            elif self.strategy == 'nlog':
                step_coeff *= global_step * np.log(global_step + np.e - 1)
            elif self.strategy == 'quadratic':
                step_coeff *= (global_step ** 2)
            else:
                raise KeyError('Decay function must be one of [\'exp\',\'log\',\'linear\',\'nlog\',\'quadratic\']')

            # if global_step%100==0:
            #     if torch.is_tensor(label) == 1:
            #         logging.info(w1)
                # logging.info(label)
                # logging.info('mean of w1: {}  '.format(torch.mean(w1)))
                # logging.info('global step: {}, step_coeff: {}'.format(global_step,step_coeff))
            if not self.direct:
                w = (1.0 + (step_coeff - 1.0) * w1) / step_coeff
            else:
                w = w1

        loss=errors*w
        return torch.mean(loss)