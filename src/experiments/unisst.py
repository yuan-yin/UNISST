import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite._utils import convert_tensor
from ignite.metrics import RunningAverage

import src.utils.fid.fid as metrics
from src.modules.losses import GANLoss
from src.utils._torch import set_requires_grad
from src.utils.fid.inception import InceptionV3
from src.utils.net import get_z_random
from src.utils.sacred import CustomInterrupt
from src.experiments.unisst_base import UnisstBaseExperiment

from ignite.metrics import Metric
from ignite.engine import Events

class GlobalAverage(Metric):
    def __init__(self, src=None, output_transform=None):
        if not (isinstance(src, Metric) or src is None):
            raise TypeError("Argument src should be a Metric or None.")

        if isinstance(src, Metric):
            if output_transform is not None:
                raise ValueError("Argument output_transform should be None if src is a Metric.")
            self.src = src
            self._get_src_value = self._get_metric_value
            self.iteration_completed = self._metric_iteration_completed
        else:
            if output_transform is None:
                raise ValueError("Argument output_transform should not be None if src corresponds "
                                 "to the output of process function.")
            self._get_src_value = self._get_output_value
            self.update = self._output_update

        super().__init__(output_transform=output_transform)

    def reset(self):
        self._value = []

    def update(self, output):
        # Implement abstract method
        pass

    def compute(self):
        self._value.append(self._get_src_value())
        return self._value

    def attach(self, engine, name):
        # restart average every epoch
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        # compute metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        # apply running average
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)

    def _get_metric_value(self):
        return self.src.compute()

    def _get_output_value(self):
        return self.src

    def _metric_iteration_completed(self, engine):
        self.src.started(engine)
        self.src.iteration_completed(engine)

    def _output_update(self, output):
        self.src = output

class UNISST(UnisstBaseExperiment):

    def __init__(self, use_l1=True, type_gan='vanilla', **kwargs):
        super().__init__(**kwargs)
        self.prior_loss = GANLoss(type_gan).to(self.device)

        RunningAverage(alpha=0.9, output_transform=lambda x: x['recon_mae'].item()).attach(self.trainer, 'recon_mae')
        RunningAverage(alpha=0.9, output_transform=lambda x: x['recon_rmse'].item()).attach(self.trainer, 'recon_rmse')

        if self.test is not None:
            GlobalAverage(output_transform=lambda x: x['recon_mae'].item()).attach(self.tester, 'recon_mae')
            GlobalAverage(output_transform=lambda x: x['recon_rmse'].item()).attach(self.tester, 'recon_rmse')

        if self.val is not None:
            GlobalAverage(output_transform=lambda x: x['recon_mae'].item()).attach(self.evaluator, 'recon_mae')
            GlobalAverage(output_transform=lambda x: x['recon_rmse'].item()).attach(self.evaluator, 'recon_rmse')
        
        self.recon_loss_func_a = nn.L1Loss(reduction='none')
        self.recon_loss_func_b = nn.MSELoss(reduction='none')
    
    def forward_backward(self, x, y, theta, mask, seq_len, backward=True, **kwargs):
        mask_          = torch.zeros_like(mask, dtype=torch.uint8)

        batch = {
            'y':       y,
            'mask': mask,
            'real_seq_len': seq_len,
            }
        output_frame = self.forward(**batch)
        mask_  = output_frame['mask_'].detach()

        loss_gen = self.backward_gen(**batch, **output_frame)

        if backward:
            self.optim_gen.zero_grad()
            loss_gen.backward()
            self.optim_gen.step()

        for _ in range(self.num_dis_step):
            loss_dis_vid, loss_dis_img = self.backward_dis(**batch, **output_frame)
                
            if backward:
                self.optim_dis.zero_grad()
                (loss_dis_img + loss_dis_vid).backward()
                self.optim_dis.step()

        return loss_gen , loss_dis_img, loss_dis_vid, {
            'mask_': mask_.detach(),
            'x_hat': output_frame.pop('x_hat'),
            'y_hat': output_frame.pop('y_hat'),
        }

    def forward(self, mask, y, real_seq_len, **kwargs):
        seq_len = y.size(2)
        mask = mask.float()

        x_hat = (1 - mask) * y + mask * self.gen(y, mask)

        y_hat_list = []
        mask_list = []
        theta = None
        for t in range(seq_len):
            y_hat, mask_, theta = self.corruption(x_hat[:, :, t], t=t, theta=theta, real_seq_len=real_seq_len, device=self.device) # \hat{y}_t
            y_hat_list.append(y_hat)
            mask_list.append(mask_)
        y_hat = torch.stack(y_hat_list, dim=2)
        mask_ = torch.stack(mask_list, dim=2)

        return  {
            'x_hat':      x_hat,
            'y_hat':      y_hat,
            'mask_':      mask_,
            'theta':      theta,
        }

    def backward_gen(self, mask, mask_, y, x_hat, y_hat, real_seq_len, **kwargs):
        set_requires_grad(self.dis_vid, False)
        set_requires_grad(self.dis_img, False)

        nc = y.size(1)
        h = y.size(3)
        w = y.size(4)

        diff_fake = y_hat[:,:,:-1].clone() - y_hat[:,:,1:].clone()

        pred_fake = self.dis_vid(diff_fake)
        loss_prior = self.prior_loss(pred_fake, True).mean()

        pred_fake_frame = self.dis_img(y_hat.clone().view(-1, nc, h, w))
        loss_prior_frame = self.prior_loss(pred_fake_frame, True).mean()
    
        loss_gen = loss_prior + loss_prior_frame

        return loss_gen

    def backward_dis(self, mask, mask_, y, y_hat, real_seq_len, **kwargs):
        set_requires_grad(self.dis_vid, True)
        set_requires_grad(self.dis_img, True)

        nc = y.size(1)
        h = y.size(3)
        w = y.size(4)

        diff_real = y[:,:,:-1].clone() - y[:,:,1:].clone()

        pred_real = self.dis_vid(diff_real)
        loss_dis_real = self.prior_loss(pred_real, True).mean()

        diff_fake = y_hat[:,:,:-1].detach().clone() - y_hat[:,:,1:].detach().clone()

        pred_fake = self.dis_vid(diff_fake)
        loss_dis_fake = self.prior_loss(pred_fake, False).mean()

        loss_dis = (loss_dis_real + loss_dis_fake) * 0.5

        pred_real_frame = self.dis_img(y.clone().view(-1, nc, h, w))
        loss_dis_real_frame = self.prior_loss(pred_real_frame, True).mean()

        pred_fake_frame = self.dis_img(y_hat.detach().clone().view(-1, nc, h, w))
        loss_dis_fake_frame = self.prior_loss(pred_fake_frame, False).mean()

        loss_dis_frame = (loss_dis_real_frame + loss_dis_fake_frame) * 0.5

        return loss_dis, loss_dis_frame

    def metric(self, x, y, x_hat, y_hat, mask, mask_, seq_len, **kwargs):
        mean = kwargs.pop('mean', None)
        std  = kwargs.pop('std', None)
        
        if mean is not None and std is not None:
            mean = mean.view(-1, 1, 1, 1, 1)
            std = std.view(-1, 1, 1, 1, 1)
            scale_factor = 0.000732444226741791
            
            x       = (      x * std * 3 + mean) * scale_factor
            x_hat   = (  x_hat * std * 3 + mean) * scale_factor

            # Values are multiplied by a scale factor, please refer to the user manuel of SST service
            # http://resources.marine.copernicus.eu/documents/PUM/CMEMS-GLO-PUM-001-024.pdf
        
        if mask.float().sum() != 0:
            reconstuction_error_mae  = (self.recon_loss_func_a(x, x_hat) * mask.float()).sum() / mask.float().sum()
            reconstuction_error_rmse = (self.recon_loss_func_b(x, x_hat) * mask.float()).sum() / mask.float().sum()
        else:
            reconstuction_error_mae  = self.recon_loss_func_a(x, x_hat).sum() * 0.
            reconstuction_error_rmse = self.recon_loss_func_b(x, x_hat).sum() * 0.

        reconstuction_error_rmse = torch.sqrt(reconstuction_error_rmse)

        return {
            'recon_mae':    reconstuction_error_mae,
            'recon_rmse':   reconstuction_error_rmse,
        }


