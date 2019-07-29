import os

from ignite._utils import convert_tensor
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.utils import make_grid
import statistics

from src.experiments.base import BaseExperiment
from src.utils._torch import colorize
from src.utils.misc import init_checkpoint_handler, make_basedir
from src.utils.writers import init_writers

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.utils.fid.fid as metrics
from src.utils.sacred import CustomInterrupt
from src.utils.fid.inception import InceptionV3
from src.utils.fvd import fvd_interface as fvd

class UnisstBaseExperiment(BaseExperiment):
    def __init__(
        self, 
        gen, dis_img, dis_vid, 
        corruption,
        train, test=None, val=None,
        optim_gen=None, optim_dis=None,
        sacred_run=None, writers=None, root=None, nepoch=10, niter=300, 
        display_frequency=1, num_dis_step:int = 1, device='cuda:0', fid_fvd: bool = True,
        colorize=False, **kwargs):
        super().__init__(**kwargs)

        self.train = train
        self.test = test
        self.val = val
        self.sacred_run = sacred_run
        self.sacred_run.result = float('Inf')
        self.device = device
        self.nepoch = nepoch
        self.display_frequency = display_frequency
        self.colorize = colorize

        if isinstance(niter, str) and niter.find('epoch') > 0:
            nepoch = int(niter.split(' ')[0])
            niter = nepoch * len(train)
        self.niter = niter

        self.fid_fvd = fid_fvd
        if root is not None:
            self.basedir = os.path.join(root, str(sacred_run._id))
        else:
            writers = None
            checkpoint = None

        if writers is not None:
            self.writers = init_writers(*writers, sacred_run=sacred_run, dirname=self.basedir)
        else:
            self.writers = None

        if checkpoint is not None:
            self.checkpoint = init_checkpoint_handler(dirname=self.basedir, **checkpoint)

        self.trainer = Engine(self.train_step)
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self.evaluate)
        self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self.log)

        self.tester = Engine(self.test_step)
        self.evaluator = Engine(self.test_step)

        self.gen = gen.to(self.device)
        self.dis_img = dis_img.to(self.device)
        self.dis_vid = dis_vid.to(self.device)

        self.optim_gen = optim_gen
        self.optim_dis = optim_dis

        self.scheduler_gen = ExponentialLR(self.optim_gen, gamma=0.99)
        self.scheduler_dis = ExponentialLR(self.optim_dis, gamma=0.99)

        self.corruption = corruption

        self.num_dis_step = num_dis_step

        RunningAverage(alpha=0.9, output_transform=lambda x: x['loss_gen'].item()).attach(self.trainer, 'loss_gen')
        RunningAverage(alpha=0.9, output_transform=lambda x: x['loss_dis_img'].item()).attach(self.trainer, 'loss_dis_img')
        RunningAverage(alpha=0.9, output_transform=lambda x: x['loss_dis_vid'].item()).attach(self.trainer, 'loss_dis_vid')

        if self.fid_fvd:
            self.dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
            self.model = InceptionV3([block_idx]).to('cuda')
            self.fid_score = float('inf')

    def train_step(self, engine, batch):
        self.training()
        batch = convert_tensor(batch, self.device)
        loss_gen, loss_dis_img, loss_dis_vid, output = self.forward_backward(**batch)
        metric = self.metric(**output, **batch)

        loss = {
            'loss_gen': loss_gen,
            'loss_dis_img': loss_dis_img,
            'loss_dis_vid': loss_dis_vid,
        }
        return {
            **batch,
            **output,
            **loss,
            **metric,
        }

    def test_step(self, engine, batch):
        self.evaluating()
        with torch.no_grad():
            batch = convert_tensor(batch, self.device)
            _, _, _, output = self.forward_backward(backward=False, **batch)
            metric = self.metric(**output, **batch)
            
            return {
                **batch,
                **output,
                **metric,
            }

    def evaluate(self, engine):
        iteration = engine.state.iteration
        if iteration % self.niter == 0:
            self.step(engine, iteration)
            
            if self.val is not None:
                self.evaluator.run(self.val, max_epochs=1)
                self.step(self.evaluator, iteration, dataset_name='val')
                columns = self.evaluator.state.metrics.keys()
                values = [value for value in self.evaluator.state.metrics.values()]
                message = 'Val: '
                for name, value in zip(columns, values):
                    message += ' | {name}: {value:.4f}, {std:.4f}'.format(name=name, value=statistics.mean(value), std=statistics.stdev(value))
                print(message)

            if self.test is not None:
                self.tester.run(self.test, max_epochs=1)
                self.step(self.tester, iteration, dataset_name='test')
                columns = self.tester.state.metrics.keys()
                values = [value for value in self.tester.state.metrics.values()]
                message = 'Test: '
                for name, value in zip(columns, values):
                    message += ' | {name}: {value:.4f}, {std:.4f}'.format(name=name, value=statistics.mean(value), std=statistics.stdev(value))
                print(message)

    def log(self, engine):
        iter = engine.state.iteration
        if iter % self.display_frequency == 0:
            columns = engine.state.metrics.keys()
            values = [value for value in engine.state.metrics.values()]
            message = '[{epoch}/{max_epoch}][{i}/{max_i}]'.format(epoch=engine.state.epoch,
                                                                  max_epoch=self.nepoch,
                                                                  i=(engine.state.iteration % len(self.train)),
                                                                  max_i=len(self.train))
            for name, value in zip(columns, values):
                message += ' | {name}: {value:.4f}'.format(name=name, value=value)
            print(message)

    def write(self, engine, dataset_name):
        iteration = self.trainer.state.iteration

        # Logging Images
        o = engine.state.output
        b = engine.state.batch

        img_tensor, nrow = self.get_tensor(o, b)

        img = make_grid(
            img_tensor,
                nrow=nrow, normalize=True, range=(-1, 1)
        )

        try:
            self.writers.add_image(dataset_name, img, iteration)
        except:
            print('IMPOSSIBLE TO SAVE')

    def forward(self, **kwargs):
        raise NotImplementedError

    def backward_gen(self, **kwargs):
        raise NotImplementedError

    def backward_dis(self, **kwargs):
        raise NotImplementedError

    def get_tensor(self, o, b, limit=2):

        batch_size, nc, seq_len, height, width = b['x'].shape
        if batch_size < limit:
            limit = batch_size

        x       = b['x'][:limit].cpu().permute(0, 2, 1, 3, 4).contiguous().view(-1, nc, height, width)
        y       = o['y'][:limit].cpu().permute(0, 2, 1, 3, 4).contiguous().view(-1, nc, height, width)
        x_hat   = o['x_hat'][:limit].cpu().permute(0, 2, 1, 3, 4).contiguous().view(-1, nc, height, width)

        if self.colorize:
            mask  = b['mask'][:limit].cpu().permute(0, 2, 1, 3, 4).contiguous().view(-1, nc, height, width)
            
            x = colorize(x)
            y = colorize(y)
            x_hat = colorize(x_hat)

            y[mask.expand_as(y)] = 1

        list_tensor = [x, y, x_hat]
        nrow = seq_len
        return torch.cat(list_tensor), nrow

    def step(self, engine, iteration, dataset_name='train'):
        values = {c: value for value, c in zip(engine.state.metrics.values(), engine.state.metrics.keys())}

        if dataset_name == 'train':
            self.scheduler_gen.step()
            self.scheduler_dis.step()

            if values['loss_dis_img'] + values['loss_dis_vid'] < 0.001:
                raise CustomInterrupt('DIS_TOO_SMALL')
            if values['loss_gen'] < 0.001:
                raise CustomInterrupt('GEN_TOO_SMALL')
            if values['recon_mae'] > 1.9:
                raise CustomInterrupt('RECON_TOO_HIGH')

        metrics = engine.state.metrics
        if self.writers is not None:
            for name, value in metrics.items():
                metric_name = dataset_name + '/' + name
                if dataset_name in ['test', 'val']:
                    m = statistics.mean(value)
                    s = statistics.stdev(value)
                    self.writers.add_scalar(metric_name, m, iteration)
                    self.writers.add_scalar(metric_name + '_std', s, iteration)
                else:
                    self.writers.add_scalar(metric_name, value, iteration)

        print(f"saving {iteration}")
        self.write(engine, dataset_name)
        if iteration % 2 * self.niter == 0 and self.fid_fvd:
            self.compute_fid(iteration, dataset_name)
            self.compute_fvd(iteration, dataset_name)


    def compute_fvd(self, iteration, dataset_name):
        self.evaluating()
        fake_list, real_list = [], []

        if dataset_name == 'train':
            dataset = self.train
        elif dataset_name == 'val':
            dataset = self.val
        else:
            dataset = self.test

        with torch.no_grad():
            for i, batch in enumerate(dataset):
                batch = convert_tensor(batch, self.device)

                output = self.forward_backward(**batch, backward=False)[-1]
                real_seq_len = batch['seq_len']
                batch_size, nc, _, _, _ = batch['x'].shape
                x_hat        = output['x_hat']
                x            = batch['x']
                # B x C x T x H x W
                if nc != 3:
                    fake = x_hat.repeat(1, 3, 1, 1, 1)
                    true = x.repeat(1, 3, 1, 1, 1)

                fake_list.append(fake.cpu())
                real_list.append(true.cpu())
                if i == 15:
                    break

        fake_vid = torch.cat(fake_list, dim=0)
        real_vid = torch.cat(real_list, dim=0)
        
        fvd_score = fvd(real_vid, fake_vid)

        print(f"FVD_{dataset_name} : {fvd_score}")
        if self.writers is not None:
            self.writers.add_scalar(f'FVD_{dataset_name}', fvd_score, iteration)

    def compute_fid(self, iteration, dataset_name):
        self.evaluating()
        fake_list, real_list = [], []

        if dataset_name == 'train':
            dataset = self.train
        elif dataset_name == 'val':
            dataset = self.val
        else:
            dataset = self.test

        with torch.no_grad():
            for i, batch in enumerate(dataset):
                batch = convert_tensor(batch, self.device)

                output = self.forward_backward(**batch, backward=False)[-1]
                real_seq_len = batch['seq_len']
                batch_size, nc, _, _, _ = batch['x'].shape
                x_hat        = output['x_hat']
                x            = batch['x']
                
                fake = []
                true = []
                for bi in range(batch_size):
                    fake.append(torch.narrow(x_hat[bi], 1, 0, real_seq_len[bi]).permute(1, 0, 2, 3))
                    true.append(torch.narrow(    x[bi], 1, 0, real_seq_len[bi]).permute(1, 0, 2, 3))

                fake = torch.cat(fake, dim=0)
                true = torch.cat(true, dim=0)

                if nc != 3:
                    fake = fake.repeat(1, 3, 1, 1)
                    true = true.repeat(1, 3, 1, 1)

                fake_list.append((fake.cpu().numpy() + 1.0) / 2.0)
                real_list.append((true.cpu().numpy() + 1.0) / 2.0)

                if i == 15:
                    break

        fake_images = np.concatenate(fake_list)
        real_images = np.concatenate(real_list)
        mu_fake, sigma_fake = metrics.calculate_activation_statistics(
            fake_images, self.model, self.train.batch_size, device=self.device
        )
        mu_real, sigma_real = metrics.calculate_activation_statistics(
            real_images, self.model, self.train.batch_size, device=self.device
        )
        fid_score = metrics.calculate_frechet_distance(
            mu_fake, sigma_fake, mu_real, sigma_real
        )
        print(f"FID_{dataset_name} : {fid_score}")
        if self.writers is not None:
            self.writers.add_scalar(f'FID_{dataset_name}', fid_score, iteration)

    def run(self):
        self.trainer.run(self.train, max_epochs=self.nepoch)
